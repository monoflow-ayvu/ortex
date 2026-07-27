//! Abstractions for creating an ONNX Runtime Session and Environment which can be safely
//! passed to and from the BEAM.

use crate::tensor::OrtexTensor;
use crate::utils::{is_bool_input, map_opt_level};
use std::convert::TryInto;
use std::iter::zip;
use std::sync::Mutex;

use ort::execution_providers::ExecutionProviderDispatch;
use ort::session::builder::SessionBuilder;
use ort::session::{Session, SessionInputValue};
use ort::value::{Outlet, ValueType};
use ort::Error;
use rustler::{Atom, Resource, ResourceArc};

/// Holds the model state which include onnxruntime session and environment. All
/// are threadsafe so this can be called concurrently from the beam.
pub struct OrtexModel {
    pub session: Mutex<Session>,
}

#[rustler::resource_impl(name = "OrtexModel")]
impl Resource for OrtexModel {}

impl std::panic::RefUnwindSafe for OrtexModel {}
impl std::panic::UnwindSafe for OrtexModel {}

/// `(Name, Type, Dimension)` triples describing a session's inputs or outputs.
pub type IoSpec = Vec<(String, String, Option<Vec<i64>>)>;

/// Keys in `qnn_opts` that configure ortex itself rather than the QNN EP, and
/// so must not be forwarded to onnxruntime as provider options.
const RESERVED: &[&str] = &[
    "backend_path",
    "provider_path",
    "trace_path",
    "intra_threads",
    "inter_threads",
    "intra_op_spinning",
    "inter_op_spinning",
];

fn parse_usize(key: &str, value: &str) -> Result<usize, Error> {
    value
        .parse()
        .map_err(|_| Error::new(format!("{key} must be a non-negative integer, got {value:?}")))
}

fn parse_bool(key: &str, value: &str) -> Result<bool, Error> {
    match value {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        other => Err(Error::new(format!("{key} must be a boolean, got {other:?}")))
    }
}

fn lookup(opts: &[(String, String)], key: &str) -> Option<String> {
    opts.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone())
}

/// Destination for ort's `tracing` output and onnxruntime's own VERBOSE log.
///
/// The file is swappable, the subscriber is not: `tracing` allows one global subscriber
/// per process, so installing it around a fixed writer means the first session to run
/// wins and every later `trace_path` is silently ignored.
static TRACE_FILE: Mutex<Option<std::fs::File>> = Mutex::new(None);

struct TraceWriter;

impl std::io::Write for TraceWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match TRACE_FILE.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
            Some(file) => file.write(buf),
            // The subscriber outlives any one trace request; with no file set there is
            // nowhere to put this.
            None => Ok(buf.len())
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match TRACE_FILE.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
            Some(file) => file.flush(),
            None => Ok(())
        }
    }
}

fn tracing_requested(qnn_opts: &[(String, String)]) -> bool {
    lookup(qnn_opts, "trace_path").is_some() || std::env::var("ORTEX_TRACE").is_ok()
}

/// Points tracing at `trace_path` (or `$ORTEX_TRACE`). A NIF's stdout goes to the Nerves
/// console rather than to the caller, so a file is the only way to read EP registration,
/// device selection and per-node placement remotely. Filter with RUST_LOG.
fn init_tracing(qnn_opts: &[(String, String)]) {
    let Some(path) = lookup(qnn_opts, "trace_path").or_else(|| std::env::var("ORTEX_TRACE").ok())
    else {
        return;
    };

    let file = match std::fs::File::create(&path) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("ortex: could not open trace file {path}: {e}");
            return;
        }
    };
    *TRACE_FILE.lock().unwrap_or_else(|e| e.into_inner()) = Some(file);

    let filter =
        tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "debug".into());

    // Harmless once a subscriber exists - see TRACE_FILE.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(|| TraceWriter)
        .with_ansi(false)
        .try_init();

    // Raise onnxruntime's own log level too - the node-placement report, which is
    // what tells you whether QNN claimed the graph, is only emitted at VERBOSE.
    if let Ok(env) = ort::environment::current() {
        env.set_log_level(ort::logging::LogLevel::Verbose);
    }
}

/// QNN is a *plugin* EP on upstream ONNX Runtime builds: it has to be registered with
/// the environment and then selected via the V2 device API, not appended by name (which
/// fails with "QNN execution provider is not supported in this build."). Every failure
/// here is an error rather than a silent CPU fallback - a QNN session that quietly runs
/// on CPU looks identical to a working one apart from being slower.
fn with_qnn(
    builder: SessionBuilder,
    qnn_opts: &[(String, String)],
) -> Result<SessionBuilder, Error> {
    let provider_path =
        lookup(qnn_opts, "provider_path").unwrap_or_else(crate::utils::qnn_provider_path);
    crate::utils::register_qnn_library(&provider_path).map_err(Error::new)?;

    let env = ort::environment::current()?;
    let backend_path =
        lookup(qnn_opts, "backend_path").unwrap_or_else(crate::utils::qnn_backend_path);

    let devices: Vec<_> = env
        .devices()
        .filter(|d| d.ep().is_ok_and(|ep| ep.contains("QNN")))
        .collect();

    if devices.is_empty() {
        let seen: Vec<String> = env
            .devices()
            .map(|d| d.ep().unwrap_or("<unknown>").to_string())
            .collect();
        return Err(Error::new(format!(
            "no QNN device found after registering {}; devices seen: [{}]; backend {}",
            provider_path,
            seen.join(", "),
            backend_path
        )));
    }

    // Provider options must be prefixed with the EP name, e.g.
    // "QNNExecutionProvider.backend_path". A bare "backend_path" is silently ignored -
    // and so is the prefixed one if an unprefixed duplicate is passed alongside it.
    // Either way the EP loads with no backend, claims no nodes, and libQnnHtp.so is
    // never mapped.
    let ep_name = devices[0].ep()?.to_string();
    let mut options = vec![(format!("{ep_name}.backend_path"), backend_path)];

    // Everything else is forwarded as a QNN provider option. htp_arch is effectively
    // required on QCS6490: without it the EP logs "Failed to get HTP arch" and the
    // graph ends up on the CPU at roughly 1/35th the speed.
    for (key, value) in qnn_opts {
        if !RESERVED.contains(&key.as_str()) && !key.starts_with("env.") {
            options.push((format!("{ep_name}.{key}"), value.clone()));
        }
    }

    let extra = std::env::var("ORTEX_QNN_OPTS").unwrap_or_default();
    for kv in extra.split(',').filter(|s| !s.trim().is_empty()) {
        let (key, value) = kv
            .split_once('=')
            .ok_or_else(|| Error::new(format!("ORTEX_QNN_OPTS entry {kv:?} is not k=v")))?;
        options.push((
            format!("{ep_name}.{}", key.trim()),
            value.trim().to_string(),
        ));
    }

    Ok(builder.with_devices(devices, Some(&options))?)
}

pub fn init(
    model_path: String,
    eps: Vec<ExecutionProviderDispatch>,
    use_qnn: bool,
    opt: i32,
    qnn_opts: Vec<(String, String)>,
) -> Result<OrtexModel, Error> {
    // `env.*` opts have to be pushed into the real C environment here: os:putenv has
    // written only Erlang's own table since OTP 21, so nothing System.put_env sets
    // reaches getenv() - including the QNN libraries' own getenv("DSP_LIBRARY_PATH").
    for (key, value) in &qnn_opts {
        if let Some(var) = key.strip_prefix("env.") {
            unsafe { std::env::set_var(var, value) };
        }
    }

    init_tracing(&qnn_opts);

    let mut builder = Session::builder()?
        .with_optimization_level(map_opt_level(opt))?
        .with_execution_providers(eps)?;

    // Raising the *environment* log level is not enough to get the node-placement
    // report - the one line that answers "did QNN actually claim this graph". That
    // is logged through the session logger, so the session's own severity has to be
    // lowered too. Tie it to the trace request: asking for a trace means asking for
    // the diagnosis.
    if tracing_requested(&qnn_opts) {
        builder = builder.with_log_level(ort::logging::LogLevel::Verbose)?;
    }

    // Thread-pool shape. This matters far more than it looks on an accelerator:
    // onnxruntime's intra-op pool descends from Eigen's non-blocking pool, built for
    // CPU graphs of hundreds of few-microsecond kernels where a futex wake (~5-50us)
    // would cost more than the kernel itself, so its workers SPIN before parking.
    // When the whole graph is one offloaded EP node that takes ~32ms, those workers
    // spin for the entire inference with nothing to do. Measured on a Dragon Q6A:
    // ~6 ARM cores at 100%, cpu0 at 89degC, the cpufreq cooling state pinned at 9/9,
    // and throughput decaying 24 -> 20.6 fps as the package throttled - heat produced
    // by threads doing no work, throttling the NPU that was doing the work.
    //
    //   intra_threads=1      run the node on the calling thread; no pool, no spin
    //   intra_op_spinning=0  keep the pool but park immediately instead of spinning
    for (key, value) in &qnn_opts {
        builder = match key.as_str() {
            "intra_threads" => builder.with_intra_threads(parse_usize(key, value)?)?,
            "inter_threads" => builder.with_inter_threads(parse_usize(key, value)?)?,
            "intra_op_spinning" => builder.with_intra_op_spinning(parse_bool(key, value)?)?,
            "inter_op_spinning" => builder.with_inter_op_spinning(parse_bool(key, value)?)?,
            _ => builder
        };
    }

    let mut builder = if use_qnn {
        with_qnn(builder, &qnn_opts)?
    } else {
        builder
    };

    Ok(OrtexModel {
        session: Mutex::new(builder.commit_from_file(model_path)?),
    })
}

/// Returns input/output information about a model. The result is a Tuple of
/// `inputs` and `outputs` with elements of `(Name, Type, Dimension)` where
/// `Dimension` elements of -1 are dynamic.
pub fn show(model: ResourceArc<OrtexModel>) -> (IoSpec, IoSpec) {
    let session = model.session.lock().unwrap_or_else(|e| e.into_inner());

    (describe(session.inputs()), describe(session.outputs()))
}

fn describe(outlets: &[Outlet]) -> IoSpec {
    outlets
        .iter()
        .map(|outlet| {
            let dims = match outlet.dtype() {
                ValueType::Tensor { shape, .. } => Some(shape.to_vec()),
                _ => None,
            };
            (
                outlet.name().to_string(),
                format!("{:#?}", outlet.dtype()),
                dims,
            )
        })
        .collect()
}

/// Runs the model with the given inputs. Returns a vector of tensors. Use `model::show`
/// to see what the model expects for input and output shapes.
pub fn run(
    model: ResourceArc<OrtexModel>,
    inputs: Vec<ResourceArc<OrtexTensor>>,
) -> Result<Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>, Error> {
    let mut session = model.session.lock().unwrap_or_else(|e| e.into_inner());

    let output_names: Vec<String> = session
        .outputs()
        .iter()
        .map(|output| output.name().to_string())
        .collect();

    let mut ortified_inputs: Vec<SessionInputValue> = Vec::new();

    // Scoped so the borrow of the input specs ends before session.run() takes the
    // session mutably.
    {
        let session_inputs = session.inputs();
        if inputs.len() != session_inputs.len() {
            return Err(Error::new(format!(
                "Expected {} input(s), got {}",
                session_inputs.len(),
                inputs.len()
            )));
        }

        for (elixir_input, onnx_input) in zip(inputs, session_inputs) {
            let derefed_input: &OrtexTensor = &elixir_input;
            if is_bool_input(onnx_input.dtype()) {
                // this assumes that the boolean input isn't huge -- we're cloning it twice;
                // once below, once in the try_into()
                let boolified_input = derefed_input.clone().to_bool()?;
                let v: SessionInputValue = (&boolified_input).try_into()?;
                ortified_inputs.push(v);
            } else {
                let v: SessionInputValue = derefed_input.try_into()?;
                ortified_inputs.push(v);
            }
        }
    }

    let outputs = session.run(&ortified_inputs[..])?;
    let mut collected_outputs = Vec::new();

    for output_name in output_names {
        let val = outputs.get(&output_name).ok_or_else(|| {
            Error::new(format!(
                "Expected {output_name} to be in the outputs, but didn't find it"
            ))
        })?;

        // NOTE: try_into impl here will implicitly map bool outputs to u8 outputs
        let ortextensor: OrtexTensor = val.try_into()?;
        let shape = ortextensor.shape();
        let (dtype, bits) = ortextensor.dtype();

        collected_outputs.push((ResourceArc::new(ortextensor), shape, dtype, bits))
    }

    Ok(collected_outputs)
}
