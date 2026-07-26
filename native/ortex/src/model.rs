//! Abstractions for creating an ONNX Runtime Session and Environment which can be safely
//! passed to and from the BEAM.
//!
//! # Examples
//!
//! ```
//! let model = init("./models/resnet50.onnx", vec![])?;
//! let (inputs, outputs) = show(model)?;
//! ```

use crate::tensor::OrtexTensor;
use crate::utils::{is_bool_input, map_opt_level};
use std::convert::TryInto;
use std::iter::zip;
use std::sync::Mutex;

use ort::execution_providers::ExecutionProviderDispatch;
use ort::session::{Session, SessionInputValue};
use ort::Error;
use rustler::{Atom, Resource, ResourceArc};

/// Holds the model state which include onnxruntime session and environment. All
/// are threadsafe so this can be called concurrently from the beam.
pub struct OrtexModel {
    pub session: Mutex<Session>,
}

/// Keys in `qnn_opts` that configure ortex itself rather than the QNN EP, and
/// so must not be forwarded to onnxruntime as provider options.
const RESERVED: [&str; 3] = ["backend_path", "provider_path", "trace_path"];

fn lookup(opts: &[(String, String)], key: &str) -> Option<String> {
    opts.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone())
}

#[rustler::resource_impl(name = "OrtexModel")]
impl Resource for OrtexModel {}

impl std::panic::RefUnwindSafe for OrtexModel {}
impl std::panic::UnwindSafe for OrtexModel {}

/// Creates a model given the path to the model and vector of execution providers.
/// The execution providers are Atoms from Erlang/Elixir.
/// Send ort's `tracing` output, and onnxruntime's own VERBOSE log, to the file
/// named by `ORTEX_TRACE`.
///
/// A NIF's stdout goes to the Nerves console, not to the caller's shell, so a
/// file is the only way to read this remotely. Without it every message ort
/// emits about EP registration, device selection and per-node placement is
/// discarded - which is precisely how a QNN session that quietly ran on the CPU
/// was indistinguishable from one on the NPU. Filter with RUST_LOG.
fn init_tracing_from(qnn_opts: &[(String, String)]) {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();

    let from_opts = qnn_opts
        .iter()
        .find(|(k, _)| k == "trace_path")
        .map(|(_, v)| v.clone());

    ONCE.get_or_init(|| {
        if let Some(path) = from_opts.or_else(|| std::env::var("ORTEX_TRACE").ok()) {
            if let Ok(file) = std::fs::File::create(&path) {
                let filter = tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "debug".into());

                let _ = tracing_subscriber::fmt()
                    .with_env_filter(filter)
                    .with_writer(std::sync::Mutex::new(file))
                    .with_ansi(false)
                    .try_init();

                if let Ok(env) = ort::environment::current() {
                    env.set_log_level(ort::logging::LogLevel::Verbose);
                }
            }
        }
    });
}

pub fn init(
    model_path: String,
    eps: Vec<ExecutionProviderDispatch>,
    use_qnn: bool,
    opt: i32,
    qnn_opts: Vec<(String, String)>,
) -> Result<OrtexModel, Error> {
    // Anything the caller wants visible to getenv() has to be pushed into the
    // real C environment here. Elixir's System.put_env cannot do it: since
    // OTP 21 os:putenv writes Erlang's own environment table and leaves the
    // process `environ` untouched, so neither this crate nor - more importantly
    // - the QNN libraries' own getenv("DSP_LIBRARY_PATH") ever saw it. That is
    // exactly how a QNN session configured from Elixir ended up loading the
    // wrong backend and running silently on the CPU.
    for (key, value) in &qnn_opts {
        if let Some(var) = key.strip_prefix("env.") {
            unsafe { std::env::set_var(var, value) };
        }
    }

    init_tracing_from(&qnn_opts);

    let builder = Session::builder()?
        .with_optimization_level(map_opt_level(opt))?
        .with_execution_providers(eps)?;

    // QNN is a *plugin* EP on upstream ONNX Runtime builds: it has to be
    // registered with the environment and then selected via the V2 device API,
    // not appended by name (which fails with "QNN execution provider is not
    // supported in this build."). Every failure here is an error rather than a
    // silent CPU fallback - a QNN session that quietly runs on CPU looks
    // identical to a working one apart from being slower.
    let builder = if use_qnn {
        let provider_path = lookup(&qnn_opts, "provider_path")
            .unwrap_or_else(crate::utils::qnn_provider_path);
        crate::utils::register_qnn_library(&provider_path).map_err(Error::new)?;

        let env = ort::environment::current()?;
        let backend_path = lookup(&qnn_opts, "backend_path")
            .unwrap_or_else(crate::utils::qnn_backend_path);

        let devices: Vec<_> = env
            .devices()
            .filter(|d| d.ep().map(|ep| ep.contains("QNN")).unwrap_or(false))
            .collect();

        if devices.is_empty() {
            let available: Vec<String> = env
                .devices()
                .map(|d| d.ep().unwrap_or("<unknown>").to_string())
                .collect();
            return Err(Error::new(format!(
                "no QNN device found after registering {}; devices seen: [{}]; backend {}",
                provider_path,
                available.join(", "),
                backend_path
            )));
        }

        // Provider options for with_devices() must be prefixed with the EP name,
        // e.g. "QNNExecutionProvider.backend_path" - ort's own doc example uses
        // "CPUExecutionProvider.use_arena". A bare "backend_path" is silently
        // ignored, and without a backend the EP loads but claims no nodes, so
        // everything runs on CPU while libQnnHtp.so is never even mapped.
        let ep_name = devices[0]
            .ep()
            .map(|s| s.to_string())
            .unwrap_or_else(|_| "QNNExecutionProvider".to_string());

        // Exactly one entry for the backend, prefixed with the EP name. An
        // additional unprefixed "backend_path" used to be passed here as
        // belt-and-braces; it is not harmless - with it present QNN takes no
        // nodes and the graph silently runs on the CPU.
        let mut options = vec![(format!("{ep_name}.backend_path"), backend_path.clone())];

        // Extra QNN provider options, comma-separated k=v, from ORTEX_QNN_OPTS.
        // `htp_arch` in particular is effectively required on QCS6490: without
        // it the EP logs "Unable to get platform info: Failed to get HTP arch",
        // claims no nodes, and the whole graph silently runs on the CPU at
        // roughly 1/35th the speed. Others worth knowing:
        // htp_performance_mode, htp_graph_finalization_optimization_mode,
        // enable_htp_shared_memory_allocator, profiling_level.
        for (key, value) in &qnn_opts {
            if !RESERVED.contains(&key.as_str()) && !key.starts_with("env.") {
                options.push((format!("{ep_name}.{key}"), value.clone()));
            }
        }

        if let Ok(extra) = std::env::var("ORTEX_QNN_OPTS") {
            for kv in extra.split(',').filter(|s| !s.trim().is_empty()) {
                match kv.split_once('=') {
                    Some((k, v)) => {
                        options.push((format!("{ep_name}.{}", k.trim()), v.trim().to_string()))
                    }
                    None => {
                        return Err(Error::new(format!(
                            "ORTEX_QNN_OPTS entry {kv:?} is not k=v"
                        )))
                    }
                }
            }
        }

        builder.with_devices(devices, Some(&options))?
    } else {
        builder
    };

    let mut builder = builder;
    let session = builder.commit_from_file(model_path)?;

    let state = OrtexModel {
        session: Mutex::new(session),
    };
    Ok(state)
}

/// Returns input/output information about a model. The result is a Tuple of
/// `inputs` and `outputs` with elements of `(Name, Type, Dimension)` where
/// `Dimension` elements of -1 are dynamic.
pub fn show(
    model: ResourceArc<OrtexModel>,
) -> (
    Vec<(String, String, Option<Vec<i64>>)>,
    Vec<(String, String, Option<Vec<i64>>)>,
) {
    let model: &OrtexModel = &*model;

    let session = model.session.lock().unwrap_or_else(|e| e.into_inner());

    let mut inputs = Vec::new();
    for input in session.inputs() {
        let name = input.name().to_string();
        let repr = format!("{:#?}", input.dtype());
        let dims = match input.dtype() {
            ort::value::ValueType::Tensor { shape, .. } => Some(shape.to_vec()),
            _ => None,
        };
        inputs.push((name, repr, dims));
    }

    let mut outputs = Vec::new();
    for output in session.outputs() {
        let name = output.name().to_string();
        let repr = format!("{:#?}", output.dtype());
        let dims = match output.dtype() {
            ort::value::ValueType::Tensor { shape, .. } => Some(shape.to_vec()),
            _ => None,
        };
        outputs.push((name, repr, dims));
    }

    (inputs, outputs)
}

/// Runs the model with the given inputs. Returns a vector of tensors. Use `model::show`
/// to see what the model expects for input and output shapes.
pub fn run(
    model: ResourceArc<OrtexModel>,
    inputs: Vec<ResourceArc<OrtexTensor>>,
) -> Result<Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>, Error> {
    // Grab the session and run a forward pass with it
    let mut session = model.session.lock().unwrap_or_else(|e| e.into_inner());
    let mut ortified_inputs: Vec<SessionInputValue> = Vec::new();
    let output_names: Vec<String>;

    {
        let session_inputs = session.inputs();
        let expected_inputs = session_inputs.len();
        if inputs.len() != expected_inputs {
            return Err(Error::new(format!(
                "Expected {} input(s), got {}",
                expected_inputs,
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

        output_names = session
            .outputs()
            .iter()
            .map(|output| output.name().to_string())
            .collect();
    }

    // Construct a Vec of ModelOutput enums based on the DynOrtTensor data type
    let outputs = session.run(&ortified_inputs[..])?;
    let mut collected_outputs = Vec::new();

    for output_name in output_names {
        let val = outputs.get(&output_name).ok_or_else(|| {
            Error::new(format!(
                "Expected {} to be in the outputs, but didn't find it",
                output_name
            ))
        })?;

        // NOTE: try_into impl here will implicitly map bool outputs to u8 outputs
        let ortextensor: OrtexTensor = val.try_into()?;
        let shape = ortextensor.shape();
        let (dtype, bits) = ortextensor.dtype();

        let collected_output = (ResourceArc::new(ortextensor), shape, dtype, bits);
        collected_outputs.push(collected_output)
    }

    Ok(collected_outputs)
}
