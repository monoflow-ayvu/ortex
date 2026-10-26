//! Abstractions for creating an ONNX Runtime Session and Environment which can be safely
//! passed to and from the BEAM.

use crate::error::OrtexError;
use crate::tensor::{OrtexTensor, TensorResource};
use crate::utils::{is_bool_input, map_opt_level};
use std::convert::TryInto;
use std::iter::zip;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, TryLockError};

use ort::ep::ExecutionProviderDispatch;
use ort::session::builder::SessionBuilder;
use ort::session::{Session, SessionInputValue, SessionInputs};
use ort::value::{Outlet, TensorElementType, ValueType};
use rustler::{Atom, Resource, ResourceArc};

/// Holds the model state which include onnxruntime sessions and environment. All
/// are threadsafe so this can be called concurrently from the beam.
pub struct OrtexModel {
    /// Ort sessions are exclusive - `Session::run` takes `&mut self` - so real
    /// concurrency comes from a pool of sessions over the *same* graph, not from more
    /// callers. `sessions: N` in `Ortex.load/4` sizes the pool; 1 is the historical
    /// behaviour. The graph is loaded once per session, so the extra memory is one set
    /// of thread pools and per-session allocations per unit of N.
    sessions: Vec<Mutex<Session>>,
    /// Round-robin cursor over `sessions`.
    next: AtomicUsize,
    /// Input/output specs, captured once at load time. They are immutable for the
    /// lifetime of the sessions, so describing a model never has to take a session
    /// lock - which is held for the whole duration of `run`.
    pub io_spec: (IoSpec, IoSpec),
}

/// Takes a session from the pool: round-robin from `next`, preferring a free slot.
///
/// `try_lock` first so a caller never queues behind a busy session while another is idle,
/// and a plain blocking `lock` only when all of them are busy - spinning over the pool
/// would just burn the dirty scheduler. A poisoned mutex is recovered, not propagated: the
/// panic that poisoned it happened in some earlier inference, not in this one.
fn acquire<'a>(sessions: &'a [Mutex<Session>], next: &AtomicUsize) -> MutexGuard<'a, Session> {
    let len = sessions.len();
    let start = next.fetch_add(1, Ordering::Relaxed) % len;

    for offset in 0..len {
        match sessions[(start + offset) % len].try_lock() {
            Ok(guard) => return guard,
            Err(TryLockError::Poisoned(poisoned)) => return poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => {}
        }
    }

    sessions[start].lock().unwrap_or_else(|e| e.into_inner())
}

#[rustler::resource_impl(name = "OrtexModel")]
impl Resource for OrtexModel {}

impl std::panic::RefUnwindSafe for OrtexModel {}
impl std::panic::UnwindSafe for OrtexModel {}

/// `(Name, Type, Dimension)` triples describing a session's inputs or outputs.
///
/// `Type` is the elementary tensor type (`"f32"`, `"i64"`, ...). Dimensions are `None`
/// for non-tensor outlets and per-axis `nil` for a dynamic axis: onnxruntime reports a
/// dynamic dimension as `-1`, which would be indistinguishable from a broken shape.
pub type IoSpec = Vec<(String, String, Option<Vec<Option<i64>>>)>;

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
    "sessions",
];

fn parse_usize(key: &str, value: &str) -> Result<usize, OrtexError> {
    value.parse().map_err(|_| {
        OrtexError::invalid_option(format!(
            "{key} must be a non-negative integer, got {value:?}"
        ))
    })
}

fn parse_bool(key: &str, value: &str) -> Result<bool, OrtexError> {
    match value {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        other => Err(OrtexError::invalid_option(format!(
            "{key} must be a boolean, got {other:?}"
        ))),
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
#[cfg(feature = "trace")]
static TRACE_FILE: Mutex<Option<std::fs::File>> = Mutex::new(None);

#[cfg(feature = "trace")]
struct TraceWriter;

#[cfg(feature = "trace")]
impl std::io::Write for TraceWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match TRACE_FILE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_mut()
        {
            Some(file) => file.write(buf),
            // The subscriber outlives any one trace request; with no file set there is
            // nowhere to put this.
            None => Ok(buf.len()),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match TRACE_FILE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_mut()
        {
            Some(file) => file.flush(),
            None => Ok(()),
        }
    }
}

/// Where `trace_path` / `ORTEX_TRACE` point, if either was set.
fn trace_path(qnn_opts: &[(String, String)]) -> Option<String> {
    lookup(qnn_opts, "trace_path").or_else(|| std::env::var("ORTEX_TRACE").ok())
}

/// True when diagnostics were asked for. This also decides whether onnxruntime's own log
/// level is raised: the node-placement report - the one line that answers "did QNN claim
/// this graph" - is only emitted at VERBOSE, and asking for a trace is asking for it.
#[cfg(feature = "trace")]
fn tracing_requested(qnn_opts: &[(String, String)]) -> bool {
    trace_path(qnn_opts).is_some()
}

/// Without the `trace` feature there is no subscriber to raise, but asking for a trace
/// still has to be visible: a silently empty file is worse than no file.
#[cfg(not(feature = "trace"))]
fn tracing_requested(qnn_opts: &[(String, String)]) -> bool {
    if let Some(path) = trace_path(qnn_opts) {
        eprintln!(
            "ortex: trace requested for {path}, but this NIF was built without the `trace` \
             feature - nothing will be written. Rebuild with ORTEX_FEATURES=trace"
        );
    }
    false
}

/// Points tracing at `trace_path` (or `$ORTEX_TRACE`). A NIF's stdout goes to the Nerves
/// console rather than to the caller, so a file is the only way to read EP registration,
/// device selection and per-node placement remotely. Filter with RUST_LOG.
#[cfg(feature = "trace")]
fn init_tracing(qnn_opts: &[(String, String)]) {
    let Some(path) = trace_path(qnn_opts) else {
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
    if let Ok(env) = crate::utils::environment() {
        env.set_log_level(ort::logging::LogLevel::Verbose);
    }
}

#[cfg(not(feature = "trace"))]
fn init_tracing(_qnn_opts: &[(String, String)]) {}

/// QNN is a *plugin* EP on upstream ONNX Runtime builds: it has to be registered with
/// the environment and then selected via the V2 device API, not appended by name (which
/// fails with "QNN execution provider is not supported in this build."). Every failure
/// here is an error rather than a silent CPU fallback - a QNN session that quietly runs
/// on CPU looks identical to a working one apart from being slower.
fn with_qnn(
    env: &ort::environment::Environment,
    builder: SessionBuilder,
    qnn_opts: &[(String, String)],
) -> Result<SessionBuilder, OrtexError> {
    let provider_path =
        lookup(qnn_opts, "provider_path").unwrap_or_else(crate::utils::qnn_provider_path);
    crate::utils::register_qnn_library(&provider_path)?;

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
        return Err(OrtexError::qnn(format!(
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
        let (key, value) = kv.split_once('=').ok_or_else(|| {
            OrtexError::invalid_option(format!("ORTEX_QNN_OPTS entry {kv:?} is not k=v"))
        })?;
        options.push((
            format!("{ep_name}.{}", key.trim()),
            value.trim().to_string(),
        ));
    }

    builder
        .with_devices(devices, Some(&options))
        .map_err(OrtexError::qnn)
}

pub fn init(
    model_path: String,
    eps: Vec<ExecutionProviderDispatch>,
    use_qnn: bool,
    opt: i32,
    qnn_opts: Vec<(String, String)>,
) -> Result<OrtexModel, OrtexError> {
    // `env.*` opts have to be pushed into the real C environment here: os:putenv has
    // written only Erlang's own table since OTP 21, so nothing System.put_env sets
    // reaches getenv() - including the QNN libraries' own getenv("DSP_LIBRARY_PATH").
    for (key, value) in &qnn_opts {
        if let Some(var) = key.strip_prefix("env.") {
            unsafe { std::env::set_var(var, value) };
        }
    }

    init_tracing(&qnn_opts);

    // `sessions` sizes the pool. It is read here, not forwarded to the QNN EP.
    let session_count = match lookup(&qnn_opts, "sessions") {
        Some(value) => parse_usize("sessions", &value)?,
        None => 1,
    };
    if session_count == 0 {
        return Err(OrtexError::invalid_option(
            "sessions must be at least 1, got 0",
        ));
    }

    // Everything below builds one session; the loop builds `session_count` of them from
    // the same configuration, so the pool is N copies of the *same* graph.
    let mut log_level: Option<ort::logging::LogLevel> = None;

    // Raising the *environment* log level is not enough to get the node-placement
    // report - the one line that answers "did QNN actually claim this graph". That
    // is logged through the session logger, so the session's own severity has to be
    // lowered too. Tie it to the trace request: asking for a trace means asking for
    // the diagnosis.
    if tracing_requested(&qnn_opts) {
        log_level = Some(ort::logging::LogLevel::Verbose);
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
    let mut intra_threads = lookup(&qnn_opts, "intra_threads")
        .map(|value| parse_usize("intra_threads", &value))
        .transpose()?;
    let inter_threads = lookup(&qnn_opts, "inter_threads")
        .map(|value| parse_usize("inter_threads", &value))
        .transpose()?;
    let mut intra_op_spinning = lookup(&qnn_opts, "intra_op_spinning")
        .map(|value| parse_bool("intra_op_spinning", &value))
        .transpose()?;
    let inter_op_spinning = lookup(&qnn_opts, "inter_op_spinning")
        .map(|value| parse_bool("inter_op_spinning", &value))
        .transpose()?;

    // An accelerator EP whose graph is one offloaded node has no CPU work for an intra-op
    // pool to do, so the spinning default is pure heat: the ~24 -> 20.6 fps throttle above.
    // Apply the measured-good defaults only when the caller did not pick a value, and say
    // so once - a silent change to a performance knob is the kind of thing that costs a
    // day to chase later.
    if use_qnn {
        let mut applied: Vec<&str> = Vec::new();
        if intra_threads.is_none() {
            intra_threads = Some(1);
            applied.push("intra_threads=1");
        }
        if intra_op_spinning.is_none() {
            intra_op_spinning = Some(false);
            applied.push("intra_op_spinning=0");
        }
        if !applied.is_empty() {
            eprintln!(
                "ortex: accelerator (QNN) EP active, defaulting {}; override with the \
                 same keys in qnn_opts",
                applied.join(" ")
            );
        }
    }

    // One builder chain, run once per pooled session. `commit_from_file` consumes the
    // builder, so the chain has to live in a closure rather than be reused.
    let environment = crate::utils::environment()?;

    let build_session = || -> Result<Session, OrtexError> {
        let mut builder = Session::builder(&environment)?
            .with_optimization_level(map_opt_level(opt))?
            .with_execution_providers(eps.as_slice())?;

        if let Some(level) = log_level {
            builder = builder.with_log_level(level)?;
        }
        if let Some(value) = intra_threads {
            builder = builder.with_intra_threads(value)?;
        }
        if let Some(value) = inter_threads {
            builder = builder.with_inter_threads(value)?;
        }
        if let Some(value) = intra_op_spinning {
            builder = builder.with_intra_op_spinning(value)?;
        }
        if let Some(value) = inter_op_spinning {
            builder = builder.with_inter_op_spinning(value)?;
        }
        let mut builder = if use_qnn {
            with_qnn(&environment, builder, &qnn_opts)?
        } else {
            builder
        };

        builder
            .commit_from_file(&model_path)
            .map_err(OrtexError::from)
    };

    let mut sessions = Vec::with_capacity(session_count);
    for _ in 0..session_count {
        sessions.push(Mutex::new(build_session()?));
    }

    // The graph is the same in every pooled session, so one spec describes the model.
    let io_spec = {
        let first = sessions[0].lock().unwrap_or_else(|e| e.into_inner());
        (describe(first.inputs()), describe(first.outputs()))
    };

    Ok(OrtexModel {
        sessions,
        next: AtomicUsize::new(0),
        io_spec,
    })
}

/// Returns input/output information about a model. The result is a Tuple of
/// `inputs` and `outputs` with elements of `(Name, Type, Dimension)` where a `nil`
/// dimension is dynamic.
///
/// This reads the spec cached at load time and takes no locks, so inspecting a model
/// never queues behind a running inference.
pub fn show(model: &OrtexModel) -> (IoSpec, IoSpec) {
    model.io_spec.clone()
}

fn describe(outlets: &[Outlet]) -> IoSpec {
    outlets
        .iter()
        .map(|outlet| {
            let (ty, dims) = match outlet.dtype() {
                ValueType::Tensor { ty, shape, .. } => (
                    ty.to_string(),
                    // onnxruntime reports a dynamic axis as -1; the BEAM side documents
                    // nil for it, and -1 also reads as "broken shape".
                    Some(
                        shape
                            .iter()
                            .map(|dim| (*dim >= 0).then_some(*dim))
                            .collect::<Vec<_>>(),
                    ),
                ),
                other => (format!("{other:?}"), None),
            };
            (outlet.name().to_string(), ty, dims)
        })
        .collect()
}

/// One output of `run`: the tensor plus the metadata the BEAM side needs to build an
/// `Nx.Tensor` around it (shape, dtype atom, dtype bits).
pub type RunOutput = (ResourceArc<TensorResource>, Vec<usize>, Atom, usize);

/// How a caller addressed the model's inputs.
pub enum Inputs {
    /// In the model's declared order: `Ortex.run(model, {x, y})`. Fast, and the reason
    /// each tensor is checked against its expected dtype and shape before the run - two
    /// same-typed inputs given in the wrong order used to produce a plausible result.
    Positional(Vec<ResourceArc<TensorResource>>),
    /// By name: `Ortex.run(model, %{"x" => x, "y" => y})`. ONNX Runtime matches these
    /// itself, so input order stops mattering.
    Named(Vec<(String, ResourceArc<TensorResource>)>),
}

impl Inputs {
    fn len(&self) -> usize {
        match self {
            Inputs::Positional(tensors) => tensors.len(),
            Inputs::Named(pairs) => pairs.len(),
        }
    }
}

/// Inputs after conversion, in the shape `Session::run` accepts.
enum OrtInputs<'a> {
    Positional(Vec<SessionInputValue<'a>>),
    Named(Vec<(String, SessionInputValue<'a>)>),
}

/// Rejects a tensor that does not fit the outlet's declared rank, shape or element type.
///
/// ONNX Runtime would also fail on a hard mismatch, but only after the caller has paid for
/// the copy into an ORT value, and its message names the tensor without saying what was
/// expected. Shapes with a dynamic axis (`-1`) are not checked on that axis.
fn validate_input(outlet: &Outlet, tensor: &OrtexTensor) -> Result<(), OrtexError> {
    let ValueType::Tensor { ty, shape, .. } = outlet.dtype() else {
        // Non-tensor inputs are not built by this crate; let ORT describe the problem.
        return Ok(());
    };

    let actual_ty = tensor.element_type();
    // A bool input arrives as 0/1 u8 (how Nx sees booleans) and is rewritten by `run`.
    let bool_from_u8 = *ty == TensorElementType::Bool && actual_ty == TensorElementType::Uint8;

    if &actual_ty != ty && !bool_from_u8 {
        return Err(OrtexError::invalid_input(format!(
            "Input {:?} expects {:?}, got {:?}",
            outlet.name(),
            ty,
            actual_ty
        )));
    }

    let actual_shape = tensor.shape();
    let declared: Vec<Option<usize>> = shape
        .iter()
        .map(|dim| (*dim >= 0).then_some(*dim as usize))
        .collect();

    if declared.len() != actual_shape.len() {
        return Err(OrtexError::invalid_input(format!(
            "Input {:?} expects rank {} (shape {declared:?}), got rank {} (shape {actual_shape:?})",
            outlet.name(),
            declared.len(),
            actual_shape.len()
        )));
    }

    for (axis, (expected, got)) in declared.iter().zip(actual_shape.iter()).enumerate() {
        if let Some(expected) = expected {
            if expected != got {
                return Err(OrtexError::invalid_input(format!(
                    "Input {:?} axis {axis} expects {expected}, got {got} (declared shape \
                     {declared:?}, got {actual_shape:?})",
                    outlet.name()
                )));
            }
        }
    }

    Ok(())
}

/// Turns tensors into session values, substituting the bool rewrite where one was made.
fn convert<'a>(
    tensors: impl Iterator<Item = &'a OrtexTensor>,
    boolified: &'a [Option<OrtexTensor>],
) -> Result<Vec<SessionInputValue<'a>>, OrtexError> {
    tensors
        .enumerate()
        .map(|(index, tensor)| match &boolified[index] {
            Some(converted) => Ok(converted.try_into()?),
            None => Ok(tensor.try_into()?),
        })
        .collect()
}

/// Runs the model with the given inputs. Returns a vector of tensors. Use `model::show`
/// to see what the model expects for input and output shapes.
pub fn run(model: ResourceArc<OrtexModel>, inputs: Inputs) -> Result<Vec<RunOutput>, OrtexError> {
    let mut session = acquire(&model.sessions, &model.next);

    let output_names: Vec<String> = session
        .outputs()
        .iter()
        .map(|output| output.name().to_string())
        .collect();

    // Boolean inputs arrive as 0/1 u8 (that is how Nx sees them) and have to be rewritten
    // into a bool tensor. The rewrite allocates, so its output has to outlive the
    // SessionInputValue that points at it - hence a slot per input, filled before any
    // conversion happens.
    let mut boolified: Vec<Option<OrtexTensor>> = (0..inputs.len()).map(|_| None).collect();

    // Scoped so the borrow of the input specs ends before session.run() takes the
    // session mutably.
    let ort_inputs = {
        let session_inputs = session.inputs();

        match &inputs {
            Inputs::Positional(tensors) => {
                if tensors.len() != session_inputs.len() {
                    return Err(OrtexError::invalid_input(format!(
                        "Expected {} input(s), got {}",
                        session_inputs.len(),
                        tensors.len()
                    )));
                }

                for (index, (tensor, outlet)) in zip(tensors.iter(), session_inputs).enumerate() {
                    validate_input(outlet, tensor)?;
                    if is_bool_input(outlet.dtype()) {
                        boolified[index] = Some(tensor.to_bool()?);
                    }
                }

                OrtInputs::Positional(convert(tensors.iter().map(|t| &***t), &boolified)?)
            }
            Inputs::Named(pairs) => {
                let expected: Vec<String> = session_inputs
                    .iter()
                    .map(|outlet| format!("{:?}", outlet.name()))
                    .collect();

                if pairs.len() != session_inputs.len() {
                    return Err(OrtexError::invalid_input(format!(
                        "Expected {} input(s) ({}), got {}",
                        session_inputs.len(),
                        expected.join(", "),
                        pairs.len()
                    )));
                }

                for (index, (name, tensor)) in pairs.iter().enumerate() {
                    let outlet = session_inputs
                        .iter()
                        .find(|outlet| outlet.name() == name)
                        .ok_or_else(|| {
                            OrtexError::invalid_input(format!(
                                "No input named {name:?}; this model expects {}",
                                expected.join(", ")
                            ))
                        })?;

                    if pairs[..index].iter().any(|(seen, _)| seen == name) {
                        return Err(OrtexError::invalid_input(format!(
                            "Input {name:?} was given twice"
                        )));
                    }

                    validate_input(outlet, tensor)?;
                    if is_bool_input(outlet.dtype()) {
                        boolified[index] = Some(tensor.to_bool()?);
                    }
                }

                let converted = convert(pairs.iter().map(|(_, t)| &***t), &boolified)?;
                OrtInputs::Named(
                    zip(pairs.iter(), converted)
                        .map(|((name, _), value)| (name.clone(), value))
                        .collect(),
                )
            }
        }
    };

    let mut outputs = match ort_inputs {
        OrtInputs::Positional(values) => session.run(&values[..])?,
        OrtInputs::Named(values) => {
            let pairs: Vec<(std::borrow::Cow<'_, str>, SessionInputValue<'_>)> = values
                .into_iter()
                .map(|(name, value)| (std::borrow::Cow::Owned(name), value))
                .collect();
            session.run::<0>(SessionInputs::ValueMap(pairs))?
        }
    };

    // Every output keeps the ORT-owned value instead of copying it into an ndarray:
    // `to_binary` then points the BEAM binary straight at ORT's buffer. Holding it past
    // the session borrow is safe because the value owns its memory (the class of bug
    // rc.15 fixed - "tensors outliving the allocator that created them").
    let mut collected_outputs = Vec::with_capacity(output_names.len());

    for output_name in output_names {
        let value = outputs.remove(&output_name).ok_or_else(|| {
            OrtexError::onnxruntime(format!(
                "Expected {output_name} to be in the outputs, but didn't find it"
            ))
        })?;

        if !matches!(value.dtype(), ValueType::Tensor { .. }) {
            return Err(OrtexError::onnxruntime(format!(
                "Output {output_name} is not a tensor, got {}",
                value.dtype()
            )));
        }

        let ortex_tensor = OrtexTensor::ort(value);
        let shape = ortex_tensor.shape();
        let (dtype, bits) = ortex_tensor.dtype();

        collected_outputs.push((TensorResource::new(ortex_tensor), shape, dtype, bits))
    }

    Ok(collected_outputs)
}
