//! Serialization and deserialization to transfer between Ortex and BinaryBackend
//! [Nx](https://hexdocs.pm/nx) backend.

use crate::constants::*;
use crate::error::OrtexError;
use crate::tensor::{OrtexTensor, TensorResource};
use ndarray::{Array, IxDyn};

use rustler::types::Binary;
use rustler::{Atom, Env, NifResult, ResourceArc};

use ort::ep::{self, ExecutionProviderDispatch};
use ort::session::builder::GraphOptimizationLevel;
use ort::value::{TensorElementType, ValueType};

/// Copies an Erlang binary into an owned array. The length has to be checked up front
/// because the copy itself is unchecked.
fn array_from_binary<T: Copy + Default>(
    bin: &Binary,
    shape: &[usize],
) -> Result<Array<T, IxDyn>, OrtexError> {
    let elements = shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| {
            OrtexError::invalid_input(format!("Tensor shape {shape:?} is too large to index"))
        })?;
    let expected = elements.saturating_mul(std::mem::size_of::<T>());

    if bin.len() != expected {
        return Err(OrtexError::invalid_input(format!(
            "Binary length mismatch for shape {shape:?}: expected {expected} bytes, got {}",
            bin.len()
        )));
    }

    let mut data = vec![T::default(); elements];
    if expected > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(bin.as_ptr(), data.as_mut_ptr() as *mut u8, expected);
        }
    }

    Array::from_shape_vec(IxDyn(shape), data).map_err(OrtexError::invalid_input)
}

/// Given a Binary term, shape, and dtype from the BEAM, constructs an OrtexTensor and
/// returns the reference to be used as an Nx.Backend representation.
///
/// # Example
///
/// ```elixir
/// bin = <<1, 0, 0, 0, 1, 0, 0, 0>>
/// ```
///
/// Create a shape `[2]` u32 OrtexTensor from a binary of 8 bytes
/// ```elixir
/// {:ok, reference} = from_binary(bin, {2}, {:u, 32})
/// ```
pub fn from_binary(
    bin: Binary,
    shape: Vec<usize>,
    dtype_str: String,
    dtype_bits: usize,
) -> Result<ResourceArc<TensorResource>, OrtexError> {
    // `typ` is the element type, `ort_tensor_kind` the matching OrtexTensor variant
    macro_rules! tensor {
        ($typ:ty, $ort_tensor_kind:ident) => {
            Ok(TensorResource::new(OrtexTensor::$ort_tensor_kind(
                array_from_binary::<$typ>(&bin, &shape)?.into_shared(),
            )))
        };
    }

    match (dtype_str.as_str(), dtype_bits) {
        ("bf", 16) => tensor!(half::bf16, bf16),
        ("f", 16) => tensor!(half::f16, f16),
        ("f", 32) => tensor!(f32, f32),
        ("f", 64) => tensor!(f64, f64),
        ("s", 8) => tensor!(i8, s8),
        ("s", 16) => tensor!(i16, s16),
        ("s", 32) => tensor!(i32, s32),
        ("s", 64) => tensor!(i64, s64),
        ("u", 8) => tensor!(u8, u8),
        ("u", 16) => tensor!(u16, u16),
        ("u", 32) => tensor!(u32, u32),
        ("u", 64) => tensor!(u64, u64),
        _ => Err(OrtexError::invalid_input(format!(
            "Unsupported dtype {dtype_str} with {dtype_bits} bits"
        ))),
    }
}

/// Given a reference to an OrtexTensor return the binary representation to be used
/// by the BinaryBackend of Nx.
pub fn to_binary<'a>(
    env: Env<'a>,
    reference: ResourceArc<TensorResource>,
    bits: usize,
    limit: usize,
) -> NifResult<Binary<'a>> {
    // The binary points at the tensor's own memory (enif_make_resource_binary): no copy is
    // made here. For an output straight from ONNX Runtime that memory is ORT's buffer, so
    // a full output costs one copy in the whole run instead of two.
    let bytes = reference.to_bytes().map_err(nif_error)?;

    // `Ortex.Backend.backend_transfer/3` asks for the whole tensor with a limit of 0.
    if limit == 0 {
        return Ok(reference.make_binary(env, |_| bytes));
    }

    let byte_limit = limit.saturating_mul(bits / 8);
    Ok(reference.make_binary(env, |_| &bytes[..byte_limit.min(bytes.len())]))
}

fn nif_error<E: std::fmt::Display>(e: E) -> rustler::Error {
    OrtexError::onnxruntime(e).into()
}

fn provider_name(env: Env, ep: &Atom) -> String {
    ep.to_term(env).atom_to_string().unwrap_or_default()
}

/// True if `:qnn` is among the requested providers.
///
/// QNN cannot go through `map_eps`: on upstream ONNX Runtime builds it is a *plugin*
/// execution provider, selected with the V2 device API (`SessionBuilder::with_devices`)
/// rather than by appending an `ExecutionProviderDispatch`. `model::init` handles it
/// separately, so `map_eps` drops it from the dispatch list.
pub fn wants_qnn(env: Env, eps: &[Atom]) -> bool {
    eps.iter().any(|e| provider_name(env, e) == QNN)
}

/// Takes a vec of Atoms and transforms them into a vec of ExecutionProvider Enums.
///
/// Since ort rc.13 the execution provider structs sit behind cargo features, so an EP
/// that was not compiled in has no type at all. Returning an error naming the feature is
/// what stops `Ortex.load(path, [:cuda])` from quietly running on the CPU instead.
pub fn map_eps(env: Env, eps: Vec<Atom>) -> Result<Vec<ExecutionProviderDispatch>, OrtexError> {
    eps.into_iter()
        .filter(|e| provider_name(env, e) != QNN)
        .map(|provider| {
            let name = provider_name(env, &provider);
            match name.as_str() {
                CPU => Ok(ep::CPU::default().build()),
                CUDA => cuda_ep(provider),
                TENSORRT => tensorrt_ep(provider),
                ACL => acl_ep(provider),
                ONEDNN | DNNL => onednn_ep(provider),
                COREML => coreml_ep(provider),
                DIRECTML => directml_ep(provider),
                ROCM => rocm_ep(provider),
                other => Err(OrtexError::execution_provider(
                    provider,
                    format!(
                        "Unknown execution provider: {}. Expected one of: {}",
                        other,
                        [CPU, CUDA, TENSORRT, ACL, ONEDNN, DNNL, COREML, DIRECTML, ROCM, QNN]
                            .join(", ")
                    ),
                )),
            }
        })
        .collect()
}

/// The one error whose cause and fix are both known: the provider exists in the request,
/// not in this build. `provider` carries the atom the caller asked for, so the BEAM can
/// match on it instead of parsing the message.
fn ep_not_compiled(provider: Atom, name: &str, feature: &str) -> OrtexError {
    OrtexError::execution_provider(
        provider,
        format!(
            "{name} execution provider is not compiled into this NIF. Rebuild with the \
             `{feature}` cargo feature: ORTEX_FEATURES={feature} mix compile (or \
             `features: [\"{feature}\"]` for `config :ortex, Ortex.Native`)"
        ),
    )
}

#[cfg(feature = "cuda")]
fn cuda_ep(_provider: Atom) -> Result<ExecutionProviderDispatch, OrtexError> {
    Ok(ep::CUDA::default().build())
}

#[cfg(not(feature = "cuda"))]
fn cuda_ep(provider: Atom) -> Result<ExecutionProviderDispatch, OrtexError> {
    Err(ep_not_compiled(provider, "CUDA", "cuda"))
}

#[cfg(feature = "tensorrt")]
fn tensorrt_ep(_provider: Atom) -> Result<ExecutionProviderDispatch, OrtexError> {
    Ok(ep::TensorRT::default().build())
}

#[cfg(not(feature = "tensorrt"))]
fn tensorrt_ep(provider: Atom) -> Result<ExecutionProviderDispatch, OrtexError> {
    Err(ep_not_compiled(provider, "TensorRT", "tensorrt"))
}

#[cfg(feature = "acl")]
fn acl_ep(_provider: Atom) -> Result<ExecutionProviderDispatch, OrtexError> {
    Ok(ep::ACL::default().build())
}

#[cfg(not(feature = "acl"))]
fn acl_ep(provider: Atom) -> Result<ExecutionProviderDispatch, OrtexError> {
    Err(ep_not_compiled(provider, "ACL", "acl"))
}

#[cfg(feature = "onednn")]
fn onednn_ep(_provider: Atom) -> Result<ExecutionProviderDispatch, OrtexError> {
    Ok(ep::OneDNN::default().build())
}

#[cfg(not(feature = "onednn"))]
fn onednn_ep(provider: Atom) -> Result<ExecutionProviderDispatch, OrtexError> {
    Err(ep_not_compiled(provider, "oneDNN", "onednn"))
}

#[cfg(feature = "coreml")]
fn coreml_ep(_provider: Atom) -> Result<ExecutionProviderDispatch, OrtexError> {
    Ok(ep::CoreML::default().build())
}

#[cfg(not(feature = "coreml"))]
fn coreml_ep(provider: Atom) -> Result<ExecutionProviderDispatch, OrtexError> {
    Err(ep_not_compiled(provider, "CoreML", "coreml"))
}

#[cfg(feature = "directml")]
fn directml_ep(_provider: Atom) -> Result<ExecutionProviderDispatch, OrtexError> {
    Ok(ep::DirectML::default().build())
}

#[cfg(not(feature = "directml"))]
fn directml_ep(provider: Atom) -> Result<ExecutionProviderDispatch, OrtexError> {
    Err(ep_not_compiled(provider, "DirectML", "directml"))
}

#[cfg(feature = "rocm")]
fn rocm_ep(_provider: Atom) -> Result<ExecutionProviderDispatch, OrtexError> {
    Ok(ep::ROCm::default().build())
}

#[cfg(not(feature = "rocm"))]
fn rocm_ep(provider: Atom) -> Result<ExecutionProviderDispatch, OrtexError> {
    Err(ep_not_compiled(provider, "ROCm", "rocm"))
}

/// Where ONNX Runtime's threads come from.
///
/// ONNX Runtime builds either one pool per session - one intra-op thread per core *per
/// session*, so `sessions:` and the number of models multiply them - or one *global* pool
/// that every session in the process shares (`CreateEnvWithGlobalThreadPools`). The pool
/// belongs to the environment, which is one per process, so this is a process-wide choice
/// read once, from the environment, when the first model is loaded.
#[derive(Debug)]
pub enum PoolMode {
    /// One shared pool. The values are what the caller asked for; `None` keeps ONNX
    /// Runtime's default (one intra-op thread per core, one inter-op thread, spinning on).
    Shared {
        intra: Option<usize>,
        inter: Option<usize>,
        spin: Option<bool>,
    },
    /// One pool per session: the shape before the global pool existed.
    PerSession,
}

/// Reads the pool configuration from variables of the same name, so the parse is testable
/// without touching the real process environment.
fn pool_mode_from(get: impl Fn(&str) -> Option<String>) -> Result<PoolMode, String> {
    fn enabled(key: &str, value: &str) -> Result<bool, String> {
        match value {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            other => Err(format!("{key} must be a boolean, got {other:?}")),
        }
    }

    fn count(key: &str, value: &str) -> Result<usize, String> {
        value
            .parse()
            .map_err(|_| format!("{key} must be a non-negative integer, got {value:?}"))
    }

    fn read<'a>(get: &impl Fn(&str) -> Option<String>, key: &'a str) -> Option<(&'a str, String)> {
        get(key).map(|value| (key, value.trim().to_string()))
    }

    if let Some((key, value)) = read(&get, "ORTEX_GLOBAL_THREAD_POOL") {
        if !enabled(key, &value)? {
            return Ok(PoolMode::PerSession);
        }
    }

    let intra = read(&get, "ORTEX_INTRA_THREADS")
        .map(|(k, v)| count(k, &v))
        .transpose()?;
    let inter = read(&get, "ORTEX_INTER_THREADS")
        .map(|(k, v)| count(k, &v))
        .transpose()?;
    let spin = read(&get, "ORTEX_SPIN_CONTROL")
        .map(|(k, v)| enabled(k, &v))
        .transpose()?;

    Ok(PoolMode::Shared { intra, inter, spin })
}

/// The pool mode for this process. Parsed once: the environment and its pools are built
/// once, so re-reading the variables later would only lie about what is in effect.
pub fn pool_mode() -> Result<&'static PoolMode, OrtexError> {
    static MODE: std::sync::LazyLock<Result<PoolMode, String>> =
        std::sync::LazyLock::new(|| pool_mode_from(|key| std::env::var(key).ok()));

    match &*MODE {
        Ok(mode) => Ok(mode),
        Err(message) => Err(OrtexError::invalid_option(message)),
    }
}

/// The process-wide ONNX Runtime environment, created on first use.
///
/// ort rc.14 made the environment explicit (`ort::init().build()`), and ONNX Runtime
/// still allows exactly one per process. It owns the thread pools and the registered
/// plugin execution providers - including QNN - so it has to outlive every session built
/// from it: a process-lifetime holder, not a field of `OrtexModel`.
static ENVIRONMENT: std::sync::Mutex<Option<ort::environment::Environment>> =
    std::sync::Mutex::new(None);

pub fn environment() -> Result<ort::environment::Environment, OrtexError> {
    let mut guard = ENVIRONMENT.lock().unwrap_or_else(|e| e.into_inner());

    if let Some(environment) = guard.as_ref() {
        return Ok(environment.clone());
    }

    let mut init = ort::init().with_name("ortex");

    match pool_mode()? {
        PoolMode::Shared { intra, inter, spin } => {
            let mut options = ort::environment::GlobalThreadPoolOptions::default();
            if let Some(intra) = *intra {
                options = options.with_intra_threads(intra)?;
            }
            if let Some(inter) = *inter {
                options = options.with_inter_threads(inter)?;
            }
            if let Some(spin) = *spin {
                options = options.with_spin_control(spin)?;
            }

            // Said once per process, at the point the pool is built: a performance default
            // that changes the thread count is not something to switch on silently.
            eprintln!(
                "ortex: ONNX Runtime global thread pool {intra}, {inter}, {spin}; shared by \
                 every session (ORTEX_GLOBAL_THREAD_POOL=0 gives each session its own)",
                intra = match intra {
                    Some(n) => format!("intra_threads={n}"),
                    None => "intra_threads=cores".to_string(),
                },
                inter = match inter {
                    Some(n) => format!("inter_threads={n}"),
                    None => "inter_threads=1".to_string(),
                },
                spin = match spin {
                    Some(true) => "spinning on",
                    Some(false) => "spinning off",
                    None => "spinning on (default)",
                },
            );

            init = init.with_global_thread_pool(options);
        }
        PoolMode::PerSession => eprintln!(
            "ortex: ORTEX_GLOBAL_THREAD_POOL=0, every session builds its own ONNX Runtime \
             thread pools (one intra-op thread per core each)"
        ),
    }

    // A failed build leaves the slot empty so the next load retries: on Nerves the
    // runtime library can be mounted after the first attempt.
    let environment = init.build()?;
    *guard = Some(environment.clone());
    Ok(environment)
}

/// Register the Qualcomm QNN *plugin* execution provider library, once.
///
/// Upstream `libonnxruntime.so` is not built with `--use_qnn`, so appending "QNN" by
/// name fails with "QNN execution provider is not supported in this build." Instead the
/// plugin has to be registered with the environment (ORT >= 1.23
/// `RegisterExecutionProviderLibrary`) and then selected via `with_devices`.
///
/// Only a *successful* registration is cached. A failure is not: on Nerves the library
/// can be mounted after the first `Ortex.load`, and caching that first error would keep
/// every later attempt dead until the OS process restarted.
pub fn register_qnn_library(provider: &str) -> Result<(), OrtexError> {
    static REGISTERED: std::sync::LazyLock<std::sync::Mutex<Option<String>>> =
        std::sync::LazyLock::new(|| std::sync::Mutex::new(None));

    let mut registered = REGISTERED.lock().unwrap_or_else(|e| e.into_inner());

    if let Some(registered) = registered.as_deref() {
        // A second provider_path cannot take effect: the process can hold only one
        // registration per EP name, and unregistering would invalidate live sessions.
        if registered != provider {
            eprintln!(
                "ortex: QNN plugin EP already registered from {registered}; \
                 ignoring requested provider_path {provider}. Restart the VM to change it."
            );
        }
        return Ok(());
    }

    if !std::path::Path::new(provider).exists() {
        return Err(OrtexError::qnn(format!(
            "QNN plugin EP library not found at {provider}"
        )));
    }

    let env = environment()?;
    // rc.15 removed `unregister`, and the registration lives with the environment, which
    // is kept for the lifetime of the process.
    env.register_ep_library("QNN", provider).map_err(|e| {
        OrtexError::qnn(format!(
            "failed to register QNN plugin EP from {provider}: {e}"
        ))
    })?;

    *registered = Some(provider.to_string());
    Ok(())
}

pub fn qnn_provider_path() -> String {
    std::env::var("ORTEX_QNN_PROVIDER_PATH")
        .unwrap_or_else(|_| "/usr/lib/libonnxruntime_providers_qnn.so".to_string())
}

pub fn qnn_backend_path() -> String {
    std::env::var("ORTEX_QNN_BACKEND_PATH").unwrap_or_else(|_| "/usr/lib/libQnnHtp.so".to_string())
}

/// Take an optimization level and returns the
pub fn map_opt_level(opt: i32) -> GraphOptimizationLevel {
    match opt {
        1 => GraphOptimizationLevel::Level1,
        2 => GraphOptimizationLevel::Level2,
        3 => GraphOptimizationLevel::Level3,
        _ => GraphOptimizationLevel::Disable,
    }
}

pub fn is_bool_input(inp: &ValueType) -> bool {
    match inp {
        ValueType::Tensor { ty, .. } => ty == &TensorElementType::Bool,
        ValueType::Map { value, .. } => value == &TensorElementType::Bool,
        ValueType::Sequence(boxed_input) => is_bool_input(boxed_input),
        ValueType::Optional(boxed_input) => is_bool_input(boxed_input),
    }
}
