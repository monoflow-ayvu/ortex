//! Serialization and deserialization to transfer between Ortex and BinaryBackend
//! [Nx](https://hexdocs.pm/nx) backend.

use crate::constants::*;
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
) -> Result<Array<T, IxDyn>, String> {
    let elements = shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| format!("Tensor shape {shape:?} is too large to index"))?;
    let expected = elements.saturating_mul(std::mem::size_of::<T>());

    if bin.len() != expected {
        return Err(format!(
            "Binary length mismatch for shape {shape:?}: expected {expected} bytes, got {}",
            bin.len()
        ));
    }

    let mut data = vec![T::default(); elements];
    if expected > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(bin.as_ptr(), data.as_mut_ptr() as *mut u8, expected);
        }
    }

    Array::from_shape_vec(IxDyn(shape), data).map_err(|e| e.to_string())
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
) -> Result<ResourceArc<TensorResource>, String> {
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
        _ => Err(format!(
            "Unsupported dtype {dtype_str} with {dtype_bits} bits"
        )),
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
    rustler::Error::Term(Box::new(e.to_string()))
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
pub fn map_eps(env: Env, eps: Vec<Atom>) -> Result<Vec<ExecutionProviderDispatch>, String> {
    eps.iter()
        .map(|e| provider_name(env, e))
        .filter(|name| name != QNN)
        .map(|name| match name.as_str() {
            CPU => Ok(ep::CPU::default().build()),
            CUDA => cuda_ep(),
            TENSORRT => tensorrt_ep(),
            ACL => acl_ep(),
            ONEDNN | DNNL => onednn_ep(),
            COREML => coreml_ep(),
            DIRECTML => directml_ep(),
            ROCM => rocm_ep(),
            other => Err(format!(
                "Unknown execution provider: {}. Expected one of: {}",
                other,
                [CPU, CUDA, TENSORRT, ACL, ONEDNN, DNNL, COREML, DIRECTML, ROCM, QNN].join(", ")
            )),
        })
        .collect()
}

fn ep_not_compiled(name: &str, feature: &str) -> String {
    format!(
        "{name} execution provider is not compiled into this NIF. Rebuild with the \
         `{feature}` cargo feature: ORTEX_FEATURES={feature} mix compile (or \
         `features: [\"{feature}\"]` for `config :ortex, Ortex.Native`)"
    )
}

#[cfg(feature = "cuda")]
fn cuda_ep() -> Result<ExecutionProviderDispatch, String> {
    Ok(ep::CUDA::default().build())
}

#[cfg(not(feature = "cuda"))]
fn cuda_ep() -> Result<ExecutionProviderDispatch, String> {
    Err(ep_not_compiled("CUDA", "cuda"))
}

#[cfg(feature = "tensorrt")]
fn tensorrt_ep() -> Result<ExecutionProviderDispatch, String> {
    Ok(ep::TensorRT::default().build())
}

#[cfg(not(feature = "tensorrt"))]
fn tensorrt_ep() -> Result<ExecutionProviderDispatch, String> {
    Err(ep_not_compiled("TensorRT", "tensorrt"))
}

#[cfg(feature = "acl")]
fn acl_ep() -> Result<ExecutionProviderDispatch, String> {
    Ok(ep::ACL::default().build())
}

#[cfg(not(feature = "acl"))]
fn acl_ep() -> Result<ExecutionProviderDispatch, String> {
    Err(ep_not_compiled("ACL", "acl"))
}

#[cfg(feature = "onednn")]
fn onednn_ep() -> Result<ExecutionProviderDispatch, String> {
    Ok(ep::OneDNN::default().build())
}

#[cfg(not(feature = "onednn"))]
fn onednn_ep() -> Result<ExecutionProviderDispatch, String> {
    Err(ep_not_compiled("oneDNN", "onednn"))
}

#[cfg(feature = "coreml")]
fn coreml_ep() -> Result<ExecutionProviderDispatch, String> {
    Ok(ep::CoreML::default().build())
}

#[cfg(not(feature = "coreml"))]
fn coreml_ep() -> Result<ExecutionProviderDispatch, String> {
    Err(ep_not_compiled("CoreML", "coreml"))
}

#[cfg(feature = "directml")]
fn directml_ep() -> Result<ExecutionProviderDispatch, String> {
    Ok(ep::DirectML::default().build())
}

#[cfg(not(feature = "directml"))]
fn directml_ep() -> Result<ExecutionProviderDispatch, String> {
    Err(ep_not_compiled("DirectML", "directml"))
}

#[cfg(feature = "rocm")]
fn rocm_ep() -> Result<ExecutionProviderDispatch, String> {
    Ok(ep::ROCm::default().build())
}

#[cfg(not(feature = "rocm"))]
fn rocm_ep() -> Result<ExecutionProviderDispatch, String> {
    Err(ep_not_compiled("ROCm", "rocm"))
}

/// The process-wide ONNX Runtime environment, created on first use.
///
/// ort rc.14 made the environment explicit (`ort::init().build()`), and ONNX Runtime
/// still allows exactly one per process. It owns the thread pools and the registered
/// plugin execution providers - including QNN - so it has to outlive every session built
/// from it: a process-lifetime holder, not a field of `OrtexModel`.
static ENVIRONMENT: std::sync::Mutex<Option<ort::environment::Environment>> =
    std::sync::Mutex::new(None);

pub fn environment() -> Result<ort::environment::Environment, ort::Error> {
    let mut guard = ENVIRONMENT.lock().unwrap_or_else(|e| e.into_inner());

    if let Some(environment) = guard.as_ref() {
        return Ok(environment.clone());
    }

    // A failed build leaves the slot empty so the next load retries: on Nerves the
    // runtime library can be mounted after the first attempt.
    let environment = ort::init().with_name("ortex").build()?;
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
pub fn register_qnn_library(provider: &str) -> Result<(), String> {
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
        return Err(format!("QNN plugin EP library not found at {provider}"));
    }

    let env = environment().map_err(|e| format!("could not get ONNX Runtime environment: {e}"))?;
    // rc.15 removed `unregister`, and the registration lives with the environment, which
    // is kept for the lifetime of the process.
    env.register_ep_library("QNN", provider)
        .map_err(|e| format!("failed to register QNN plugin EP from {provider}: {e}"))?;

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
