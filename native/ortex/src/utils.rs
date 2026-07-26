//! Serialization and deserialization to transfer between Ortex and BinaryBackend
//! [Nx](https://hexdocs.pm/nx) backend.

use crate::constants::*;
use crate::tensor::OrtexTensor;
use ndarray::{Array, IxDyn};

use rustler::types::Binary;
use rustler::{Atom, Env, NifResult, ResourceArc};

use ort::execution_providers::{
    ACLExecutionProvider, CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider,
    DirectMLExecutionProvider, ExecutionProviderDispatch, OneDNNExecutionProvider,
    ROCmExecutionProvider, TensorRTExecutionProvider,
};
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
) -> Result<ResourceArc<OrtexTensor>, String> {
    // `typ` is the element type, `ort_tensor_kind` the matching OrtexTensor variant
    macro_rules! tensor {
        ($typ:ty, $ort_tensor_kind:ident) => {
            Ok(ResourceArc::new(OrtexTensor::$ort_tensor_kind(
                array_from_binary::<$typ>(&bin, &shape)?,
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
    reference: ResourceArc<OrtexTensor>,
    bits: usize,
    limit: usize,
) -> NifResult<Binary<'a>> {
    // `Ortex.Backend.backend_transfer/3` asks for the whole tensor with a limit of 0.
    if limit == 0 {
        return Ok(reference.make_binary(env, |x| x.to_bytes()));
    }

    let byte_limit = limit.saturating_mul(bits / 8);
    Ok(reference.make_binary(env, |x| {
        let bytes = x.to_bytes();
        &bytes[..byte_limit.min(bytes.len())]
    }))
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

/// Takes a vec of Atoms and transforms them into a vec of ExecutionProvider Enums
pub fn map_eps(env: Env, eps: Vec<Atom>) -> Result<Vec<ExecutionProviderDispatch>, String> {
    eps.iter()
        .map(|e| provider_name(env, e))
        .filter(|name| name != QNN)
        .map(|name| match name.as_str() {
            CPU => Ok(CPUExecutionProvider::default().build()),
            CUDA => Ok(CUDAExecutionProvider::default().build()),
            TENSORRT => Ok(TensorRTExecutionProvider::default().build()),
            ACL => Ok(ACLExecutionProvider::default().build()),
            ONEDNN | DNNL => Ok(OneDNNExecutionProvider::default().build()),
            COREML => Ok(CoreMLExecutionProvider::default().build()),
            DIRECTML => Ok(DirectMLExecutionProvider::default().build()),
            ROCM => Ok(ROCmExecutionProvider::default().build()),
            _ => Err(format!(
                "Unknown execution provider: {}. Expected one of: {}",
                name,
                [CPU, CUDA, TENSORRT, ACL, ONEDNN, DNNL, COREML, DIRECTML, ROCM, QNN].join(", ")
            )),
        })
        .collect()
}

/// Register the Qualcomm QNN *plugin* execution provider library, once.
///
/// Upstream `libonnxruntime.so` is not built with `--use_qnn`, so appending "QNN" by
/// name fails with "QNN execution provider is not supported in this build." Instead the
/// plugin has to be registered with the environment (ORT >= 1.23
/// `RegisterExecutionProviderLibrary`) and then selected via `with_devices`.
pub fn register_qnn_library(provider: &str) -> Result<(), String> {
    static REGISTERED: std::sync::OnceLock<Result<(), String>> = std::sync::OnceLock::new();

    REGISTERED
        .get_or_init(|| {
            if !std::path::Path::new(provider).exists() {
                return Err(format!("QNN plugin EP library not found at {provider}"));
            }

            let env = ort::environment::current()
                .map_err(|e| format!("could not get ONNX Runtime environment: {e}"))?;
            let lib = env
                .register_ep_library("QNN", provider)
                .map_err(|e| format!("failed to register QNN plugin EP from {provider}: {e}"))?;

            // Unregistering would invalidate live sessions, and the EP has to stay
            // available for the lifetime of the process.
            std::mem::forget(lib);
            Ok(())
        })
        .clone()
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
