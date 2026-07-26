//! Serialization and deserialization to transfer between Ortex and BinaryBackend
//! [Nx](https://hexdocs.pm/nx) backend.

use crate::constants::*;
use crate::tensor::OrtexTensor;
use ndarray::{Array, IxDyn};

use rustler::types::Binary;
use rustler::{Atom, Env, Error as RustlerError, NifResult, ResourceArc};

use ort::execution_providers::{
    ACLExecutionProvider, CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider,
    DirectMLExecutionProvider, ExecutionProviderDispatch, OneDNNExecutionProvider,
    QNNExecutionProvider, ROCmExecutionProvider, TensorRTExecutionProvider,
};
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorElementType;
use ort::value::ValueType;

fn element_count(shape: &[usize]) -> Result<usize, String> {
    shape
        .iter()
        .try_fold(1usize, |acc, dim| {
            acc.checked_mul(*dim)
                .ok_or_else(|| "Tensor shape is too large to index".to_string())
        })
}

fn array_from_binary<T: Copy + Default>(
    bin: &Binary,
    shape: &[usize],
) -> Result<Array<T, IxDyn>, String> {
    let elements = element_count(shape)?;
    let elem_size = std::mem::size_of::<T>();
    let expected_bytes = elements
        .checked_mul(elem_size)
        .ok_or_else(|| "Tensor binary size overflows usize".to_string())?;

    if bin.len() != expected_bytes {
        return Err(format!(
            "Binary length mismatch for shape {:?}: expected {} bytes, got {}",
            shape,
            expected_bytes,
            bin.len()
        ));
    }

    let mut data = vec![T::default(); elements];
    if expected_bytes > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(
                bin.as_ptr(),
                data.as_mut_ptr() as *mut u8,
                expected_bytes,
            );
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
    match (dtype_str.as_ref(), dtype_bits) {
        ("bf", 16) => Ok(ResourceArc::new(OrtexTensor::bf16(
            array_from_binary::<half::bf16>(&bin, &shape)?,
        ))),
        ("f", 16) => Ok(ResourceArc::new(OrtexTensor::f16(
            array_from_binary::<half::f16>(&bin, &shape)?,
        ))),
        ("f", 32) => Ok(ResourceArc::new(OrtexTensor::f32(array_from_binary::<f32>(
            &bin, &shape,
        )?))),
        ("f", 64) => Ok(ResourceArc::new(OrtexTensor::f64(array_from_binary::<f64>(
            &bin, &shape,
        )?))),
        ("s", 8) => Ok(ResourceArc::new(OrtexTensor::s8(array_from_binary::<i8>(
            &bin, &shape,
        )?))),
        ("s", 16) => Ok(ResourceArc::new(OrtexTensor::s16(array_from_binary::<i16>(
            &bin, &shape,
        )?))),
        ("s", 32) => Ok(ResourceArc::new(OrtexTensor::s32(array_from_binary::<i32>(
            &bin, &shape,
        )?))),
        ("s", 64) => Ok(ResourceArc::new(OrtexTensor::s64(array_from_binary::<i64>(
            &bin, &shape,
        )?))),
        ("u", 8) => Ok(ResourceArc::new(OrtexTensor::u8(array_from_binary::<u8>(
            &bin, &shape,
        )?))),
        ("u", 16) => Ok(ResourceArc::new(OrtexTensor::u16(array_from_binary::<u16>(
            &bin, &shape,
        )?))),
        ("u", 32) => Ok(ResourceArc::new(OrtexTensor::u32(array_from_binary::<u32>(
            &bin, &shape,
        )?))),
        ("u", 64) => Ok(ResourceArc::new(OrtexTensor::u64(array_from_binary::<u64>(
            &bin, &shape,
        )?))),
        (&_, _) => Err(format!(
            "Unsupported dtype {} with {} bits",
            dtype_str, dtype_bits
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
    if bits == 0 || limit == 0 {
        return Ok(reference.make_binary(env, |x| x.to_bytes()));
    }

    if bits % 8 != 0 {
        return Err(RustlerError::Term(Box::new(format!(
            "Invalid element bit size: {}",
            bits
        ))));
    }

    let elem_size = bits / 8;
    let elem_count = element_count(&reference.shape())
        .map_err(|e| RustlerError::Term(Box::new(e)))?;
    let capped = std::cmp::min(limit, elem_count);
    let byte_limit = capped
        .checked_mul(elem_size)
        .ok_or_else(|| RustlerError::Term(Box::new("Binary size overflows usize".to_string())))?;

    Ok(reference.make_binary(env, |x| {
        let bytes = x.to_bytes();
        let len = std::cmp::min(byte_limit, bytes.len());
        &bytes[..len]
    }))
}

/// Takes a vec of Atoms and transforms them into a vec of ExecutionProvider Enums
/// True if `:qnn` is among the requested providers.
///
/// QNN cannot go through `map_eps`: on upstream ONNX Runtime builds it is a
/// *plugin* execution provider, which is selected with the V2 device API
/// (`SessionBuilder::with_devices`) rather than by appending an
/// `ExecutionProviderDispatch`. `model::init` handles it separately, so it is
/// filtered out of the dispatch list here.
pub fn wants_qnn(env: rustler::env::Env, eps: &[Atom]) -> bool {
    eps.iter().any(|e| {
        e.to_term(env)
            .atom_to_string()
            .map(|s| s == QNN)
            .unwrap_or(false)
    })
}

pub fn map_eps(
    env: rustler::env::Env,
    eps: Vec<Atom>,
) -> Result<Vec<ExecutionProviderDispatch>, String> {
    eps.iter()
        .filter(|e| {
            e.to_term(env)
                .atom_to_string()
                .map(|s| s != QNN)
                .unwrap_or(true)
        })
        .map(|e| {
            let atom_str = e
                .to_term(env)
                .atom_to_string()
                .map_err(|_| "Execution provider must be an atom".to_string())?;
            match atom_str.as_str() {
                CPU => Ok(CPUExecutionProvider::default().build()),
                CUDA => Ok(CUDAExecutionProvider::default().build()),
                TENSORRT => Ok(TensorRTExecutionProvider::default().build()),
                ACL => Ok(ACLExecutionProvider::default().build()),
                ONEDNN | "dnnl" => Ok(OneDNNExecutionProvider::default().build()),
                COREML => Ok(CoreMLExecutionProvider::default().build()),
                DIRECTML => Ok(DirectMLExecutionProvider::default().build()),
                ROCM => Ok(ROCmExecutionProvider::default().build()),
                _ => Err(format!(
                    "Unknown execution provider: {}. Expected one of: {}",
                    atom_str,
                    vec![
                        CPU, CUDA, TENSORRT, ACL, ONEDNN, "dnnl", COREML, DIRECTML, ROCM, QNN
                    ]
                    .join(", ")
                )),
            }
        })
        .collect()
}

/// Register the Qualcomm QNN *plugin* execution provider library, once.
///
/// Upstream `libonnxruntime.so` is NOT built with `--use_qnn`, so appending
/// "QNN" by name fails with "QNN execution provider is not supported in this
/// build." QNN ships as a plugin EP (`libonnxruntime_providers_qnn.so`) which
/// must be registered with the environment (ORT >= 1.23
/// `RegisterExecutionProviderLibrary`) and then selected via the V2 device API,
/// `SessionBuilder::with_devices` - NOT via ExecutionProviderDispatch.
///
///   ORTEX_QNN_PROVIDER_PATH  plugin EP library
///                            (default /usr/lib/libonnxruntime_providers_qnn.so)
///   ORTEX_QNN_BACKEND_PATH   backend the EP loads
///                            (default /usr/lib/libQnnHtp.so)
pub fn register_qnn_library(provider: &str) -> Result<(), String> {
    static REGISTERED: std::sync::OnceLock<Result<(), String>> = std::sync::OnceLock::new();

    REGISTERED
        .get_or_init(|| {
            let provider = provider.to_string();
            if !std::path::Path::new(&provider).exists() {
                return Err(format!("QNN plugin EP library not found at {provider}"));
            }

            let env = ort::environment::current()
                .map_err(|e| format!("could not get ONNX Runtime environment: {e}"))?;

            match env.register_ep_library("QNN", &provider) {
                Ok(lib) => {
                    // Leak: unregistering would invalidate live sessions, and the
                    // EP must remain available for the process lifetime.
                    std::mem::forget(lib);
                    Ok(())
                }
                Err(e) => Err(format!("failed to register QNN plugin EP from {provider}: {e}"))
            }
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
