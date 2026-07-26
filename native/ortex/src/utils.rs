//! Serialization and deserialization to transfer between Ortex and BinaryBackend
//! [Nx](https://hexdocs.pm/nx) backend.

use crate::constants::*;
use crate::tensor::OrtexTensor;
use ndarray::{ArrayViewMut, Ix, IxDyn};

use ndarray::ShapeError;

use rustler::resource::ResourceArc;
use rustler::types::Binary;
use rustler::{Atom, Env, NifResult};

use ort::{ExecutionProviderDispatch, GraphOptimizationLevel};

/// A faster (unsafe) way of creating an Array from an Erlang binary
fn initialize_from_raw_ptr<T>(ptr: *const T, shape: &[Ix]) -> ArrayViewMut<T, IxDyn> {
    let array = unsafe { ArrayViewMut::from_shape_ptr(shape, ptr as *mut T) };
    array
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
) -> Result<ResourceArc<OrtexTensor>, ShapeError> {
    match (dtype_str.as_ref(), dtype_bits) {
        ("bf", 16) => Ok(ResourceArc::new(OrtexTensor::bf16(
            initialize_from_raw_ptr(bin.as_ptr() as *const half::bf16, &shape).to_owned(),
        ))),
        ("f", 16) => Ok(ResourceArc::new(OrtexTensor::f16(
            initialize_from_raw_ptr(bin.as_ptr() as *const half::f16, &shape).to_owned(),
        ))),
        ("f", 32) => Ok(ResourceArc::new(OrtexTensor::f32(
            initialize_from_raw_ptr(bin.as_ptr() as *const f32, &shape).to_owned(),
        ))),
        ("f", 64) => Ok(ResourceArc::new(OrtexTensor::f64(
            initialize_from_raw_ptr(bin.as_ptr() as *const f64, &shape).to_owned(),
        ))),
        ("s", 8) => Ok(ResourceArc::new(OrtexTensor::s8(
            initialize_from_raw_ptr(bin.as_ptr() as *const i8, &shape).to_owned(),
        ))),
        ("s", 16) => Ok(ResourceArc::new(OrtexTensor::s16(
            initialize_from_raw_ptr(bin.as_ptr() as *const i16, &shape).to_owned(),
        ))),
        ("s", 32) => Ok(ResourceArc::new(OrtexTensor::s32(
            initialize_from_raw_ptr(bin.as_ptr() as *const i32, &shape).to_owned(),
        ))),
        ("s", 64) => Ok(ResourceArc::new(OrtexTensor::s64(
            initialize_from_raw_ptr(bin.as_ptr() as *const i64, &shape).to_owned(),
        ))),
        ("u", 8) => Ok(ResourceArc::new(OrtexTensor::u8(
            initialize_from_raw_ptr(bin.as_ptr() as *const u8, &shape).to_owned(),
        ))),
        ("u", 16) => Ok(ResourceArc::new(OrtexTensor::u16(
            initialize_from_raw_ptr(bin.as_ptr() as *const u16, &shape).to_owned(),
        ))),
        ("u", 32) => Ok(ResourceArc::new(OrtexTensor::u32(
            initialize_from_raw_ptr(bin.as_ptr() as *const u32, &shape).to_owned(),
        ))),
        ("u", 64) => Ok(ResourceArc::new(OrtexTensor::u64(
            initialize_from_raw_ptr(bin.as_ptr() as *const u64, &shape).to_owned(),
        ))),
        (&_, _) => unimplemented!(),
    }
}

/// Given a reference to an OrtexTensor return the binary representation to be used
/// by the BinaryBackend of Nx.
pub fn to_binary<'a>(
    env: Env<'a>,
    reference: ResourceArc<OrtexTensor>,
    _bits: usize,
    _limit: usize,
) -> NifResult<Binary<'a>> {
    Ok(reference.make_binary(env, |x| x.to_bytes()))
}

/// Takes a vec of Atoms and transforms them into a vec of ExecutionProvider Enums
pub fn map_eps(env: rustler::env::Env, eps: Vec<Atom>) -> Vec<ExecutionProviderDispatch> {
    eps.iter()
        .map(|e| match &e.to_term(env).atom_to_string().unwrap()[..] {
            CPU => ort::CPUExecutionProvider::default().build(),
            CUDA => ort::CUDAExecutionProvider::default().build(),
            TENSORRT => ort::TensorRTExecutionProvider::default().build(),
            ACL => ort::ACLExecutionProvider::default().build(),
            ONEDNN => ort::OneDNNExecutionProvider::default().build(),
            COREML => ort::CoreMLExecutionProvider::default().build(),
            DIRECTML => ort::DirectMLExecutionProvider::default().build(),
            ROCM => ort::ROCmExecutionProvider::default().build(),
            QNN => qnn_ep(),
            other => {
                // Previously this silently fell through to CPU, which meant a
                // typo'd or unsupported EP atom produced working inference with
                // no acceleration and no error at all. Still fall back (so
                // behaviour is unchanged for callers) but say so.
                eprintln!("[ortex] unknown execution provider {other:?}; falling back to CPU");
                ort::CPUExecutionProvider::default().build()
            }
        })
        .collect()
}

/// Build the Qualcomm QNN execution provider.
///
/// Configured by environment rather than by the Elixir API, because `map_eps`
/// only receives bare atoms. Defaults suit the Radxa Dragon Q6A (QCS6490, HTP
/// v68), where libQnnHtp.so is installed by the qairt-runtime package.
///
///   ORTEX_QNN_BACKEND_PATH   default "/usr/lib/libQnnHtp.so"
///
/// Note the backend .so is loaded by onnxruntime at session creation, so a
/// wrong path surfaces as a session error, not a load error here.
///
/// Performance mode / profiling are deliberately not wired up: in ort
/// 2.0.0-rc.8 the `PerformanceMode` enum is not re-exported at the `ort` root
/// (it lives in the qnn EP module), so naming it here fails to compile. Add it
/// once the ort dependency is bumped and the path is confirmed.
fn qnn_ep() -> ExecutionProviderDispatch {
    let backend_path = std::env::var("ORTEX_QNN_BACKEND_PATH")
        .unwrap_or_else(|_| "/usr/lib/libQnnHtp.so".to_string());

    ort::QNNExecutionProvider::default()
        .with_backend_path(backend_path)
        .build()
        // Without this, ort's apply_execution_providers() logs the registration
        // failure via `tracing` and returns Ok(()), so onnxruntime quietly falls
        // back to CPU: you get correct numbers and no error, which is
        // indistinguishable from working acceleration. ortex also has its
        // tracing_subscriber init commented out, so the log goes nowhere.
        // Fail loudly instead - a QNN session that can't use QNN is a bug.
        .error_on_failure()
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

pub fn is_bool_input(inp: &ort::ValueType) -> bool {
    match inp {
        ort::ValueType::Tensor { ty, .. } => ty == &ort::TensorElementType::Bool,
        ort::ValueType::Map { value, .. } => value == &ort::TensorElementType::Bool,
        ort::ValueType::Sequence(boxed_input) => is_bool_input(boxed_input),
        ort::ValueType::Optional(boxed_input) => is_bool_input(boxed_input),
    }
}
