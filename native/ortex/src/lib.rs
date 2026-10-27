//! # Ortex
//! Rust bindings between [ONNX Runtime](https://github.com/microsoft/onnxruntime) and
//! Erlang/Elixir using [Ort](https://docs.rs/ort) and [Rustler](https://docs.rs/rustler).
//! These are only meant to be accessed via the NIF interface provided by Rustler and not
//! directly.

mod constants;
mod error;
mod model;
mod tensor;
mod utils;

use model::{IoSpec, RunOutput};
use tensor::TensorResource;

use rustler::types::Binary;
use rustler::{Atom, Env, NifResult, ResourceArc, Term};

#[rustler::nif(schedule = "DirtyIo")]
fn init(
    env: Env,
    model_path: String,
    eps: Vec<Atom>,
    opt: i32,
    qnn_opts: Vec<(String, String)>,
) -> NifResult<ResourceArc<model::OrtexModel>> {
    let use_qnn = utils::wants_qnn(env, &eps);
    let eps = utils::map_eps(env, eps)?;
    let model = model::init(model_path, eps, use_qnn, opt, qnn_opts)?;
    Ok(ResourceArc::new(model))
}

#[rustler::nif]
fn show_session(model: ResourceArc<model::OrtexModel>) -> NifResult<(IoSpec, IoSpec)> {
    Ok(model::show(&model))
}

/// Inference is CPU/NPU-bound: it holds a dirty scheduler for the whole run, so it belongs
/// on the dirty CPU pool, which scales with the machine. DirtyIo is a fixed pool of 10
/// threads regardless of core count (an artificial ceiling on concurrent inference) and is
/// shared with real I/O - file and serial port work that a hung NPU run would starve.
/// `init` stays DirtyIo: loading a model really does read a file.
#[rustler::nif(schedule = "DirtyCpu")]
fn run(
    model: ResourceArc<model::OrtexModel>,
    inputs: Vec<ResourceArc<TensorResource>>,
) -> NifResult<Vec<RunOutput>> {
    Ok(model::run(model, inputs)?)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn from_binary(bin: Binary, shape: Term, dtype: Term) -> NifResult<ResourceArc<TensorResource>> {
    let shape: Vec<usize> = rustler::types::tuple::get_tuple(shape)?
        .iter()
        .map(|x| -> NifResult<usize> { x.decode::<usize>() })
        .collect::<NifResult<Vec<usize>>>()?;
    let (dtype_t, dtype_bits): (Term, usize) = dtype.decode()?;
    let dtype_str = dtype_t.atom_to_string()?;

    Ok(utils::from_binary(bin, shape, dtype_str, dtype_bits)?)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn to_binary<'a>(
    env: Env<'a>,
    reference: ResourceArc<TensorResource>,
    bits: usize,
    limit: usize,
) -> NifResult<Binary<'a>> {
    utils::to_binary(env, reference, bits, limit)
}

/// Bytes of tensor data currently held by live `OrtexTensor` resources.
///
/// Everything the BEAM sees as an Ortex tensor lives on the Rust heap, invisible to
/// `:erlang.memory/0` and to the collector; this is the number that makes it visible.
#[rustler::nif]
fn memory_usage() -> usize {
    tensor::allocated_bytes()
}

#[rustler::nif]
pub fn slice(
    tensor: ResourceArc<TensorResource>,
    start_indicies: Vec<isize>,
    lengths: Vec<isize>,
    strides: Vec<isize>,
) -> NifResult<ResourceArc<TensorResource>> {
    Ok(TensorResource::new(tensor.slice(
        start_indicies,
        lengths,
        strides,
    )?))
}

#[rustler::nif]
pub fn reshape(
    tensor: ResourceArc<TensorResource>,
    shape: Vec<usize>,
) -> NifResult<ResourceArc<TensorResource>> {
    Ok(TensorResource::new(tensor.reshape(shape)?))
}

#[rustler::nif]
pub fn concatenate(
    tensors: Vec<ResourceArc<TensorResource>>,
    dtype: Term,
    axis: i32,
) -> NifResult<ResourceArc<TensorResource>> {
    let (dtype_t, dtype_bits): (Term, usize) = dtype.decode()?;
    let dtype_str = dtype_t.atom_to_string()?;
    let concatted = tensor::concatenate(tensors, (&dtype_str, dtype_bits), axis as usize)?;
    Ok(TensorResource::new(concatted))
}

rustler::init!("Elixir.Ortex.Native");
