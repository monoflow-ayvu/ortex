//! # Ortex
//! Rust bindings between [ONNX Runtime](https://github.com/microsoft/onnxruntime) and
//! Erlang/Elixir using [Ort](https://docs.rs/ort) and [Rustler](https://docs.rs/rustler).
//! These are only meant to be accessed via the NIF interface provided by Rustler and not
//! directly.

mod constants;
mod model;
mod tensor;
mod utils;

use model::OrtexModel;
use tensor::OrtexTensor;

use std::ptr;
use std::sync::atomic::{AtomicPtr, Ordering};

use rustler::resource::{
    open_struct_resource_type, NIF_RESOURCE_FLAGS, ResourceType, ResourceTypeProvider,
};
use rustler::resource::ResourceArc;
use rustler::types::Binary;
use rustler::{Atom, Env, NifResult, Term};

static ORTEX_MODEL_TYPE: AtomicPtr<ResourceType<OrtexModel>> = AtomicPtr::new(ptr::null_mut());
static ORTEX_TENSOR_TYPE: AtomicPtr<ResourceType<OrtexTensor>> = AtomicPtr::new(ptr::null_mut());

impl ResourceTypeProvider for OrtexModel {
    fn get_type() -> &'static ResourceType<Self> {
        let ptr = ORTEX_MODEL_TYPE.load(Ordering::Acquire);
        if ptr.is_null() {
            panic!(
                "OrtexModel resource type is not initialized. Did you call the NIF load function?"
            );
        }
        unsafe { &*ptr }
    }
}

impl ResourceTypeProvider for OrtexTensor {
    fn get_type() -> &'static ResourceType<Self> {
        let ptr = ORTEX_TENSOR_TYPE.load(Ordering::Acquire);
        if ptr.is_null() {
            panic!(
                "OrtexTensor resource type is not initialized. Did you call the NIF load function?"
            );
        }
        unsafe { &*ptr }
    }
}

fn load(env: Env, _info: Term) -> bool {
    if ORTEX_MODEL_TYPE.load(Ordering::Acquire).is_null() {
        let model_type = match open_struct_resource_type::<OrtexModel>(
            env,
            concat!(stringify!(OrtexModel), "\x00"),
            NIF_RESOURCE_FLAGS::ERL_NIF_RT_CREATE,
        ) {
            Some(resource_type) => resource_type,
            None => {
                println!("Failure in creating OrtexModel resource type");
                return false;
            }
        };

        let model_ptr = Box::into_raw(Box::new(model_type));
        if ORTEX_MODEL_TYPE
            .compare_exchange(ptr::null_mut(), model_ptr, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            unsafe {
                drop(Box::from_raw(model_ptr));
            }
        }
    }

    if ORTEX_TENSOR_TYPE.load(Ordering::Acquire).is_null() {
        let tensor_type = match open_struct_resource_type::<OrtexTensor>(
            env,
            concat!(stringify!(OrtexTensor), "\x00"),
            NIF_RESOURCE_FLAGS::ERL_NIF_RT_CREATE,
        ) {
            Some(resource_type) => resource_type,
            None => {
                println!("Failure in creating OrtexTensor resource type");
                return false;
            }
        };

        let tensor_ptr = Box::into_raw(Box::new(tensor_type));
        if ORTEX_TENSOR_TYPE
            .compare_exchange(ptr::null_mut(), tensor_ptr, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            unsafe {
                drop(Box::from_raw(tensor_ptr));
            }
        }
    }

    true
}

#[rustler::nif(schedule = "DirtyIo")]
fn init(
    env: Env,
    model_path: String,
    eps: Vec<Atom>,
    opt: i32,
) -> NifResult<ResourceArc<model::OrtexModel>> {
    let eps = utils::map_eps(env, eps);
    let model = model::init(model_path, eps, opt)
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?;
    Ok(ResourceArc::new(model))
}

#[rustler::nif]
fn show_session(
    model: ResourceArc<model::OrtexModel>,
) -> NifResult<(
    Vec<(String, String, Option<Vec<i64>>)>,
    Vec<(String, String, Option<Vec<i64>>)>,
)> {
    Ok(model::show(model))
}

#[rustler::nif(schedule = "DirtyIo")]
fn run(
    model: ResourceArc<model::OrtexModel>,
    inputs: Vec<ResourceArc<OrtexTensor>>,
) -> NifResult<Vec<(ResourceArc<OrtexTensor>, Vec<usize>, Atom, usize)>> {
    model::run(model, inputs).map_err(|e| rustler::Error::Term(Box::new(e.to_string())))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn from_binary(bin: Binary, shape: Term, dtype: Term) -> NifResult<ResourceArc<OrtexTensor>> {
    let shape: Vec<usize> = rustler::types::tuple::get_tuple(shape)?
        .iter()
        .map(|x| -> NifResult<usize> { Ok(x.decode::<usize>())? })
        .collect::<NifResult<Vec<usize>>>()?;
    let (dtype_t, dtype_bits): (Term, usize) = dtype.decode()?;
    let dtype_str = dtype_t.atom_to_string()?;

    utils::from_binary(bin, shape, dtype_str, dtype_bits)
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn to_binary<'a>(
    env: Env<'a>,
    reference: ResourceArc<OrtexTensor>,
    bits: usize,
    limit: usize,
) -> NifResult<Binary<'a>> {
    utils::to_binary(env, reference, bits, limit)
}

#[rustler::nif]
pub fn slice(
    tensor: ResourceArc<OrtexTensor>,
    start_indicies: Vec<isize>,
    lengths: Vec<isize>,
    strides: Vec<isize>,
) -> NifResult<ResourceArc<OrtexTensor>> {
    Ok(ResourceArc::new(tensor.slice(
        start_indicies,
        lengths,
        strides,
    )?))
}

#[rustler::nif]
pub fn reshape(
    tensor: ResourceArc<OrtexTensor>,
    shape: Vec<usize>,
) -> NifResult<ResourceArc<OrtexTensor>> {
    Ok(ResourceArc::new(tensor.reshape(shape)?))
}

#[rustler::nif]
pub fn concatenate(
    tensors: Vec<ResourceArc<OrtexTensor>>,
    dtype: Term,
    axis: i32,
) -> NifResult<ResourceArc<OrtexTensor>> {
    let (dtype_t, dtype_bits): (Term, usize) = dtype.decode()?;
    let dtype_str = dtype_t.atom_to_string()?;
    let concatted = tensor::concatenate(tensors, (&dtype_str, dtype_bits), axis as usize)
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))?;
    Ok(ResourceArc::new(concatted))
}

rustler::init!(
    "Elixir.Ortex.Native",
    [
        run,
        init,
        from_binary,
        to_binary,
        show_session,
        slice,
        reshape,
        concatenate
    ],
    load = load
);
