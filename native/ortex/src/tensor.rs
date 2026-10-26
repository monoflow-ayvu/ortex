//! Conversions for packing/unpacking `OrtexTensor`s into different types
use core::convert::TryFrom;
use ndarray::prelude::*;
use ndarray::{ArcArray, IxDyn, IxDynImpl, ViewRepr};
use ort::session::SessionInputValue;
use ort::value::{TensorElementType, TensorRef, Value, ValueType};
use ort::Error;
use rustler::{Atom, Resource, ResourceArc};

use crate::constants::ortex_atoms;

/// Tensor storage passed back and forth with the BEAM.
///
/// The buffers are `ArcArray`s - reference counted, copy-on-write - not plain `Array`s,
/// so `clone` and `reshape` are O(1) in the number of elements: Nx's `clone`, `reshape`,
/// `flatten` and `squeeze` on an Ortex tensor bump a refcount instead of copying the
/// whole buffer, which matters for activations in the MBs.
///
/// Invariant: every variant holds a standard (C-order, no offset) layout. `to_bytes`
/// hands the raw buffer to the BEAM as one flat range, which is only the tensor's
/// contents under that layout. Every constructor here normalizes: `from_binary` builds
/// from a shaped vec, `slice` and `concatenate` materialize with `to_owned`/
/// `as_standard_layout`, `reshape` only rewrites strides of an already-standard array,
/// and ORT outputs are contiguous. `get_bytes` asserts it in debug builds.
#[derive(Debug)]
#[allow(non_camel_case_types)]
/// Enum for wrapping different types to pass back to the BEAM since rustler can't
/// pass type generics back and forth
pub enum OrtexTensor {
    s8(ArcArray<i8, IxDyn>),
    s16(ArcArray<i16, IxDyn>),
    s32(ArcArray<i32, IxDyn>),
    s64(ArcArray<i64, IxDyn>),
    u8(ArcArray<u8, IxDyn>),
    u16(ArcArray<u16, IxDyn>),
    u32(ArcArray<u32, IxDyn>),
    u64(ArcArray<u64, IxDyn>),
    f16(ArcArray<half::f16, IxDyn>),
    bf16(ArcArray<half::bf16, IxDyn>),
    f32(ArcArray<f32, IxDyn>),
    f64(ArcArray<f64, IxDyn>),
    // bool is for internal use only: Nx sees these as u8
    bool(ArcArray<bool, IxDyn>),
    /// A tensor still owned by ONNX Runtime, as `Ortex.run/2` returns it.
    ///
    /// The value owns the ORT buffer (and, since rc.15, outlives the allocator that made
    /// it), so `to_bytes` can point the BEAM binary straight at that memory - one copy
    /// per output instead of "ORT -> ndarray -> BEAM". Operations that need an ndarray
    /// (reshape, slice, concatenate) materialize first and pay the copy then; `to_binary`
    /// is the common path, and it no longer does.
    ort(Value),
}

#[rustler::resource_impl(name = "OrtexTensor")]
impl Resource for OrtexTensor {}

// `ort::Value` holds raw pointers and so is not unwind-safe; rustler's blanket
// `NifReturnable` impl requires it to encode the resource term. Encoding only touches the
// resource handle, never the ORT value behind it - the same reasoning already applied to
// OrtexModel.
impl std::panic::RefUnwindSafe for OrtexTensor {}
impl std::panic::UnwindSafe for OrtexTensor {}

impl OrtexTensor {
    pub fn shape(&self) -> Vec<usize> {
        match self {
            OrtexTensor::s8(y) => y.shape().to_owned(),
            OrtexTensor::s16(y) => y.shape().to_owned(),
            OrtexTensor::s32(y) => y.shape().to_owned(),
            OrtexTensor::s64(y) => y.shape().to_owned(),
            OrtexTensor::u8(y) => y.shape().to_owned(),
            OrtexTensor::u16(y) => y.shape().to_owned(),
            OrtexTensor::u32(y) => y.shape().to_owned(),
            OrtexTensor::u64(y) => y.shape().to_owned(),
            OrtexTensor::f16(y) => y.shape().to_owned(),
            OrtexTensor::bf16(y) => y.shape().to_owned(),
            OrtexTensor::f32(y) => y.shape().to_owned(),
            OrtexTensor::f64(y) => y.shape().to_owned(),
            OrtexTensor::bool(y) => y.shape().to_owned(),
            OrtexTensor::ort(value) => value_shape(value),
        }
    }

    /// Rewrites the shape without touching the data.
    ///
    /// `into_shape_with_order` is a metadata-only operation when the array is standard
    /// layout, and sharing is a refcount - so a reshape of a 600 KB output is a couple of
    /// pointer writes rather than a memcpy. An ORT-owned value has no ndarray to share,
    /// so it is materialized first.
    pub fn reshape(&self, shape: Vec<usize>) -> rustler::NifResult<Self> {
        let shape: IxDyn = IxDyn(&shape);
        let tensor = self.to_shared().map_err(nif_term)?;

        Ok(match tensor {
            OrtexTensor::s8(y) => OrtexTensor::s8(reshape_arc(&y, shape)?),
            OrtexTensor::s16(y) => OrtexTensor::s16(reshape_arc(&y, shape)?),
            OrtexTensor::s32(y) => OrtexTensor::s32(reshape_arc(&y, shape)?),
            OrtexTensor::s64(y) => OrtexTensor::s64(reshape_arc(&y, shape)?),
            OrtexTensor::u8(y) => OrtexTensor::u8(reshape_arc(&y, shape)?),
            OrtexTensor::u16(y) => OrtexTensor::u16(reshape_arc(&y, shape)?),
            OrtexTensor::u32(y) => OrtexTensor::u32(reshape_arc(&y, shape)?),
            OrtexTensor::u64(y) => OrtexTensor::u64(reshape_arc(&y, shape)?),
            OrtexTensor::f16(y) => OrtexTensor::f16(reshape_arc(&y, shape)?),
            OrtexTensor::bf16(y) => OrtexTensor::bf16(reshape_arc(&y, shape)?),
            OrtexTensor::f32(y) => OrtexTensor::f32(reshape_arc(&y, shape)?),
            OrtexTensor::f64(y) => OrtexTensor::f64(reshape_arc(&y, shape)?),
            OrtexTensor::bool(y) => OrtexTensor::bool(reshape_arc(&y, shape)?),
            // `to_shared` materializes this variant, so it cannot come back here.
            OrtexTensor::ort(value) => {
                return Err(rustler::Error::Term(Box::new(format!(
                    "could not materialize output tensor: {}",
                    value.dtype()
                ))))
            }
        })
    }

    pub fn dtype(&self) -> (Atom, usize) {
        match self {
            OrtexTensor::s8(_) => (ortex_atoms::s(), 8),
            OrtexTensor::s16(_) => (ortex_atoms::s(), 16),
            OrtexTensor::s32(_) => (ortex_atoms::s(), 32),
            OrtexTensor::s64(_) => (ortex_atoms::s(), 64),
            OrtexTensor::u8(_) => (ortex_atoms::u(), 8),
            OrtexTensor::u16(_) => (ortex_atoms::u(), 16),
            OrtexTensor::u32(_) => (ortex_atoms::u(), 32),
            OrtexTensor::u64(_) => (ortex_atoms::u(), 64),
            OrtexTensor::f16(_) => (ortex_atoms::f(), 16),
            OrtexTensor::bf16(_) => (ortex_atoms::bf(), 16),
            OrtexTensor::f32(_) => (ortex_atoms::f(), 32),
            OrtexTensor::f64(_) => (ortex_atoms::f(), 64),
            OrtexTensor::bool(_) => (ortex_atoms::u(), 8),
            OrtexTensor::ort(value) => value_dtype(value),
        }
    }

    pub fn to_bytes(&self) -> Result<&[u8], Error> {
        let contents: &[u8] = match self {
            OrtexTensor::s8(y) => get_bytes(y),
            OrtexTensor::s16(y) => get_bytes(y),
            OrtexTensor::s32(y) => get_bytes(y),
            OrtexTensor::s64(y) => get_bytes(y),
            OrtexTensor::u8(y) => get_bytes(y),
            OrtexTensor::u16(y) => get_bytes(y),
            OrtexTensor::u32(y) => get_bytes(y),
            OrtexTensor::u64(y) => get_bytes(y),
            OrtexTensor::f16(y) => get_bytes(y),
            OrtexTensor::bf16(y) => get_bytes(y),
            OrtexTensor::f32(y) => get_bytes(y),
            OrtexTensor::f64(y) => get_bytes(y),
            OrtexTensor::bool(y) => get_bytes(y),
            OrtexTensor::ort(value) => value_bytes(value)?,
        };
        Ok(contents)
    }

    pub fn slice(
        &self,
        start_indicies: Vec<isize>,
        lengths: Vec<isize>,
        strides: Vec<isize>,
    ) -> rustler::NifResult<Self> {
        // ndarray panics on a rank mismatch or a zero step, and these come
        // straight from the BEAM.
        let rank = self.shape().len();
        if start_indicies.len() != rank || lengths.len() != rank || strides.len() != rank {
            return Err(rustler::Error::Term(Box::new(format!(
                "Slice arguments must match tensor rank of {}",
                rank
            ))));
        }
        if strides.contains(&0) {
            return Err(rustler::Error::Term(Box::new(
                "Slice stride must be non-zero".to_string(),
            )));
        }

        let mut slice_specs: Vec<(isize, Option<isize>, isize)> = vec![];
        for ((start_index, length), stride) in start_indicies
            .iter()
            .zip(lengths.iter())
            .zip(strides.iter())
        {
            slice_specs.push((*start_index, Some(*length + *start_index), *stride));
        }
        // A strided slice is not a contiguous window of the buffer, so the result is
        // materialized: standard layout is part of this type's contract (`to_bytes`).
        let tensor = self.to_shared().map_err(nif_term)?;

        Ok(match tensor {
            OrtexTensor::s8(y) => OrtexTensor::s8(slice_arc(&y, &slice_specs)),
            OrtexTensor::s16(y) => OrtexTensor::s16(slice_arc(&y, &slice_specs)),
            OrtexTensor::s32(y) => OrtexTensor::s32(slice_arc(&y, &slice_specs)),
            OrtexTensor::s64(y) => OrtexTensor::s64(slice_arc(&y, &slice_specs)),
            OrtexTensor::u8(y) => OrtexTensor::u8(slice_arc(&y, &slice_specs)),
            OrtexTensor::u16(y) => OrtexTensor::u16(slice_arc(&y, &slice_specs)),
            OrtexTensor::u32(y) => OrtexTensor::u32(slice_arc(&y, &slice_specs)),
            OrtexTensor::u64(y) => OrtexTensor::u64(slice_arc(&y, &slice_specs)),
            OrtexTensor::f16(y) => OrtexTensor::f16(slice_arc(&y, &slice_specs)),
            OrtexTensor::bf16(y) => OrtexTensor::bf16(slice_arc(&y, &slice_specs)),
            OrtexTensor::f32(y) => OrtexTensor::f32(slice_arc(&y, &slice_specs)),
            OrtexTensor::f64(y) => OrtexTensor::f64(slice_arc(&y, &slice_specs)),
            OrtexTensor::bool(y) => OrtexTensor::bool(slice_arc(&y, &slice_specs)),
            OrtexTensor::ort(value) => {
                return Err(rustler::Error::Term(Box::new(format!(
                    "could not materialize output tensor: {}",
                    value.dtype()
                ))))
            }
        })
    }

    /// Rewrites a 0/1 u8 tensor as a bool tensor, for models that declare boolean inputs.
    ///
    /// Borrowed rather than consuming: callers keep the input they were given, and the new
    /// buffer has to be written element by element anyway.
    pub fn to_bool(&self) -> Result<OrtexTensor, Error> {
        match self {
            OrtexTensor::u8(y) => {
                if y.iter().any(|x| *x > 1) {
                    return Err(Error::new(
                        "Tried to convert a u8 tensor to bool, but not every element is 0 or 1",
                    ));
                }
                Ok(OrtexTensor::bool(y.mapv(|x| x == 1).into_shared()))
            }
            // A model output feeding a bool input: materialize, then rewrite.
            OrtexTensor::ort(value) => OrtexTensor::try_from(value)?.to_bool(),
            t => Err(Error::new(format!(
                "Can't convert this type {:?} to bool",
                t.dtype()
            ))),
        }
    }

    /// An owned handle to this tensor's data.
    ///
    /// Array variants share their buffer (refcount); an ORT-owned output has no ndarray to
    /// share, so it is copied out once - the price of an array operation on a fresh
    /// output.
    fn to_shared(&self) -> Result<OrtexTensor, Error> {
        Ok(match self {
            OrtexTensor::s8(y) => OrtexTensor::s8(y.clone()),
            OrtexTensor::s16(y) => OrtexTensor::s16(y.clone()),
            OrtexTensor::s32(y) => OrtexTensor::s32(y.clone()),
            OrtexTensor::s64(y) => OrtexTensor::s64(y.clone()),
            OrtexTensor::u8(y) => OrtexTensor::u8(y.clone()),
            OrtexTensor::u16(y) => OrtexTensor::u16(y.clone()),
            OrtexTensor::u32(y) => OrtexTensor::u32(y.clone()),
            OrtexTensor::u64(y) => OrtexTensor::u64(y.clone()),
            OrtexTensor::f16(y) => OrtexTensor::f16(y.clone()),
            OrtexTensor::bf16(y) => OrtexTensor::bf16(y.clone()),
            OrtexTensor::f32(y) => OrtexTensor::f32(y.clone()),
            OrtexTensor::f64(y) => OrtexTensor::f64(y.clone()),
            OrtexTensor::bool(y) => OrtexTensor::bool(y.clone()),
            OrtexTensor::ort(value) => OrtexTensor::try_from(value)?,
        })
    }
}

/// Nx-visible shape of an ORT-owned value.
fn value_shape(value: &Value) -> Vec<usize> {
    match value.dtype() {
        // A materialized value always has concrete dimensions; the clamp only protects the
        // metadata path if a symbolic dimension ever reaches here.
        ValueType::Tensor { shape, .. } => shape.iter().map(|d| (*d).max(0) as usize).collect(),
        _ => Vec::new(),
    }
}

/// Nx-visible dtype of an ORT-owned value.
fn value_dtype(value: &Value) -> (Atom, usize) {
    match value.dtype() {
        ValueType::Tensor { ty, .. } => match ty {
            TensorElementType::Bfloat16 => (ortex_atoms::bf(), 16),
            TensorElementType::Float16 => (ortex_atoms::f(), 16),
            TensorElementType::Float32 => (ortex_atoms::f(), 32),
            TensorElementType::Float64 => (ortex_atoms::f(), 64),
            TensorElementType::Uint8 | TensorElementType::Bool => (ortex_atoms::u(), 8),
            TensorElementType::Uint16 => (ortex_atoms::u(), 16),
            TensorElementType::Uint32 => (ortex_atoms::u(), 32),
            TensorElementType::Uint64 => (ortex_atoms::u(), 64),
            TensorElementType::Int8 => (ortex_atoms::s(), 8),
            TensorElementType::Int16 => (ortex_atoms::s(), 16),
            TensorElementType::Int32 => (ortex_atoms::s(), 32),
            TensorElementType::Int64 => (ortex_atoms::s(), 64),
            _ => (ortex_atoms::u(), 8),
        },
        _ => (ortex_atoms::u(), 8),
    }
}

/// The value's bytes, read straight out of ONNX Runtime's buffer.
fn value_bytes(value: &Value) -> Result<&[u8], Error> {
    let ValueType::Tensor { ty, .. } = value.dtype() else {
        return Err(Error::new(format!(
            "Can't read bytes from non tensor, got {}",
            value.dtype()
        )));
    };

    macro_rules! bytes {
        ($typ:ty) => {{
            let (_, data) = value.try_extract_tensor::<$typ>()?;
            unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
            }
        }};
    }

    Ok(match ty {
        TensorElementType::Bfloat16 => bytes!(half::bf16),
        TensorElementType::Float16 => bytes!(half::f16),
        TensorElementType::Float32 => bytes!(f32),
        TensorElementType::Float64 => bytes!(f64),
        TensorElementType::Uint8 => bytes!(u8),
        TensorElementType::Uint16 => bytes!(u16),
        TensorElementType::Uint32 => bytes!(u32),
        TensorElementType::Uint64 => bytes!(u64),
        TensorElementType::Int8 => bytes!(i8),
        TensorElementType::Int16 => bytes!(i16),
        TensorElementType::Int32 => bytes!(i32),
        TensorElementType::Int64 => bytes!(i64),
        TensorElementType::Bool => bytes!(bool),
        other => return Err(Error::new(format!("Unsupported tensor type {:?}", other))),
    })
}

fn nif_term<E: std::fmt::Display>(e: E) -> rustler::Error {
    rustler::Error::Term(Box::new(e.to_string()))
}

fn reshape_arc<T: Clone>(
    array: &ArcArray<T, IxDyn>,
    shape: IxDyn,
) -> rustler::NifResult<ArcArray<T, IxDyn>> {
    array
        .clone()
        .into_shape_with_order(shape)
        .map_err(|e| rustler::Error::Term(Box::new(e.to_string())))
}

fn slice_array<'a, T, D>(
    array: &'a ArcArray<T, D>,
    slice_specs: &'a [(isize, Option<isize>, isize)],
) -> ArrayView<'a, T, D>
where
    D: Dimension,
{
    array.slice_each_axis(|ax: ndarray::AxisDescription| {
        let (start, end, step) = slice_specs[ax.axis.index()];
        ndarray::Slice { start, end, step }
    })
}

fn slice_arc<'a, T, D>(
    array: &'a ArcArray<T, D>,
    slice_specs: &'a [(isize, Option<isize>, isize)],
) -> ArcArray<T, D>
where
    T: Clone,
    D: Dimension,
{
    slice_array(array, slice_specs).to_owned().into_shared()
}

fn get_bytes<T>(array: &ArcArray<T, IxDyn>) -> &[u8] {
    // `to_binary` hands this to the BEAM as one flat byte range, which is only the
    // contents of the tensor under a standard (C-order, no offset) layout. Every array
    // reaching here is normalized to that layout on construction - see the type's docs -
    // and this assert is what catches a future constructor that forgets.
    debug_assert!(
        array.is_standard_layout(),
        "get_bytes requires a standard layout; use as_standard_layout() first"
    );

    if array.is_empty() {
        return &[];
    }
    // `as_ptr` points at the first logical element; under standard layout that is also
    // the start of the buffer.
    unsafe {
        std::slice::from_raw_parts(
            array.as_ptr() as *const u8,
            array.len() * std::mem::size_of::<T>(),
        )
    }
}

impl TryFrom<&Value> for OrtexTensor {
    type Error = Error;
    fn try_from(e: &Value) -> Result<Self, Self::Error> {
        let dtype = e.dtype();
        let ValueType::Tensor { ty, .. } = dtype else {
            return Err(Error::new(format!(
                "Can't decode non tensor, got {}",
                dtype
            )));
        };

        macro_rules! extract {
            ($kind:ident, $typ:ty) => {
                OrtexTensor::$kind(e.try_extract_array::<$typ>()?.to_owned().into_shared())
            };
        }

        let tensor = match *ty {
            TensorElementType::Bfloat16 => extract!(bf16, half::bf16),
            TensorElementType::Float16 => extract!(f16, half::f16),
            TensorElementType::Float32 => extract!(f32, f32),
            TensorElementType::Float64 => extract!(f64, f64),
            TensorElementType::Uint8 => extract!(u8, u8),
            TensorElementType::Uint16 => extract!(u16, u16),
            TensorElementType::Uint32 => extract!(u32, u32),
            TensorElementType::Uint64 => extract!(u64, u64),
            TensorElementType::Int8 => extract!(s8, i8),
            TensorElementType::Int16 => extract!(s16, i16),
            TensorElementType::Int32 => extract!(s32, i32),
            TensorElementType::Int64 => extract!(s64, i64),
            // map the output into u8 space
            TensorElementType::Bool => OrtexTensor::u8(
                e.try_extract_array::<bool>()?
                    .mapv(|x| x as u8)
                    .into_shared(),
            ),
            other => return Err(Error::new(format!("Unsupported tensor type {:?}", other))),
        };

        Ok(tensor)
    }
}

impl<'a> TryFrom<&'a OrtexTensor> for SessionInputValue<'a> {
    type Error = Error;

    /// Borrows the tensor's buffer into an ORT value instead of copying it.
    ///
    /// `TensorRef::from_array_view` over an `ArcArray` keeps a refcounted guard of the
    /// buffer alive for as long as the ORT value exists, and ORT reads the data in place
    /// during `run`, so handing over a clone of the `ArcArray` costs one refcount bump -
    /// not the second full copy per input this used to do.
    fn try_from(ort_tensor: &'a OrtexTensor) -> Result<Self, Self::Error> {
        let r: SessionInputValue = match ort_tensor {
            OrtexTensor::s8(arr) => TensorRef::from_array_view(arr.clone())?.into(),
            OrtexTensor::s16(arr) => TensorRef::from_array_view(arr.clone())?.into(),
            OrtexTensor::s32(arr) => TensorRef::from_array_view(arr.clone())?.into(),
            OrtexTensor::s64(arr) => TensorRef::from_array_view(arr.clone())?.into(),
            OrtexTensor::f16(arr) => TensorRef::from_array_view(arr.clone())?.into(),
            OrtexTensor::f32(arr) => TensorRef::from_array_view(arr.clone())?.into(),
            OrtexTensor::f64(arr) => TensorRef::from_array_view(arr.clone())?.into(),
            OrtexTensor::bf16(arr) => TensorRef::from_array_view(arr.clone())?.into(),
            OrtexTensor::u8(arr) => TensorRef::from_array_view(arr.clone())?.into(),
            OrtexTensor::u16(arr) => TensorRef::from_array_view(arr.clone())?.into(),
            OrtexTensor::u32(arr) => TensorRef::from_array_view(arr.clone())?.into(),
            OrtexTensor::u64(arr) => TensorRef::from_array_view(arr.clone())?.into(),
            OrtexTensor::bool(arr) => TensorRef::from_array_view(arr.clone())?.into(),
            // A run output fed back in as an input: its Value already owns the ORT buffer,
            // so it can be passed as-is - no copy, no materialization.
            OrtexTensor::ort(value) => SessionInputValue::from(value),
        };
        Ok(r)
    }
}

// Currently only supports concatenating tenors of the same type.
//
// This is a similar structure to the above match clauses, except each function
// in map is more complex and needs to be written out explicitly. To reduce
// repetition, the concatenate! macro expands that code and makes the necessary
// minor tweaks

macro_rules! concatenate {
    // `typ` is the actual datatype, `ort_tensor_kind` is the OrtexTensor variant
    ($tensors:expr, $axis:expr, $typ:ty, $ort_tensor_kind:ident) => {{
        type ArrayType<'a> = ArrayBase<ViewRepr<&'a $typ>, Dim<IxDynImpl>>;
        fn filter<'a>(tensor: &'a OrtexTensor) -> Option<ArrayType<'a>> {
            match tensor {
                OrtexTensor::$ort_tensor_kind(x) => Some(x.view()),
                _ => None,
            }
        }
        // hack way to type coalesce. Filters out any ndarray's that don't
        // have the desired type
        let expected = $tensors.len();
        let views: Vec<ArrayType> = $tensors
            .iter()
            .filter_map(|tensor| filter(tensor))
            .collect();
        if views.len() != expected {
            return Err(Error::new("Concatenate called with mixed tensor types"));
        }

        let joined =
            ndarray::concatenate(Axis($axis), &views).map_err(|e| Error::new(e.to_string()))?;
        // data is not contiguous after the concatenation above. To decode properly, need to
        // create a new contiguous vector: as_standard_layout does it in one pass with the
        // exact capacity, where iter().cloned().collect() could reallocate mid-way and
        // from_shape_vec then reshaped a second buffer.
        let contiguous = joined.as_standard_layout().into_owned().into_shared();
        Ok(OrtexTensor::$ort_tensor_kind(contiguous))
    }};
}

pub fn concatenate(
    tensors: Vec<ResourceArc<OrtexTensor>>,
    dtype: (&str, usize),
    axis: usize,
) -> Result<OrtexTensor, Error> {
    // An ORT-owned output has no ndarray for the macro below to view; materialize it once
    // for this concatenation. Array variants are shared, not copied.
    let tensors: Vec<ResourceArc<OrtexTensor>> = tensors
        .iter()
        .map(|tensor| match &**tensor {
            OrtexTensor::ort(value) => Ok(ResourceArc::new(OrtexTensor::try_from(value)?)),
            _ => Ok(tensor.clone()),
        })
        .collect::<Result<_, Error>>()?;

    match dtype {
        ("s", 8) => concatenate!(tensors, axis, i8, s8),
        ("s", 16) => concatenate!(tensors, axis, i16, s16),
        ("s", 32) => concatenate!(tensors, axis, i32, s32),
        ("s", 64) => concatenate!(tensors, axis, i64, s64),
        ("u", 8) => concatenate!(tensors, axis, u8, u8),
        ("u", 16) => concatenate!(tensors, axis, u16, u16),
        ("u", 32) => concatenate!(tensors, axis, u32, u32),
        ("u", 64) => concatenate!(tensors, axis, u64, u64),
        ("f", 16) => concatenate!(tensors, axis, half::f16, f16),
        ("bf", 16) => concatenate!(tensors, axis, half::bf16, bf16),
        ("f", 32) => concatenate!(tensors, axis, f32, f32),
        ("f", 64) => concatenate!(tensors, axis, f64, f64),
        (kind, bits) => Err(Error::new(format!(
            "Can't concatenate {}{} tensors",
            kind, bits
        ))),
    }
}
