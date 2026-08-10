//! Strided elementwise binary ops (mul/add) with ggml-style broadcasting.
//!
//! The kernel takes raw pointers: operands may share one buffer (inplace
//! aliasing), which safe Rust slices cannot express. All bounds checks are
//! done in safe code before any dereference; within each iteration the read
//! happens strictly before the write, so full aliasing is safe.

use feml_core::error::{Error, Result};
use feml_core::layout::Layout;

pub(crate) struct Operand<'a> {
    pub ptr: *const u8,
    pub len: usize,
    pub offset: usize,
    pub layout: &'a Layout,
}

pub(crate) struct OutOperand<'a> {
    pub ptr: *mut u8,
    pub len: usize,
    pub offset: usize,
    pub layout: &'a Layout,
}

/// # Safety
/// - `a`/`b`/`c` must point to live buffers of at least `len` bytes, for the
///   duration of the call.
/// - All offset+access computations below are bounds-checked before deref.
/// - `c` may fully alias `a` or `b` (identical byte range); partial overlap
///   must be rejected by the caller.
pub(crate) unsafe fn elementwise_binary(
    a: &Operand<'_>,
    b: &Operand<'_>,
    c: &OutOperand<'_>,
    is_mul: bool,
) -> Result<()> {
    let c_layout = c.layout;
    let a_layout = a.layout;
    let b_layout = b.layout;
    let c_nbytes = c_layout.nbytes(feml_core::dtype::DType::F32);
    if c.offset + c_nbytes > c.len {
        return Err(Error::shape("elementwise: dst out of bounds"));
    }
    let a_nbytes = a_layout.nbytes(feml_core::dtype::DType::F32);
    let b_nbytes = b_layout.nbytes(feml_core::dtype::DType::F32);
    if a.offset + a_nbytes > a.len || b.offset + b_nbytes > b.len {
        return Err(Error::shape("elementwise: source out of bounds"));
    }

    let sa = a_layout.shape;
    let sb = b_layout.shape;
    let sc = c_layout.shape;

    for i3 in 0..sc[3] {
        for i2 in 0..sc[2] {
            for i1 in 0..sc[1] {
                for i0 in 0..sc[0] {
                    let (ai0, ai1, ai2, ai3) = broadcast(&sa, i0, i1, i2, i3);
                    let (bi0, bi1, bi2, bi3) = broadcast(&sb, i0, i1, i2, i3);
                    let off_a = a.offset
                        + ai0 * a_layout.stride[0]
                        + ai1 * a_layout.stride[1]
                        + ai2 * a_layout.stride[2]
                        + ai3 * a_layout.stride[3];
                    let off_b = b.offset
                        + bi0 * b_layout.stride[0]
                        + bi1 * b_layout.stride[1]
                        + bi2 * b_layout.stride[2]
                        + bi3 * b_layout.stride[3];
                    let off_c = c.offset
                        + i0 * c_layout.stride[0]
                        + i1 * c_layout.stride[1]
                        + i2 * c_layout.stride[2]
                        + i3 * c_layout.stride[3];
                    // SAFETY: offsets are within the checked ranges above.
                    let x = unsafe { *(a.ptr.add(off_a) as *const f32) };
                    let y = unsafe { *(b.ptr.add(off_b) as *const f32) };
                    let z = if is_mul { x * y } else { x + y };
                    unsafe {
                        *(c.ptr.add(off_c) as *mut f32) = z;
                    }
                }
            }
        }
    }
    Ok(())
}

fn broadcast(
    shape: &feml_core::shape::Shape,
    i0: usize,
    i1: usize,
    i2: usize,
    i3: usize,
) -> (usize, usize, usize, usize) {
    (
        if shape[0] == 1 { 0 } else { i0 },
        if shape[1] == 1 { 0 } else { i1 },
        if shape[2] == 1 { 0 } else { i2 },
        if shape[3] == 1 { 0 } else { i3 },
    )
}
