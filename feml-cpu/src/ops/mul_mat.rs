//! f32 matrix multiply (ggml mul_mat semantics, single-threaded reference).
//!
//! dst[m, n] = sum_k a[k, m] * b[k, n]  (a: [K, M], b: [K, N], dst: [M, N])

use feml_core::error::{Error, Result};

use super::elementwise::{Operand, OutOperand};

/// # Safety
/// - `a`/`b`/`c` must point to live buffers of at least `len` bytes, for the
///   duration of the call.
/// - The caller must guarantee `c` does not overlap `a` or `b` (the planner
///   never inplaces mul_mat; dispatch rejects overlapping ranges).
pub(crate) unsafe fn mul_mat(a: &Operand<'_>, b: &Operand<'_>, c: &OutOperand<'_>) -> Result<()> {
    let a_layout = a.layout;
    let b_layout = b.layout;
    let c_layout = c.layout;
    let k = a_layout.shape[0];
    let m = a_layout.shape[1];
    let n = b_layout.shape[1];
    if c_layout.shape[0] != m || c_layout.shape[1] != n {
        return Err(Error::shape(format!(
            "mul_mat: dst {}x{} but expected {}x{}",
            c_layout.shape[0], c_layout.shape[1], m, n
        )));
    }
    if c.offset + c_layout.nbytes(feml_core::dtype::DType::F32) > c.len {
        return Err(Error::shape("mul_mat: dst out of bounds"));
    }
    if a.offset + a_layout.nbytes(feml_core::dtype::DType::F32) > a.len
        || b.offset + b_layout.nbytes(feml_core::dtype::DType::F32) > b.len
    {
        return Err(Error::shape("mul_mat: source out of bounds"));
    }

    for mm in 0..m {
        for nn in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                // SAFETY: kk/mm/nn are within the checked shapes above.
                let x = unsafe {
                    *(a.ptr.add(a.offset + kk * a_layout.stride[0] + mm * a_layout.stride[1])
                        as *const f32)
                };
                let y = unsafe {
                    *(b.ptr.add(b.offset + kk * b_layout.stride[0] + nn * b_layout.stride[1])
                        as *const f32)
                };
                acc += x * y;
            }
            // SAFETY: within the checked dst range.
            unsafe {
                *(c.ptr.add(c.offset + mm * c_layout.stride[0] + nn * c_layout.stride[1])
                    as *mut f32) = acc;
            }
        }
    }
    Ok(())
}
