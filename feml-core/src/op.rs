//! Inference-focused operation set.
//!
//! Parameters are stored inline in the enum variants (no heap allocation,
//! mirroring ggml's fixed `op_params` array). Backends dispatch on `Op` via
//! `supports_op` / their kernel registry.

use crate::error::{Error, Result};
use crate::shape::Shape;

#[derive(Debug, Copy, Clone, PartialEq)]
#[non_exhaustive]
pub enum Op {
    /// Leaf: input, weight or KV-cache buffer. No computation.
    None,
    /// Zero-copy view into a source tensor.
    View,
    /// Element-wise multiply (broadcastable, ggml-compatible).
    Mul,
    /// Element-wise add (broadcastable).
    Add,
    /// Matrix multiply: dst = src0^T · src1 (src0 is a weight, ggml convention).
    MulMat,
    /// Grouped matrix multiply for MoE: selected experts per row.
    MulMatId { n_experts: u32, n_selected: u32 },
    /// RMS normalization.
    RmsNorm { eps: f32 },
    /// Softmax along `axis`.
    Softmax { axis: i32 },
    /// Rotary position embedding. srcs: (x, pos).
    Rope { n_dims: u32, mode: u32, n_ctx: u32, freq_base: f32, freq_scale: f32 },
    /// SiLU activation (LLaMA feed-forward).
    Silu,
    /// Gather rows by index tensor (embedding lookup). srcs: (table, idx).
    GetRows,
    /// Concatenate along `axis`.
    Concat { axis: i32 },
    /// Cross-device / cross-buffer copy.
    Copy,
    /// Mask the upper triangle of the last two dims with -inf (causal).
    DiagMaskInf { n_past: i32 },
    /// Element-wise scaling.
    Scale { alpha: f32 },
}

/// Rotary position embedding parameters (LLaMA linear scaling, ggml-compatible).
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct RopeParams {
    pub n_dims: u32,
    pub mode: u32,
    pub n_ctx: u32,
    pub freq_base: f32,
    pub freq_scale: f32,
}

impl Default for RopeParams {
    fn default() -> Self {
        Self { n_dims: 0, mode: 0, n_ctx: 0, freq_base: 10000.0, freq_scale: 1.0 }
    }
}

impl Op {
    pub fn name(&self) -> &'static str {
        match self {
            Op::None => "none",
            Op::View => "view",
            Op::Mul => "mul",
            Op::Add => "add",
            Op::MulMat => "mul_mat",
            Op::MulMatId { .. } => "mul_mat_id",
            Op::RmsNorm { .. } => "rms_norm",
            Op::Softmax { .. } => "softmax",
            Op::Rope { .. } => "rope",
            Op::Silu => "silu",
            Op::GetRows => "get_rows",
            Op::Concat { .. } => "concat",
            Op::Copy => "copy",
            Op::DiagMaskInf { .. } => "diag_mask_inf",
            Op::Scale { .. } => "scale",
        }
    }

    /// Number of source tensors this op consumes (ggml `ggml_nsrc` analog).
    pub fn n_srcs(&self) -> usize {
        match self {
            Op::None | Op::View => 1,
            Op::Mul | Op::Add | Op::MulMat | Op::Concat { .. } | Op::Copy => 2,
            Op::MulMatId { .. } => 3,
            Op::Rope { .. } | Op::GetRows => 2,
            Op::RmsNorm { .. }
            | Op::Softmax { .. }
            | Op::Silu
            | Op::DiagMaskInf { .. }
            | Op::Scale { .. } => 1,
        }
    }

    /// Whether the op may overwrite its first source in place.
    pub fn can_inplace(&self) -> bool {
        matches!(
            self,
            Op::Mul
                | Op::Add
                | Op::RmsNorm { .. }
                | Op::Silu
                | Op::Softmax { .. }
                | Op::Rope { .. }
                | Op::DiagMaskInf { .. }
                | Op::Scale { .. }
        )
    }
}

/// Shape-inference for binary broadcast ops.
///
/// Trailing (innermost) dims align, ggml-style: `[3]` broadcasts over `[2, 3]`.
pub fn broadcast_shape(a: &Shape, b: &Shape) -> Result<Shape> {
    let rank = a.rank().max(b.rank());
    let mut dims = [1usize; crate::shape::MAX_DIMS];
    for back in 0..rank {
        let da = if back < a.rank() { a[a.rank() - 1 - back] } else { 1 };
        let db = if back < b.rank() { b[b.rank() - 1 - back] } else { 1 };
        dims[rank - 1 - back] = if da == db {
            da
        } else if da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            return Err(Error::shape(format!(
                "cannot broadcast dim {}: {da} vs {db} (a={a:?}, b={b:?})",
                rank - 1 - back
            )));
        };
    }
    Shape::new(&dims[..rank])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape;

    #[test]
    fn names() {
        assert_eq!(Op::Mul.name(), "mul");
        assert_eq!(Op::MulMat {}.name(), "mul_mat");
        assert_eq!(Op::None.name(), "none");
        assert_eq!(Op::Scale { alpha: 2.0 }.name(), "scale");
        assert_eq!(Op::DiagMaskInf { n_past: 0 }.name(), "diag_mask_inf");
    }

    #[test]
    fn n_srcs() {
        assert_eq!(Op::Mul.n_srcs(), 2);
        assert_eq!(Op::RmsNorm { eps: 1e-5 }.n_srcs(), 1);
        assert_eq!(Op::MulMatId { n_experts: 8, n_selected: 2 }.n_srcs(), 3);
        assert_eq!(
            Op::Rope { n_dims: 64, mode: 0, n_ctx: 0, freq_base: 1e4, freq_scale: 1.0 }.n_srcs(),
            2
        );
        assert_eq!(Op::GetRows.n_srcs(), 2);
        assert_eq!(Op::DiagMaskInf { n_past: 4 }.n_srcs(), 1);
    }

    #[test]
    fn can_inplace_flags() {
        assert!(Op::RmsNorm { eps: 1e-5 }.can_inplace());
        assert!(
            Op::Rope { n_dims: 64, mode: 0, n_ctx: 0, freq_base: 1e4, freq_scale: 1.0 }
                .can_inplace()
        );
        assert!(Op::Scale { alpha: 0.5 }.can_inplace());
        assert!(!Op::MulMat.can_inplace());
        assert!(!Op::View.can_inplace());
        assert!(!Op::GetRows.can_inplace());
    }

    #[test]
    fn broadcast_same_shape() {
        assert_eq!(broadcast_shape(&shape![2, 3], &shape![2, 3]).unwrap(), shape![2, 3]);
    }

    #[test]
    fn broadcast_row() {
        assert_eq!(broadcast_shape(&shape![2, 3], &shape![3]).unwrap(), shape![2, 3]);
        assert_eq!(broadcast_shape(&shape![3], &shape![2, 3]).unwrap(), shape![2, 3]);
    }

    #[test]
    fn broadcast_scalar() {
        assert_eq!(broadcast_shape(&shape![2, 3], &shape![1]).unwrap(), shape![2, 3]);
    }

    #[test]
    fn broadcast_mismatch_errors() {
        assert!(broadcast_shape(&shape![2, 4], &shape![2, 3]).is_err());
    }
}
