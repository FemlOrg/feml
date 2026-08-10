//! Tensor memory layout: shape + byte strides (`nb` in ggml terms) + offset.
//!
//! Stride semantics follow ggml exactly (ggml.h tensor struct):
//! - `stride[0] = type_size` (bytes per block; == bytes per element for plain types)
//! - `stride[1] = stride[0] * (shape[0] / block_size)`
//! - `stride[i] = stride[i-1] * shape[i-1]` for i >= 2
//!
//! Note: ggml adds row padding (`GGML_PAD`) inside `stride[1]` in some paths;
//! feml-core keeps rows unpacked for now — padding is a backend/kernel concern.

use crate::dtype::DType;
use crate::error::Result;
use crate::shape::{MAX_DIMS, Shape};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Layout {
    pub shape: Shape,
    /// Byte stride per dimension (ggml `nb[4]`).
    pub stride: [usize; MAX_DIMS],
    /// Start offset of this tensor within its buffer (ggml `view_offs`).
    pub offset: usize,
}

impl Default for Layout {
    fn default() -> Self {
        Self { shape: Shape::default(), stride: [0; MAX_DIMS], offset: 0 }
    }
}

impl Layout {
    /// Compute the canonical (row-major, contiguous) layout for `dtype`/`shape`.
    pub fn new(dtype: DType, shape: Shape) -> Result<Self> {
        let type_size = dtype.type_size();
        let block_size = dtype.block_size();

        let mut stride = [0usize; MAX_DIMS];
        stride[0] = type_size;
        stride[1] = stride[0]
            .checked_mul(shape[0] / block_size)
            .ok_or_else(|| crate::error::Error::shape("stride[1] overflow"))?;
        for i in 2..MAX_DIMS {
            stride[i] = stride[i - 1]
                .checked_mul(shape[i - 1])
                .ok_or_else(|| crate::error::Error::shape("stride overflow"))?;
        }

        Ok(Self { shape, stride, offset: 0 })
    }

    /// Number of bytes required for this tensor (ggml `ggml_nbytes`).
    pub fn nbytes(&self, dtype: DType) -> usize {
        let block_size = dtype.block_size();
        let type_size = dtype.type_size();
        // bytes for dim 0: n_elements * bytes_per_block / elements_per_block
        let mut bytes = self.shape[0] * type_size / block_size;
        for i in 1..MAX_DIMS {
            bytes += (self.shape[i] - 1) * self.stride[i];
        }
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape;

    #[test]
    fn f32_stride_matches_ggml() {
        // shape [2,3]: nb0 = 4, nb1 = 4*2 = 8, nb2 = 8*3 = 24
        let l = Layout::new(DType::F32, shape![2, 3]).unwrap();
        assert_eq!(l.stride[0], 4);
        assert_eq!(l.stride[1], 8);
        assert_eq!(l.stride[2], 24);
    }

    /// Regression lock: the old code computed 10 bytes for a [2,3] F32 tensor.
    #[test]
    fn f32_nbytes_regression_lock() {
        let l = Layout::new(DType::F32, shape![2, 3]).unwrap();
        assert_eq!(l.nbytes(DType::F32), 24);
    }

    #[test]
    fn f32_nbytes_1d() {
        let l = Layout::new(DType::F32, shape![100]).unwrap();
        assert_eq!(l.nbytes(DType::F32), 400);
    }

    #[test]
    fn f32_nbytes_4d() {
        let l = Layout::new(DType::F32, shape![1, 3, 224, 224]).unwrap();
        assert_eq!(l.nbytes(DType::F32), 3 * 224 * 224 * 4);
    }

    #[test]
    fn quantized_nbytes() {
        // 64 elements of q4_0 = 2 blocks * 18 bytes
        let l = Layout::new(DType::Q4_0, shape![64]).unwrap();
        assert_eq!(l.nbytes(DType::Q4_0), 36);
        // 256 elements of q4_k = 1 super-block * 144 bytes
        let l = Layout::new(DType::Q4_K, shape![256]).unwrap();
        assert_eq!(l.nbytes(DType::Q4_K), 144);
    }

    #[test]
    fn quantized_stride() {
        // [64, 2] q4_0: row = 2 blocks = 36 bytes; stride[1] = 18 * (64/32) = 36
        let l = Layout::new(DType::Q4_0, shape![64, 2]).unwrap();
        assert_eq!(l.stride[0], 18);
        assert_eq!(l.stride[1], 36);
        assert_eq!(l.nbytes(DType::Q4_0), 36 + 36);
    }

    #[test]
    fn offset_is_zero_for_fresh_layout() {
        let l = Layout::new(DType::F32, shape![2, 2]).unwrap();
        assert_eq!(l.offset, 0);
    }
}
