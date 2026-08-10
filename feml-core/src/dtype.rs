//! Data types for tensors.
//!
//! The size/block-size table below is ported from ggml (`src/ggml.c` type_traits
//! table + `src/ggml-common.h` block structs, verified against their static_asserts):
//! - `type_size`:  bytes per block (for plain types: bytes per element, block == 1 element)
//! - `block_size`: number of elements per block
//! - row size in bytes = `type_size * n_elements / block_size`

use crate::error::{Error, Result};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u8)]
#[allow(non_camel_case_types)] // quantized names mirror ggml/GGUF conventions (q4_0, q4_k, ...)
pub enum DType {
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
    F64,
    // --- block-quantized (ggml-compatible layouts) ---
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
}

pub struct DTypeTraits {
    pub name: &'static str,
    /// Elements per block (ggml `blck_size`). 1 for plain types.
    pub block_size: usize,
    /// Bytes per block (ggml `type_size`).
    pub type_size: usize,
    pub quantized: bool,
}

impl DType {
    pub const COUNT: usize = 19;

    pub const fn traits(self) -> &'static DTypeTraits {
        // Table order MUST match the enum declaration order (indexed by discriminant);
        // the tests `plain_types_match_ggml` / `quantized_types_match_ggml` lock this.
        const TABLE: [DTypeTraits; DType::COUNT] = [
            DTypeTraits { name: "i8", block_size: 1, type_size: 1, quantized: false },
            DTypeTraits { name: "i16", block_size: 1, type_size: 2, quantized: false },
            DTypeTraits { name: "i32", block_size: 1, type_size: 4, quantized: false },
            DTypeTraits { name: "i64", block_size: 1, type_size: 8, quantized: false },
            DTypeTraits { name: "f16", block_size: 1, type_size: 2, quantized: false },
            DTypeTraits { name: "f32", block_size: 1, type_size: 4, quantized: false },
            DTypeTraits { name: "f64", block_size: 1, type_size: 8, quantized: false },
            DTypeTraits { name: "q4_0", block_size: 32, type_size: 18, quantized: true },
            DTypeTraits { name: "q4_1", block_size: 32, type_size: 20, quantized: true },
            DTypeTraits { name: "q5_0", block_size: 32, type_size: 22, quantized: true },
            DTypeTraits { name: "q5_1", block_size: 32, type_size: 24, quantized: true },
            DTypeTraits { name: "q8_0", block_size: 32, type_size: 34, quantized: true },
            DTypeTraits { name: "q8_1", block_size: 32, type_size: 36, quantized: true },
            DTypeTraits { name: "q2_k", block_size: 256, type_size: 84, quantized: true },
            DTypeTraits { name: "q3_k", block_size: 256, type_size: 110, quantized: true },
            DTypeTraits { name: "q4_k", block_size: 256, type_size: 144, quantized: true },
            DTypeTraits { name: "q5_k", block_size: 256, type_size: 176, quantized: true },
            DTypeTraits { name: "q6_k", block_size: 256, type_size: 210, quantized: true },
            DTypeTraits { name: "q8_k", block_size: 256, type_size: 292, quantized: true },
        ];
        &TABLE[self as usize]
    }

    pub const fn name(self) -> &'static str {
        self.traits().name
    }

    /// Elements per block. 1 for plain types, QK (32/256) for quantized types.
    pub const fn block_size(self) -> usize {
        self.traits().block_size
    }

    /// Bytes per block (bytes per element for plain types).
    pub const fn type_size(self) -> usize {
        self.traits().type_size
    }

    pub const fn is_quantized(self) -> bool {
        self.traits().quantized
    }

    pub const fn is_float(self) -> bool {
        matches!(self, Self::F16 | Self::F32 | Self::F64)
    }

    pub const fn is_int(self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }

    /// Bytes for a row of `n_elements` (ggml `ggml_row_size`).
    ///
    /// `n_elements` must be a multiple of `block_size`.
    pub fn row_size(self, n_elements: usize) -> Result<usize> {
        if n_elements % self.block_size() != 0 {
            return Err(Error::msg(format!(
                "{}: row size {} is not aligned to block size {}",
                self.name(),
                n_elements,
                self.block_size()
            )));
        }
        Ok(self.type_size() * n_elements / self.block_size())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lock the table against the ggml reference values (ggml.c type_traits).
    #[test]
    fn plain_types_match_ggml() {
        let cases = [
            (DType::I8, 1, 1, false),
            (DType::I16, 1, 2, false),
            (DType::I32, 1, 4, false),
            (DType::I64, 1, 8, false),
            (DType::F16, 1, 2, false),
            (DType::F32, 1, 4, false),
            (DType::F64, 1, 8, false),
        ];
        for (dt, blck, size, quant) in cases {
            assert_eq!(dt.block_size(), blck, "{dt:?} block_size");
            assert_eq!(dt.type_size(), size, "{dt:?} type_size");
            assert_eq!(dt.is_quantized(), quant, "{dt:?} quantized");
        }
    }

    #[test]
    fn quantized_types_match_ggml() {
        let cases = [
            (DType::Q4_0, 32, 18),
            (DType::Q4_1, 32, 20),
            (DType::Q5_0, 32, 22),
            (DType::Q5_1, 32, 24),
            (DType::Q8_0, 32, 34),
            (DType::Q8_1, 32, 36),
            (DType::Q2_K, 256, 84),
            (DType::Q3_K, 256, 110),
            (DType::Q4_K, 256, 144),
            (DType::Q5_K, 256, 176),
            (DType::Q6_K, 256, 210),
            (DType::Q8_K, 256, 292),
        ];
        for (dt, blck, size) in cases {
            assert_eq!(dt.block_size(), blck, "{dt:?} block_size");
            assert_eq!(dt.type_size(), size, "{dt:?} type_size");
            assert!(dt.is_quantized(), "{dt:?} should be quantized");
        }
    }

    #[test]
    fn row_size_plain() {
        assert_eq!(DType::F32.row_size(4096).unwrap(), 16384);
        assert_eq!(DType::F16.row_size(3).unwrap(), 6);
    }

    #[test]
    fn row_size_quantized() {
        // 64 f32 elements quantized to q4_0 = 2 blocks * 18 bytes
        assert_eq!(DType::Q4_0.row_size(64).unwrap(), 36);
        // 256 elements in one q4_k super-block
        assert_eq!(DType::Q4_K.row_size(256).unwrap(), 144);
    }

    #[test]
    fn row_size_rejects_misaligned() {
        assert!(DType::Q4_0.row_size(33).is_err());
    }
}
