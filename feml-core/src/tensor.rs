//! Tensor identity and metadata.
//!
//! Design: a `TensorId` is a Copy handle; metadata lives in a flat
//! `TensorMeta` struct (ggml-style: fixed arrays, no per-tensor heap).
//! Data buffers and graph planning arrive with the GraphBuilder (M1).

use crate::dtype::DType;
use crate::error::Result;
use crate::layout::Layout;
use crate::op::Op;
use crate::shape::Shape;
use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};

/// ggml `GGML_MAX_SRC`.
pub const MAX_SRC: usize = 10;

static NEXT_TENSOR_ID: AtomicU32 = AtomicU32::new(1);

/// Unique, copyable tensor identifier.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TensorId(u32);

impl TensorId {
    pub fn new() -> Self {
        Self(NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed))
    }

    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

impl Default for TensorId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TensorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Tensor role flags (ggml `GGML_TENSOR_FLAG_*`), reduced to the inference set.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub struct TensorFlags(u8);

impl TensorFlags {
    pub const INPUT: u8 = 1 << 0;
    pub const OUTPUT: u8 = 1 << 1;
    /// Weight / constant: allocated once, never reused by the planner.
    pub const WEIGHT: u8 = 1 << 2;

    pub fn new() -> Self {
        Self(0)
    }

    pub fn with(mut self, flag: u8) -> Self {
        self.0 |= flag;
        self
    }

    pub fn has(&self, flag: u8) -> bool {
        self.0 & flag != 0
    }
}

/// Flat tensor metadata (ggml `struct ggml_tensor` analog).
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub id: TensorId,
    pub name: String,
    pub dtype: DType,
    pub layout: Layout,
    pub op: Op,
    /// Fixed-size source list (ggml `src[GGML_MAX_SRC]`).
    pub srcs: [Option<TensorId>; MAX_SRC],
    pub n_srcs: usize,
    /// View parent and byte offset into it (ggml `view_src`/`view_offs`).
    pub view_src: Option<TensorId>,
    pub flags: TensorFlags,
}

impl TensorMeta {
    pub fn new(name: impl Into<String>, dtype: DType, shape: Shape) -> Result<Self> {
        let layout = Layout::new(dtype, shape)?;
        Ok(Self {
            id: TensorId::new(),
            name: name.into(),
            dtype,
            layout,
            op: Op::None,
            srcs: [None; MAX_SRC],
            n_srcs: 0,
            view_src: None,
            flags: TensorFlags::new(),
        })
    }

    pub fn set_src(mut self, src: TensorId) -> Self {
        debug_assert!(self.n_srcs < MAX_SRC, "too many source tensors");
        self.srcs[self.n_srcs] = Some(src);
        self.n_srcs += 1;
        self
    }

    pub fn srcs(&self) -> &[Option<TensorId>] {
        &self.srcs[..self.n_srcs]
    }

    pub fn nbytes(&self) -> usize {
        self.layout.nbytes(self.dtype)
    }

    pub fn shape(&self) -> Shape {
        self.layout.shape
    }

    /// The view chain root: self if not a view.
    pub fn root(&self) -> TensorId {
        self.view_src.unwrap_or(self.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape;

    #[test]
    fn tensor_id_unique_and_copy() {
        let a = TensorId::new();
        let b = TensorId::new();
        let c = a; // Copy
        assert_ne!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn meta_new_computes_layout() {
        let t = TensorMeta::new("x", DType::F32, shape![2, 3]).unwrap();
        assert_eq!(t.nbytes(), 24);
        assert_eq!(t.shape(), shape![2, 3]);
        assert!(t.srcs().is_empty());
        assert!(!t.flags.has(TensorFlags::OUTPUT));
    }

    #[test]
    fn meta_srcs_and_view() {
        let a = TensorMeta::new("a", DType::F32, shape![4]).unwrap();
        let b = TensorMeta::new("b", DType::F32, shape![4]).unwrap();
        let mut m = TensorMeta::new("out", DType::F32, shape![4]).unwrap();
        m = m.set_src(a.id).set_src(b.id);
        assert_eq!(m.srcs(), &[Some(a.id), Some(b.id)]);
        assert_eq!(m.root(), m.id);

        let mut v = TensorMeta::new("v", DType::F32, shape![2]).unwrap();
        v.view_src = Some(a.id);
        v.layout.offset = 8;
        assert_eq!(v.root(), a.id);
    }

    #[test]
    fn flags() {
        let f = TensorFlags::new().with(TensorFlags::WEIGHT);
        assert!(f.has(TensorFlags::WEIGHT));
        assert!(!f.has(TensorFlags::OUTPUT));
    }
}
