//! Context module for managing tensors and compute graphs.
//!
//! This module provides the core context structures for tensor operations,
//! including memory management through object pools and table-based storage
//! for tensors and compute graphs.

use crate::compute_graph::{ComputeGraph, ComputeGraphInner, GraphId};
use crate::data_type::{get_block_size, get_type_size, DataType};
use crate::defs::MAX_DIMS;
use crate::error::Result;
use crate::error::{Error, ErrorKind};
use crate::object_pool::ObjectPool;
use crate::shape::Shape;
use crate::tensor::{Tensor, TensorId, TensorInner};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ContextConfig {
    pub tensor_pool_capacity: usize,
    pub graph_pool_cacacity: usize,
}

#[derive(Debug, Clone)]
pub struct ContextBuilder {
    config: ContextConfig,
}

impl Default for ContextBuilder {
    fn default() -> Self {
        Self { config: ContextConfig::default() }
    }
}

impl ContextBuilder {
    pub fn tensor_pool_capacity(mut self, capacity: usize) -> Self {
        self.config.tensor_pool_capacity = capacity;
        self
    }

    pub fn graph_pool_capacity(mut self, capacity: usize) -> Self {
        self.config.graph_pool_cacacity = capacity;
        self
    }

    pub fn build(self) -> Context {
        Context::with_config(self.config)
    }
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self { tensor_pool_capacity: 1024, graph_pool_cacacity: 0 }
    }
}

/// Internal context structure holding tensor and graph management data.
///
/// This struct contains the core data structures for managing tensors and compute graphs,
/// including an object pool for tensor instances and hash tables for lookup.
pub struct ContextInner {
    /// Object pool for tensor instances to reduce allocation overhead.
    pub tensor_pool: ObjectPool<TensorInner>,
    pub graph_pool: ObjectPool<ComputeGraphInner>,
    /// Hash table mapping tensor IDs to tensor objects.
    pub tensor_tables: HashMap<TensorId, Tensor>,
    /// Hash table mapping graph IDs to compute graph objects.
    pub graph_tables: HashMap<GraphId, ComputeGraph>,
}

/// Public context wrapper providing thread-safe access to the internal context.
///
/// This struct wraps the internal `Context_` in an `Arc<RefCell<>>` to allow
/// shared mutable access across threads while maintaining interior mutability.
#[derive(Clone)]
pub struct Context(pub(crate) Rc<RefCell<ContextInner>>);

impl AsRef<Context> for Context {
    fn as_ref(&self) -> &Context {
        self
    }
}

impl std::ops::Deref for Context {
    type Target = RefCell<ContextInner>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl ContextInner {
    /// Creates a new internal context with specified tensor pool capacity.
    ///
    /// @param size The initial capacity for the tensor object pool.
    /// @return Some(Context_) if successful, None if initialization fails.
    fn new(config: ContextConfig) -> Option<Self> {
        (Self {
            tensor_pool: ObjectPool::with_capacity(
                TensorInner::default,
                config.tensor_pool_capacity,
            ),
            graph_pool: ObjectPool::with_capacity(
                ComputeGraphInner::default,
                config.graph_pool_cacacity,
            ),
            tensor_tables: HashMap::new(),
            graph_tables: HashMap::new(),
        })
        .into()
    }
}

impl Context {
    fn new_tensor_impl(
        self: &mut Self,
        dtype: DataType,
        shape: &Shape,
        view_src: Option<Tensor>,
    ) -> Result<Tensor> {
        // Validate shape: ensure no dimension is zero
        if shape.iter().any(|&dim| dim == 0) {
            return Err(Error::new(ErrorKind::Msg("shape cannot contain zero dimensions".into()))
                .context("in new_tensor_impl"));
        }

        // Validate data type: check if supported for tensor creation
        if !matches!(dtype, DataType::F32 | DataType::I32) {
            return Err(Error::new(ErrorKind::UnsupportedDataTypeForOp {
                dtype,
                op: "tensor creation",
            }));
        }

        let mut tensor_inner = self.borrow_mut().tensor_pool.get();

        tensor_inner.dtype = dtype;
        tensor_inner.layout.shape = shape.clone();
        tensor_inner.id = TensorId::new();
        tensor_inner.ctx = Rc::downgrade(&self.0);

        // Handle view source if provided
        if let Some(src) = view_src {
            if !self.contain_tensor(src.tensor_id()) {
                return Err(Error::msg("view source tensor not found in context")
                    .context("in new_tensor_impl"));
            }

            tensor_inner.view_offset += src.borrow().view_offset;

            if let Some(src_view) = src.borrow().view_tensor.clone() {
                tensor_inner.storage = src_view.borrow().storage.clone();
                tensor_inner.view_tensor = Some(src_view);
            } else {
                tensor_inner.storage = src.borrow().storage.clone();
                tensor_inner.view_tensor = Some(src.clone());
            }
        }

        // Calculate strides with overflow checks
        let type_size = get_type_size(dtype);
        if type_size == 0 {
            return Err(Error::msg("invalid data type size").context("in stride calculation"));
        }
        tensor_inner.layout.stride[0] = type_size;

        let block_size = get_block_size(dtype);
        if block_size == 0 {
            return Err(
                Error::msg("invalid block size for data type").context("in stride calculation")
            );
        }
        tensor_inner.layout.stride[1] =
            tensor_inner.layout.stride[0] * (tensor_inner.layout.stride[0] / block_size);

        // Calculate remaining strides with overflow protection
        for i in 2..MAX_DIMS {
            let next_stride = tensor_inner.layout.stride[i - 1]
                .checked_mul(tensor_inner.layout.shape[i - 1])
                .ok_or_else(|| {
                    Error::msg("stride calculation overflow").context("in stride calculation")
                })?;
            tensor_inner.layout.stride[i] = next_stride;
        }

        let tensor = Tensor(Rc::new(RefCell::new(tensor_inner)));
        self.borrow_mut().tensor_tables.insert(tensor.tensor_id(), tensor.clone());

        Ok(tensor)
    }

    pub fn new_graph(&mut self, size: usize) -> Result<ComputeGraph> {
        let mut graph_inner = self.borrow_mut().graph_pool.get();
        let id = graph_inner.id;
        graph_inner.size = size;
        graph_inner.nodes.reserve(size);
        graph_inner.leafs.reserve(size);

        let graph = ComputeGraph(Rc::new(RefCell::new(graph_inner)));

        self.borrow_mut().graph_tables.insert(id, graph.clone());

        Ok(graph)
    }
    /// Creates a new tensor with the specified data type and shape.
    ///
    /// This is the public API for tensor creation. It delegates to the internal
    /// implementation and handles error propagation.
    ///
    /// @param dtype The data type of the tensor elements.
    /// @param shape The shape (dimensions) of the tensor.
    /// @return Result containing the new tensor or an error.
    pub fn new_tensor(self: &mut Self, dtype: DataType, shape: &Shape) -> Result<Tensor> {
        let mut tensor = self.new_tensor_impl(dtype, shape, None)?;
        Ok(tensor)
    }

    pub fn new_tensor_view(self: &mut Self, view_src: Tensor) -> Result<Tensor> {
        let dtype = view_src.dtype();
        let shape = view_src.shape();
        let stride = view_src.borrow().layout.stride;

        let tensor = self.new_tensor_impl(dtype, &shape, Some(view_src.clone()))?;

        tensor.borrow_mut().layout.stride = stride;

        Ok(tensor)
    }

    /// Creates a new tensor by duplicating the shape and data type of an existing tensor.
    /// This method is a convenience wrapper around `new_tensor` that extracts the necessary
    /// information from the source tensor.
    pub fn dup_tensor(self: &mut Self, src: Tensor) -> Result<Tensor> {
        self.new_tensor(src.dtype(), &src.shape())
    }

    pub fn contain_tensor(&self, tensor_id: TensorId) -> bool {
        self.borrow().tensor_tables.contains_key(&tensor_id)
    }

    pub fn with_config(config: ContextConfig) -> Self {
        Context(Rc::new(RefCell::new(ContextInner::new(config).unwrap())))
    }

    pub fn builder() -> ContextBuilder {
        ContextBuilder::default()
    }

    pub fn get_tensor(&self, tensor_id: TensorId) -> Result<Tensor> {
        self.borrow().tensor_tables.get(&tensor_id).cloned().ok_or_else(|| {
            Error::msg(format!("tensor {} not found", tensor_id.as_usize()))
                .context("in Context::get_tensor")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_type::DataType;
    use crate::shape;
    use crate::shape::Shape;

    #[test]
    fn test_context_new() {
        let ctx = Context::builder().tensor_pool_capacity(10).build();
        assert_eq!(ctx.borrow().tensor_tables.len(), 0);
        assert_eq!(ctx.borrow().graph_tables.len(), 0);
    }

    #[test]
    fn test_new_tensor_success() {
        let mut ctx = Context::builder().tensor_pool_capacity(10).build();
        let shape = shape![2, 3, 4, 5];
        let tensor = ctx.new_tensor(DataType::F32, &shape).unwrap();
        assert_eq!(tensor.borrow().dtype, DataType::F32);
        assert_eq!(tensor.borrow().layout.shape, shape);
        assert!(ctx.contain_tensor(tensor.tensor_id()));
    }

    #[test]
    fn test_new_tensor_zero_dimension() {
        let mut ctx = Context::builder().tensor_pool_capacity(10).build();
        let shape = shape![0, 3, 4, 5];
        let result = ctx.new_tensor(DataType::F32, &shape);
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("zero dimensions"));
    }

    #[test]
    fn test_new_tensor_unsupported_dtype() {
        let mut ctx = Context::builder().tensor_pool_capacity(10).build();
        let shape = shape![2, 3, 4, 5];
        let result = ctx.new_tensor(DataType::U8, &shape);
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("unsupported dtype"));
    }

    #[test]
    fn test_new_tensor_view_success() {
        let mut ctx = Context::builder().tensor_pool_capacity(10).build();
        let shape = shape![2, 3, 4, 5];
        let src_tensor = ctx.new_tensor(DataType::I32, &shape).unwrap();
        assert!(ctx.contain_tensor(src_tensor.tensor_id()));
        let view_tensor = ctx.new_tensor_view(src_tensor.clone()).unwrap();
        assert_eq!(view_tensor.borrow().dtype, DataType::I32);
        assert_eq!(view_tensor.borrow().layout.shape, shape);
        for i in 0..MAX_DIMS {
            assert_eq!(view_tensor.borrow().layout.stride[i], src_tensor.borrow().layout.stride[i]);
        }
        assert!(ctx.contain_tensor(view_tensor.tensor_id()));
    }

    #[test]
    fn test_new_tensor_view_invalid_source() {
        let mut ctx = Context::builder().tensor_pool_capacity(10).build();
        let mut other_ctx = Context::builder().tensor_pool_capacity(10).build();
        let shape = shape![2, 3, 4, 5];
        let src_tensor = other_ctx.new_tensor(DataType::F32, &shape).unwrap();
        let result = ctx.new_tensor_view(src_tensor);
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("view source tensor not found"));
    }
}
