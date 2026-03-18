//! Context module for managing tensors and compute graphs.
//!
//! This module provides the core context structures for tensor operations,
//! including memory management through object pools and table-based storage
//! for tensors and compute graphs.

use crate::compute_graph::{ ComputeGraph, GraphId };
use crate::data_type::{ DataType, get_block_size, get_type_size };
use crate::object_pool::ObjectPool;
use crate::shape::Shape;
use crate::tensor::{ self, Tensor, Tensor_, TensorId };
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use crate::error::Result;
use crate::error::{ Error, ErrorKind };

/// Internal context structure holding tensor and graph management data.
///
/// This struct contains the core data structures for managing tensors and compute graphs,
/// including an object pool for tensor instances and hash tables for lookup.
pub struct Context_ {
    /// Object pool for tensor instances to reduce allocation overhead.
    pub tensor_pool: ObjectPool<Tensor_>,
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
pub struct Context(Arc<RefCell<Context_>>);

impl AsRef<Context> for Context {
    fn as_ref(&self) -> &Context {
        self
    }
}

impl std::ops::Deref for Context {
    type Target = RefCell<Context_>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Context_ {
    /// Creates a new internal context with specified tensor pool capacity.
    ///
    /// @param size The initial capacity for the tensor object pool.
    /// @return Some(Context_) if successful, None if initialization fails.
    pub fn new(size: &usize) -> Option<Self> {
        (Self {
            tensor_pool: ObjectPool::with_capacity(Tensor_::default, *size),
            tensor_tables: HashMap::new(),
            graph_tables: HashMap::new(),
        }).into()
    }

    /// Internal implementation for creating a new tensor.
    ///
    /// This method handles the core logic of tensor creation, including validation,
    /// memory allocation, and layout computation.
    ///
    /// @param dtype The data type of the tensor elements.
    /// @param shape The shape (dimensions) of the tensor.
    /// @param view_src Optional source tensor for creating a view.
    /// @return Result containing the new tensor or an error.
    fn new_tensor_impl(
        self: &mut Self,
        dtype: DataType,
        shape: &Shape,
        view_src: Option<Tensor>
    ) -> Result<Tensor> {
        // Validate shape: ensure no dimension is zero
        if shape.0.iter().any(|&dim| dim == 0) {
            return Err(
                Error::new(ErrorKind::Msg("shape cannot contain zero dimensions".into())).context(
                    "in new_tensor_impl"
                )
            );
        }

        // Validate data type: check if supported for tensor creation
        if !matches!(dtype, DataType::F32 | DataType::I32) {
            return Err(
                Error::new(ErrorKind::UnsupportedDataTypeForOp {
                    dtype,
                    op: "tensor creation",
                })
            );
        }

        let mut tensor_ = self.tensor_pool.get(); // This operation always succeeds as ObjectPool provides objects

        tensor_.dtype = dtype;
        tensor_.layout.shape = shape.clone();
        tensor_.id = TensorId::new();

        // Handle view source if provided
        if let Some(src) = view_src {
            // Verify source tensor exists in the context
            if !self.tensor_tables.contains_key(&src.borrow().id) {
                return Err(
                    Error::msg("view source tensor not found in context").context(
                        "in new_tensor_impl"
                    )
                );
            }
            tensor_.storage = src.borrow().storage.clone();
        }

        // Calculate strides with overflow checks
        let type_size = get_type_size(dtype);
        if type_size == 0 {
            return Err(Error::msg("invalid data type size").context("in stride calculation"));
        }
        tensor_.layout.stride[0] = type_size;

        let block_size = get_block_size(dtype);
        if block_size == 0 {
            return Err(
                Error::msg("invalid block size for data type").context("in stride calculation")
            );
        }
        tensor_.layout.stride[1] =
            tensor_.layout.stride[0] * (tensor_.layout.stride[0] / block_size);

        // Calculate remaining strides with overflow protection
        for i in 2..4 {
            let next_stride = tensor_.layout.stride[i - 1]
                .checked_mul(tensor_.layout.shape.0[i - 1])
                .ok_or_else(||
                    Error::msg("stride calculation overflow").context("in stride calculation")
                )?;
            tensor_.layout.stride[i] = next_stride;
        }

        let tensor = Tensor(Arc::new(RefCell::new(tensor_)));
        self.tensor_tables.insert(tensor.borrow().id, tensor.clone());

        Ok(tensor)
    }
}

impl Context {
    /// Creates a new tensor with the specified data type and shape.
    ///
    /// This is the public API for tensor creation. It delegates to the internal
    /// implementation and handles error propagation.
    ///
    /// @param dtype The data type of the tensor elements.
    /// @param shape The shape (dimensions) of the tensor.
    /// @return Result containing the new tensor or an error.
    pub fn new_tensor(self: &mut Self, dtype: DataType, shape: &Shape) -> Result<Tensor> {
        let tensor = self.0.borrow_mut().new_tensor_impl(dtype, shape, None)?;
        tensor.borrow_mut().set_context(self.clone());
        Ok(tensor)
    }

    pub fn new_tensor_view(self: &mut Self, view_src: Tensor) -> Result<Tensor> {
        let tensor = self.0
            .borrow_mut()
            .new_tensor_impl(
                view_src.borrow().dtype,
                &view_src.borrow().layout.shape,
                Some(view_src.clone())
            )?;
        tensor.borrow_mut().set_context(self.clone());

        for i in 0..4 {
            tensor.borrow_mut().layout.stride[i] = view_src.borrow().layout.stride[i];
        }
        Ok(tensor)
    }

    /// Creates a new compute graph.
    ///
    /// @return Result containing the new compute graph or an error.
    /// @note This method is not yet implemented.
    pub fn new_graph(self: &Self) -> Result<ComputeGraph> {
        todo!();
    }

    /// Creates a new context with the specified tensor pool capacity.
    pub fn new(size: usize) -> Self {
        Context(Arc::new(RefCell::new(Context_::new(&size).unwrap())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_type::DataType;
    use crate::shape::Shape;

    #[test]
    fn test_context_new() {
        let ctx = Context::new(10);
        assert_eq!(ctx.borrow().tensor_tables.len(), 0);
        assert_eq!(ctx.borrow().graph_tables.len(), 0);
    }

    #[test]
    fn test_new_tensor_success() {
        let mut ctx = Context::new(10);
        let shape = Shape([2, 3, 4, 5]);
        let tensor = ctx.new_tensor(DataType::F32, &shape).unwrap();
        assert_eq!(tensor.borrow().dtype, DataType::F32);
        assert_eq!(tensor.borrow().layout.shape, shape);
        assert!(ctx.borrow().tensor_tables.contains_key(&tensor.borrow().id));
    }

    #[test]
    fn test_new_tensor_zero_dimension() {
        let mut ctx = Context::new(10);
        let shape = Shape([0, 3, 4, 5]);
        let result = ctx.new_tensor(DataType::F32, &shape);
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("zero dimensions"));
    }

    #[test]
    fn test_new_tensor_unsupported_dtype() {
        let mut ctx = Context::new(10);
        let shape = Shape([2, 3, 4, 5]);
        let result = ctx.new_tensor(DataType::U8, &shape);
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("unsupported dtype"));
    }

    #[test]
    fn test_new_tensor_view_success() {
        let mut ctx = Context::new(10);
        let shape = Shape([2, 3, 4, 5]);
        let src_tensor = ctx.new_tensor(DataType::I32, &shape).unwrap();
        let view_tensor = ctx.new_tensor_view(src_tensor.clone()).unwrap();
        assert_eq!(view_tensor.borrow().dtype, DataType::I32);
        assert_eq!(view_tensor.borrow().layout.shape, shape);
        for i in 0..4 {
            assert_eq!(view_tensor.borrow().layout.stride[i], src_tensor.borrow().layout.stride[i]);
        }
        assert!(ctx.borrow().tensor_tables.contains_key(&view_tensor.borrow().id));
    }

    #[test]
    fn test_new_tensor_view_invalid_source() {
        let mut ctx = Context::new(10);
        let mut other_ctx = Context::new(10);
        let shape = Shape([2, 3, 4, 5]);
        let src_tensor = other_ctx.new_tensor(DataType::F32, &shape).unwrap();
        let result = ctx.new_tensor_view(src_tensor);
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("view source tensor not found"));
    }

    #[test]
    #[should_panic(expected = "not yet implemented")]
    fn test_new_graph_panics() {
        let ctx = Context::new(10);
        let _ = ctx.new_graph();
    }
}
