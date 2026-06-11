//! Context module for managing tensors, compute graphs and backend resources.
//!
//! Context is the primary user-facing abstraction. It owns:
//! - A Backend (execution engine: CPU/OpenCL/CUDA)
//! - A BackendBuffer (device memory)
//! - An object pool for Tensor metadata
//! - A tensor/graph registry for ID-based lookup
//!
//! Three entry levels:
//!   L1: Context::auto()                    — auto-select best backend
//!   L2: Context::with_backend("opencl")    — named backend
//!   L3: Context::builder().with_backend(b).build() — full control

use crate::backend::{Backend, BackendBuffer, BackendBufferUsage, BackendDevice};
use crate::compute_graph::{ComputeGraph, GraphId};
use crate::data_type::{get_block_size, get_type_size, DataType};
use crate::defs::MAX_DIMS;
use crate::error::Result;
use crate::error::{Error, ErrorKind};
use crate::object_pool::ObjectPool;
use crate::registry::Registry;
use crate::shape::Shape;
use crate::tensor::{Tensor, TensorId, TensorInner};
use std::cell::{Ref, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

// ─── Builder Configuration ────────────────────────────────────────

/// How the backend is sourced during Context creation.
enum BackendSource {
    /// L1: auto-select the best available backend.
    Auto,
    /// L2: look up by name via Registry.
    Named(String),
    /// L3: caller provides pre-built backend.
    Provided(Box<dyn Backend>),
}

/// Fluent builder for Context.
pub struct ContextBuilder {
    tensor_pool_capacity: usize,
    graph_pool_capacity: usize,
    buffer_size: usize,
    buffer_usage: BackendBufferUsage,
    backend_source: BackendSource,
}

impl Default for ContextBuilder {
    fn default() -> Self {
        Self {
            tensor_pool_capacity: 1024,
            graph_pool_capacity: 10,
            buffer_size: 256 * 1024 * 1024, // 256 MB
            buffer_usage: BackendBufferUsage::Any,
            backend_source: BackendSource::Auto,
        }
    }
}

impl ContextBuilder {
    /// Set the object-pool capacity for TensorInner reuse.
    pub fn tensor_pool_capacity(mut self, cap: usize) -> Self {
        self.tensor_pool_capacity = cap;
        self
    }

    /// Set the pre-allocated buffer size in bytes.
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set buffer usage hint (weights / compute / any).
    pub fn buffer_usage(mut self, usage: BackendBufferUsage) -> Self {
        self.buffer_usage = usage;
        self
    }

    // ─── Backend selection ─────────────────────────────────────

    /// **L2 entry**: specify a backend name (e.g. `"opencl"`, `"cpu"`).
    /// Builder will discover it via `Registry`, initialise it,
    /// and create a buffer — all inside `build()`.
    pub fn backend(mut self, name: impl Into<String>) -> Self {
        self.backend_source = BackendSource::Named(name.into());
        self
    }

    /// **L3 entry**: inject an already-built `Backend`.
    /// Builder will create a buffer from it during `build()`.
    ///
    /// Use this when you need full control over backend initialisation
    /// (e.g. custom device selection, multi-GPU, dynamic loading).
    pub fn with_backend(mut self, backend: Box<dyn Backend>) -> Self {
        self.backend_source = BackendSource::Provided(backend);
        self
    }

    // ─── Finalise ──────────────────────────────────────────────

    /// Consume the builder and produce a ready-to-use `Context`.
    pub fn build(self) -> Result<Context> {
        let (backend, buffer) = match self.backend_source {
            // ── L3: caller already built the Backend ───────────
            BackendSource::Provided(backend) => {
                let buffer = backend.create_buffer(self.buffer_size, self.buffer_usage)?;
                (backend, buffer)
            }

            // ── L1 / L2: discover via Registry ────────────────
            ref source @ (BackendSource::Auto | BackendSource::Named(_)) => {
                let registry = Registry::discover()?;

                let reg = match source {
                    BackendSource::Named(name) => registry
                        .find(name)
                        .ok_or_else(|| Error::msg(format!("backend '{}' not found", name)))?,
                    BackendSource::Auto => registry.best().ok_or_else(|| {
                        Error::msg("no backend available (tried CUDA > OpenCL > CPU)")
                    })?,
                    _ => unreachable!(),
                };

                reg.init_devices()?;
                let device = reg.device(0)?;
                let backend = device.init_backend()?;
                let buffer = backend.create_buffer(self.buffer_size, self.buffer_usage)?;

                (backend, buffer)
            }
        };

        let inner = ContextInner {
            tensor_pool: ObjectPool::with_capacity(TensorInner::default, self.tensor_pool_capacity),
            tensor_tables: HashMap::new(),
            graph_tables: HashMap::new(),
            backend: Some(backend),
            buffer: Some(buffer),
        };

        Ok(Context(Rc::new(RefCell::new(inner))))
    }
}

// ─── ContextInner ──────────────────────────────────────────────────

pub struct ContextInner {
    pub tensor_pool: ObjectPool<TensorInner>,
    pub tensor_tables: HashMap<TensorId, Tensor>,
    pub graph_tables: HashMap<GraphId, ComputeGraph>,
    pub backend: Option<Box<dyn Backend>>,
    pub buffer: Option<Box<dyn BackendBuffer>>,
}

impl ContextInner {
    fn contain_tensor_impl(&self, tensor_id: TensorId) -> bool {
        self.tensor_tables.contains_key(&tensor_id)
    }

    // ─── Tensor creation (metadata only, no data allocation) ────

    fn new_tensor_impl(
        self: &mut Self,
        dtype: DataType,
        shape: &Shape,
        view_src: Option<Tensor>,
    ) -> Result<Tensor> {
        if shape.iter().any(|&dim| dim == 0) {
            return Err(Error::new(ErrorKind::Msg("shape cannot contain zero dimensions".into()))
                .context("in new_tensor_impl"));
        }

        if !matches!(dtype, DataType::F32 | DataType::I32) {
            return Err(Error::new(ErrorKind::UnsupportedDataTypeForOp {
                dtype,
                op: "tensor creation",
            }));
        }

        let mut tensor_inner = self.tensor_pool.get();

        tensor_inner.dtype = dtype;
        tensor_inner.layout.shape = shape.clone();
        tensor_inner.id = TensorId::new();

        if let Some(src) = view_src {
            if !self.contain_tensor_impl(src.get_tensor_id()) {
                return Err(Error::msg("view source tensor not found in context")
                    .context("in new_tensor_impl"));
            }

            tensor_inner.view_offset += src.borrow().view_offset;

            if let Some(src_view) = src.borrow().view_tensor.clone() {
                tensor_inner.self_storage = src_view.borrow().self_storage.clone();
                tensor_inner.view_tensor = Some(src_view);
            } else {
                tensor_inner.self_storage = src.borrow().self_storage.clone();
                tensor_inner.view_tensor = Some(src.clone());
            }
        }

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

        for i in 2..MAX_DIMS {
            let next_stride = tensor_inner.layout.stride[i - 1]
                .checked_mul(tensor_inner.layout.shape[i - 1])
                .ok_or_else(|| {
                    Error::msg("stride calculation overflow").context("in stride calculation")
                })?;
            tensor_inner.layout.stride[i] = next_stride;
        }

        let tensor = Tensor(Arc::new(RefCell::new(tensor_inner)));
        self.tensor_tables.insert(tensor.get_tensor_id(), tensor.clone());

        Ok(tensor)
    }
}

// ─── Context (public) ──────────────────────────────────────────────

#[derive(Clone)]
pub struct Context(Rc<RefCell<ContextInner>>);

// ── Deprecated internal-context helpers (kept for existing callers) ──

impl Context {
    /// --- L1: auto-select the best backend (GPU > CPU) ---
    pub fn auto() -> Result<Self> {
        ContextBuilder::default().build()
    }

    /// --- L2: select a named backend ---
    pub fn with_backend(name: &str) -> Result<Self> {
        ContextBuilder::default().backend(name).build()
    }

    /// Start a fluent builder.
    pub fn builder() -> ContextBuilder {
        ContextBuilder::default()
    }

    /// Access the inner backend (panics if not set).
    pub fn backend(&self) -> Ref<'_, Box<dyn Backend>> {
        Ref::map(self.0.borrow(), |inner| inner.backend.as_ref().unwrap())
    }

    /// Access the inner buffer (panics if not set).
    pub fn buffer(&self) -> Ref<'_, Box<dyn BackendBuffer>> {
        Ref::map(self.0.borrow(), |inner| inner.buffer.as_ref().unwrap())
    }

    // ── Tensor management ──────────────────────────────────────

    pub fn new_tensor(&self, dtype: DataType, shape: &Shape) -> Result<Tensor> {
        self.0.borrow_mut().new_tensor_impl(dtype, shape, None)
    }

    pub fn new_tensor_view(&self, view_src: Tensor) -> Result<Tensor> {
        let dtype = view_src.get_dtype();
        let shape = view_src.get_shape();
        let stride = view_src.borrow().layout.stride;

        let tensor = self.0.borrow_mut().new_tensor_impl(dtype, &shape, Some(view_src.clone()))?;
        tensor.borrow_mut().layout.stride = stride;

        Ok(tensor)
    }

    pub fn dup_tensor(&self, src: &Tensor) -> Result<Tensor> {
        self.new_tensor(src.get_dtype(), &src.get_shape())
    }

    pub fn contain_tensor(&self, tensor_id: TensorId) -> bool {
        self.0.borrow().contain_tensor_impl(tensor_id)
    }

    pub fn get_tensor(&self, tensor_id: TensorId) -> Result<Tensor> {
        self.0.borrow().tensor_tables.get(&tensor_id).cloned().ok_or_else(|| {
            Error::msg(format!("tensor {} not found", tensor_id.as_usize()))
                .context("in Context::get_tensor")
        })
    }

    // ── Graph ──────────────────────────────────────────────────

    pub fn new_graph(&self) -> Result<ComputeGraph> {
        todo!("new_graph")
    }
}

// ── internal deref (legacy, keep for existing tests) ────────────────

impl std::ops::Deref for Context {
    type Target = RefCell<ContextInner>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// ── tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_type::DataType;
    use crate::shape;
    use crate::shape::Shape;

    #[test]
    fn test_context_empty() {
        let ctx = Context::with_config(ContextConfig::default());
        assert_eq!(ctx.borrow().tensor_tables.len(), 0);
        assert_eq!(ctx.borrow().graph_tables.len(), 0);
    }

    #[test]
    fn test_new_tensor_success() {
        let ctx = Context::with_config(ContextConfig {
            tensor_pool_capacity: 10,
            graph_pool_cacacity: 0,
        });
        let shape = shape![2, 3, 4, 5];
        let tensor = ctx.new_tensor(DataType::F32, &shape).unwrap();
        assert_eq!(tensor.borrow().dtype, DataType::F32);
        assert_eq!(tensor.borrow().layout.shape, shape);
        assert!(ctx.contain_tensor(tensor.get_tensor_id()));
    }

    #[test]
    fn test_new_tensor_zero_dimension() {
        let ctx = Context::with_config(ContextConfig {
            tensor_pool_capacity: 10,
            graph_pool_cacacity: 0,
        });
        let shape = shape![0, 3, 4, 5];
        let result = ctx.new_tensor(DataType::F32, &shape);
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("zero dimensions"));
    }

    #[test]
    fn test_new_tensor_unsupported_dtype() {
        let ctx = Context::with_config(ContextConfig {
            tensor_pool_capacity: 10,
            graph_pool_cacacity: 0,
        });
        let shape = shape![2, 3, 4, 5];
        let result = ctx.new_tensor(DataType::U8, &shape);
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("unsupported dtype"));
    }

    #[test]
    fn test_new_tensor_view_success() {
        let ctx = Context::with_config(ContextConfig {
            tensor_pool_capacity: 10,
            graph_pool_cacacity: 0,
        });
        let shape = shape![2, 3, 4, 5];
        let src_tensor = ctx.new_tensor(DataType::I32, &shape).unwrap();
        assert!(ctx.contain_tensor(src_tensor.get_tensor_id()));
        let view_tensor = ctx.new_tensor_view(src_tensor.clone()).unwrap();
        assert_eq!(view_tensor.borrow().dtype, DataType::I32);
        assert_eq!(view_tensor.borrow().layout.shape, shape);
        for i in 0..MAX_DIMS {
            assert_eq!(view_tensor.borrow().layout.stride[i], src_tensor.borrow().layout.stride[i]);
        }
        assert!(ctx.contain_tensor(view_tensor.get_tensor_id()));
    }

    #[test]
    fn test_new_tensor_view_invalid_source() {
        let ctx = Context::with_config(ContextConfig {
            tensor_pool_capacity: 10,
            graph_pool_cacacity: 0,
        });
        let other_ctx = Context::with_config(ContextConfig {
            tensor_pool_capacity: 10,
            graph_pool_cacacity: 0,
        });
        let shape = shape![2, 3, 4, 5];
        let src_tensor = other_ctx.new_tensor(DataType::F32, &shape).unwrap();
        let result = ctx.new_tensor_view(src_tensor);
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("view source tensor not found"));
    }
}
