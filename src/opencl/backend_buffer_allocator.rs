use super::backend_buffer::OpenclBackendBuffer;
use super::backend_context::OpenclBackendContext;
use crate::backend::BackendBuffer;
use crate::backend::BackendBufferAllocator;
use crate::error::{Error, Result};
use crate::tensor::Tensor;
use std::any::Any;

use std::cell::RefCell;
use std::rc::Rc;
pub(crate) struct OpenclBackendBufferAllocator {
    pub(super) backend_ctx: Option<Rc<RefCell<OpenclBackendContext>>>,
}
impl BackendBufferAllocator for OpenclBackendBufferAllocator {
    fn allocate_buffer(&self, size: usize) -> Result<Box<dyn BackendBuffer>> {
        ocl::Buffer::<u8>::builder()
            .queue(self.backend_ctx.as_ref().unwrap().borrow().queue.clone())
            .len(size)
            .build()
            .map(|buffer| {
                Box::new(OpenclBackendBuffer::new(
                    self.backend_ctx.as_ref().unwrap().clone(),
                    buffer,
                    size,
                )) as Box<dyn BackendBuffer>
            })
            .map_err(|e| Error::msg(format!("Failed to allocate OpenCL buffer: {}", e)))
    }

    fn alignment(&self) -> Result<usize> {
        Ok(self.backend_ctx.as_ref().unwrap().borrow().alignment)
    }

    fn max_size(&self) -> Result<usize> {
        Ok(self.backend_ctx.as_ref().unwrap().borrow().max_alloc_size)
    }

    fn alloc_size(&self, tensor: Tensor) -> Result<usize> {
        Err(Error::msg(format!("opencl: alloc_size is not implemented yet"))
            .context("in OpenclBackendBufferAllocator::alloc_size"))
    }

    fn is_host(&self) -> Result<bool> {
        Err(Error::msg(format!("opencl: is_host is not implemented yet"))
            .context("in OpenclBackendBufferAllocator::is_host"))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl OpenclBackendBufferAllocator {
    pub(super) fn new(backend_ctx: Rc<RefCell<OpenclBackendContext>>) -> Self {
        OpenclBackendBufferAllocator { backend_ctx: Some(backend_ctx) }
    }
}
