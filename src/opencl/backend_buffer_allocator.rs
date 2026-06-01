use super::backend_buffer::OpenclBackendBuffer;
use super::backend_context::OpenclBackendContext;
use crate::backend::BackendBufferAllocator;
use std::cell::RefCell;
use std::rc::Rc;
pub(crate) struct OpenclBackendBufferAllocator {
    backend_ctx: Option<Rc<RefCell<OpenclBackendContext>>>,
}
impl BackendBufferAllocator for OpenclBackendBufferAllocator {
    fn allocate_buffer(&self, size: usize) -> Result<Box<dyn BackendBuffer>> {
        ocl::Buffer::<u8>::builder()
            .queue(self.backend_ctx.as_ref().unwrap().queue().clone())
            .len(size)
            .build()
            .map(|buffer| {
                Box::new(OpenclBackendBuffer::new(
                    self.backend_ctx.as_ref().unwrap().clone(),
                    buffer,
                    size,
                )) as Box<dyn BackendBuffer>
            })
            .map_err(|e| format!("Failed to allocate OpenCL buffer: {}", e).into())
    }

    fn alignment(&self) -> usize {
        std::mem::align_of::<usize>()
    }

    fn max_size(&self) -> usize {
        usize::MAX
    }

    fn alloc_size(&self, tensor: Tensor) -> Result<usize>;
}
