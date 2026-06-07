use crate::backend::BackendBufferAllocator;

pub(super) struct CudaBackendBufferAllocator;

impl BackendBufferAllocator for CudaBackendBufferAllocator {
    fn allocate_buffer(
        &self,
        size: usize,
    ) -> crate::error::Result<Box<dyn crate::backend::BackendBuffer>> {
        todo!()
    }

    fn alloc_size(&self, tensor: crate::tensor::Tensor) -> crate::error::Result<usize> {
        todo!()
    }

    fn is_host(&self) -> crate::error::Result<bool> {
        todo!()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        todo!()
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        todo!()
    }
}
