use crate::tensor::Tensor;
use crate::compute_graph::ComputeGraph;

// BackendBuffer = ggml_backend_buffer_type + ggml_backend_buffer
pub trait BackendBuffer: Send + Sync {
    type Device: BackendDevice;

    fn size(&self) -> usize;
    fn as_ptr(&self) -> *mut u8;

    fn memset(&self, value: u8);

    fn copy_from(&self, src: &dyn BackendBuffer<Device = Self::Device>, size: usize);

    fn device(&self) -> &Self::Device;
}

pub trait BackendBufferAllocator {
    type Buffer: BackendBuffer;

    fn allocate(&self, size: usize) -> Self::Buffer;

    fn alignment(&self) -> usize {
        std::mem::align_of::<usize>()
    }

    fn max_size(&self) -> usize {
        usize::MAX
    }
}

pub trait Backend {
    type Device: BackendDevice;

    fn device(&self) -> &Self::Device;

    fn synchronize(&self);

    fn compute(&self, graph: &mut ComputeGraph);

    fn compute_async(&self, graph: &mut ComputeGraph);

    fn memcpy_async(&self, dst: *mut u8, src: *const u8, size: usize);

    fn copy_tensor(&self, src: Tensor, dst: Tensor);
}
pub trait BackendDevice: Send + Sync {
    fn name(&self) -> &str;

    fn memory(&self) -> (usize, usize);

    fn supports_op(&self, op: Tensor) -> bool;

    fn create_backend(&self) -> Box<dyn Backend<Device = Self>>;
}
