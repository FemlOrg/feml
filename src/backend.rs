use crate::compute_graph::ComputeGraph;
use crate::error::Result;
use crate::tensor::Tensor;

pub enum BackendDeviceType {
    Cpu,
    Gpu,
    ACCEL,
}

pub struct BackendDeviceCaps {
    pub aysnc: bool,
    pub host_buffer: bool,
    pub buffer_from_host_ptr: bool,
    pub events: bool,
}

pub struct BackendDeviceProps {
    pub name: &'static str,
    pub description: &'static str,
    pub memory_free: usize,
    pub memory_total: usize,
    pub device_type: BackendDeviceType,
    pub caps: BackendDeviceCaps,
}

// BackendBuffer = ggml_backend_buffer_type + ggml_backend_buffer
pub trait BackendBuffer: Send + Sync {
    fn as_ptr(&self) -> *mut u8;

    fn device(&self) -> Box<dyn BackendDevice>;

    fn get_base(&self) -> *mut u8;

    fn clear(&self, value: u8) -> Result<()>;

    fn reset(&self) -> Result<()>;

    fn init_tensor(&self, tensor: Tensor) -> Result<()>;

    fn memset_tensor(&self, tensor: Tensor, value: u8, offset: usize, size: usize) -> Result<()>;

    fn set_tensor(&self, tensor: Tensor, data: [u8], offset: usize, size: usize) -> Result<()>;

    fn get_tensor(&self, tensor: Tensor, data: [u8], offset: usize, size: usize) -> Result<()>;

    fn copy_tensor(&self, src: Tensor, dst: Tensor) -> Result<()>;
}

pub trait BackendBufferAllocator {
    fn allocate_buffer(&self, size: usize) -> Result<Box<dyn BackendBuffer>>;

    fn alignment(&self) -> usize {
        std::mem::align_of::<usize>()
    }

    fn max_size(&self) -> usize {
        usize::MAX
    }

    fn alloc_size(&self, tensor: Tensor) -> Result<usize>;
}

pub trait Backend {
    type Device: BackendDevice;

    fn name(&self) -> &str;

    fn synchronize(&self) -> Result<()>;

    fn graph_compute(&self, graph: &mut ComputeGraph) -> Result<()>;

    fn memcpy_async(&self, dst: [u8], src: [u8], size: usize) -> Result<()>;

    fn set_tensor_async(
        &self,
        tensor: Tensor,
        data: *mut u8,
        offset: usize,
        size: usize,
    ) -> Result<()>;

    fn get_tensor_async(
        &self,
        tensor: Tensor,
        data: *mut u8,
        offset: usize,
        size: usize,
    ) -> Result<()>;

    fn copy_tensor_async(&self, src: Tensor, dst: Tensor) -> Result<()>;
}

pub trait BackendDevice: Send + Sync {
    fn name(&self) -> &str;

    fn memory(&self) -> (usize, usize);

    fn description(&self) -> &str;

    fn device_type(&self) -> BackendDeviceType;

    fn props(&self) -> BackendDeviceProps;

    fn init(&self, params: [u8]) -> Result<()>;

    fn supports_op(&self, tensor: Tensor) -> bool;

    fn supports_buffer_allocator(&self, buffer_allocator: &dyn BackendBufferAllocator) -> bool;

    fn offload_op(&self, tensor: Tensor) -> bool;
}

pub trait BackendRegister: Send + Sync {
    fn name(&self) -> &str;

    fn device_count(&self) -> usize;

    fn device(&self, index: usize) -> Result<Box<dyn BackendDevice>>;
}
