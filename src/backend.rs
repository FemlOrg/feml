use crate::compute_graph::ComputeGraph;
use crate::defs::Status;
use crate::tensor::Tensor;

pub enum BackendDeviceType {
    Cpu,
    Gpu,
    ACCEL,
}

struct BackendDeviceCaps {
    aysnc: bool,
    host_buffer: bool,
    buffer_from_host_ptr: bool,
    events: bool,
}

struct BackendDeviceProps {
    name: &'static str,
    description: &'static str,
    memory_free: usize,
    memory_total: usize,
    device_type: BackendDeviceType,
    caps: BackendDeviceCaps,
}

// BackendBuffer = ggml_backend_buffer_type + ggml_backend_buffer
pub trait BackendBuffer: Send + Sync {
    fn as_ptr(&self) -> *mut u8;

    fn device(&self) -> Box<dyn BackendDevice>;

    fn get_base(&self) -> *mut u8;

    fn clear(&self, value: u8);

    fn reset(&self);
}

pub trait BackendBufferAllocator {
    fn allocate_buffer(&self, size: usize) -> Box<dyn BackendBuffer>;

    fn alignment(&self) -> usize {
        std::mem::align_of::<usize>()
    }

    fn max_size(&self) -> usize {
        usize::MAX
    }

    fn alloc_size(&self, tensor: Tensor) -> usize;
}

pub trait Backend {
    type Device: BackendDevice;

    fn get_name(&self) -> &str;

    fn device(&self) -> &Self::Device;

    fn synchronize(&self);

    fn graph_compute(&self, graph: &mut ComputeGraph) -> Status;

    fn memcpy_async(&self, dst: *mut u8, src: *const u8, size: usize);

    fn init_tensor(&self, tensor: Tensor);

    fn memset_tensor(&self, tensor: Tensor, value: u8, offset: usize, size: usize);

    fn set_tensor(&self, tensor: Tensor, data: *mut u8, offset: usize, size: usize);

    fn get_tensor(&self, tensor: Tensor, data: *mut u8, offset: usize, size: usize);

    fn copy_tensor(&self, src: Tensor, dst: Tensor);

    fn set_tensor_async(&self, tensor: Tensor, data: *mut u8, offset: usize, size: usize);

    fn get_tensor_async(&self, tensor: Tensor, data: *mut u8, offset: usize, size: usize);

    fn copy_tensor_async(&self, src: Tensor, dst: Tensor);
}
pub trait BackendDevice: Send + Sync {
    fn name(&self) -> &str;

    fn memory(&self) -> (usize, usize);

    fn description(&self) -> &str;

    fn device_type(&self) -> BackendDeviceType;

    fn props(&self) -> BackendDeviceProps;

    fn init(&self, params: *mut u8);

    fn supports_op(&self, tensor: Tensor) -> bool;

    fn supports_buffer_allocator(&self, buffer_allocator: &Box<dyn BackendBufferAllocator>) -> bool;

    fn offload_op(&self, tensor: Tensor) -> bool;
}

pub trait BackendRegister: Send + Sync {
    fn name(&self) -> &str;

    fn device_count(&self) -> usize;

    fn device(&self) -> Box<dyn BackendDevice>;
}
