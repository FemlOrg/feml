use crate::backend::{
    Backend, BackendBufferAllocator, BackendDevice, BackendDeviceCaps, BackendDeviceProps,
    BackendDeviceType,
};
use crate::compute_graph::ComputeGraph;
use crate::defs::Status;
use crate::tensor::Tensor;
use cudarc::driver::result::event;
use ocl::ffi::{clEnqueueBarrier, clEnqueueBarrierWithWaitList};
#[cfg(feature = "opencl")]
use ocl::{ocl_core, Context, Device, Event, Platform, Queue};

struct OpenclBackendContext {
    queue: ocl::Queue,
}

pub struct OpenclBackend {
    device: Vec<OpenclDevice>,
    context: OpenclBackendContext,
}

pub struct OpenclDevice;

impl OpenclBackend {
    pub fn new() -> Self {
        Self { device: Vec::new(), context: OpenclBackendContext }
    }
}

impl Default for OpenclBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for OpenclBackend {
    type Device = OpenclDevice;

    fn get_name(&self) -> &str {
        "opencl"
    }

    fn synchronize(&self) {
        let mut event = self.context.queue.enqueue_marker(None)?;
        event.wait_for();
    }

    fn graph_compute(&self, _graph: &mut ComputeGraph) -> Status {
        Status::Aborted
    }

    fn memcpy_async(&self, _dst: *mut u8, _src: *const u8, _size: usize) {}

    fn init_tensor(&self, _tensor: Tensor) {}

    fn memset_tensor(&self, _tensor: Tensor, _value: u8, _offset: usize, _size: usize) {}

    fn set_tensor(&self, _tensor: Tensor, _data: *mut u8, _offset: usize, _size: usize) {}

    fn get_tensor(&self, _tensor: Tensor, _data: *mut u8, _offset: usize, _size: usize) {}

    fn copy_tensor(&self, _src: Tensor, _dst: Tensor) {}

    fn set_tensor_async(&self, _tensor: Tensor, _data: *mut u8, _offset: usize, _size: usize) {}

    fn get_tensor_async(&self, _tensor: Tensor, _data: *mut u8, _offset: usize, _size: usize) {}

    fn copy_tensor_async(&self, _src: Tensor, _dst: Tensor) {}
}

impl BackendDevice for OpenclDevice {
    fn name(&self) -> &str {
        "opencl"
    }

    fn memory(&self) -> (usize, usize) {
        (0, 0)
    }

    fn description(&self) -> &str {
        "OpenCL device"
    }

    fn device_type(&self) -> BackendDeviceType {
        BackendDeviceType::Gpu
    }

    fn props(&self) -> BackendDeviceProps {
        BackendDeviceProps {
            name: "opencl",
            description: "OpenCL device",
            memory_free: 0,
            memory_total: 0,
            device_type: BackendDeviceType::Gpu,
            caps: BackendDeviceCaps {
                aysnc: false,
                host_buffer: false,
                buffer_from_host_ptr: false,
                events: false,
            },
        }
    }

    fn init(&self, _params: *mut u8) {}

    fn supports_op(&self, _tensor: Tensor) -> bool {
        false
    }

    fn supports_buffer_allocator(
        &self,
        _buffer_allocator: &Box<dyn BackendBufferAllocator>,
    ) -> bool {
        false
    }

    fn offload_op(&self, _tensor: Tensor) -> bool {
        false
    }
}
