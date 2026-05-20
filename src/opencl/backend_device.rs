use ocl::ocl_core::OpenclVersion;
use ocl::{Context, Device, Platform};

#[derive(Clone)]
pub struct OpenclBackendDevice {
    pub(super) platform: ocl::Platform,
    pub(super) platform_name: String,
    pub(super) device: ocl::Device,
    pub(super) device_name: String,
    pub(super) device_version: ocl_core::OpenclVersion,
    pub(super) context: ocl::Context,
}

impl BackendDevice for OpenclBackendDevice {
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

    fn init_backend(&self, _params: *mut u8) -> Result<()> {
        Ok(())
    }

    fn supports_op(&self, _tensor: Tensor) -> bool {
        false
    }

    fn supports_buffer_allocator(&self, _buffer_allocator: &dyn BackendBufferAllocator) -> bool {
        false
    }

    fn offload_op(&self, _tensor: Tensor) -> bool {
        false
    }
}
