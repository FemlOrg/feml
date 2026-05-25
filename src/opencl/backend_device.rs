use super::backend_context::OpenclBackendContext;
use ocl::core::CommandQueue;
use ocl::ocl_core::OpenclVersion;
use ocl::{CommandQueueProperties, Context, Device, Platform};

#[derive(Clone)]
pub struct OpenclBackendDevice {
    pub(super) platform: Platform,
    pub(super) platform_name: String,
    pub(super) device: Device,
    pub(super) device_name: String,
    pub(super) device_version: OpenclVersion,
    pub(super) context: Context,
    pub(super) backend_ctx: OpenclBackendContext,
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

impl OpenclBackendDevice {
    pub(super) fn init(&self) -> Result<()> {
        self.backend_ctx = OpenclBackendContext::new(self);

        self.backend_ctx.context = Some(self.context.clone());
        self.backend_ctx.device = Some(self.device.clone());
        self.backend_ctx.device_name = Some(self.device_name.clone());

        let mut props = CommandQueueProperties::ON_DEVICE_DEFAULT;
        #[cfg(feature = "opencl-profiling")]
        {
            props.profiling();
        }

        self.backend_ctx.queue = Some(ocl::Queue::new(&self.context, self.device, Some(props))?);

        Ok(())
    }
}
