use super::backend_context::OpenclBackendContext;
use super::backend_context::OpenclGpuFamlily;
use crate::backend::BackendDevice;
use ocl::core::CommandQueue;
use ocl::core::DeviceInfoResult;
use ocl::ocl_core::OpenclVersion;
use ocl::{CommandQueueProperties, Context, Device, Platform};
use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;
use tracing::info;
#[derive(Clone)]
pub struct OpenclBackendDevice {
    pub(super) platform: Platform,
    pub(super) platform_name: String,
    pub(super) device: Device,
    pub(super) device_name: String,
    pub(super) device_version: OpenclVersion,
    pub(super) context: Context,
    pub(super) backend_ctx: Option<Rc<RefCell<OpenclBackendContext>>>,
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

    fn init_backend(&self, _params: &[u8]) -> Result<()> {
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

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl OpenclBackendDevice {
    pub(super) fn init(&mut self) -> Result<()> {
        if self.backend_ctx.is_some() {
            return Ok(());
        }
        self.backend_ctx = Some(Rc::new(RefCell::new(OpenclBackendContext::new(self))));

        let mut ctx = &self.backend_ctx.unwrap();

        if self.device_name == "Intel" {
            ctx.gpu_family = OpenclGpuFamlily::Intel;
            ctx.wave_size = 64;
        } else if self.device_name == "Qualcomm" {
            ctx.gpu_family = OpenclGpuFamlily::Intel;
            ctx.wave_size = 64;
        } else {
            return Err(Error::msg(format!("Unsupported gpu {}", self.device_name))
                .context("in OpenclBackendDevice::init"));
        }

        let mut info = ocl::ocl_core::DeviceInfo::MaxMemAllocSize;
        info!(
            "Opencl: max mem alloc size {}",
            to_stringocl::core::get_device_info(
                ctx.device,
                ocl::ocl_core::DeviceInfo::MaxMemAllocSize
            )
        );
        info!(
            "Opencl: device max image buffer size {}",
            ocl::core::get_device_info(ctx.device, ocl::ocl_core::DeviceInfo::ImageMaxBufferSize);
        );
        info!(
            "Opencl: device max workgropu size: {}",
            ocl::core::get_device_info(ctx.device, ocl::ocl_core::DeviceInfo::MaxWorkGroupSize)
        );

        // load cl kernels
        ctx.load_cl_kernels();

        // TODO: add more device info

        Ok(())
    }
}
