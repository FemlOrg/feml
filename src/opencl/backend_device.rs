use super::backend::OpenclBackend;
use super::backend_buffer_allocator::OpenclBackendBufferAllocator;
use super::backend_context::OpenclBackendContext;
use super::backend_context::OpenclGpuFamlily;
use crate::backend::{
    Backend, BackendBuffer, BackendBufferAllocator, BackendDevice, BackendDeviceCaps,
    BackendDeviceProps, BackendDeviceType,
};
use crate::data_type::TensorOpType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;
use ocl::ocl_core::OpenclVersion;
use ocl::{Context, Device, Platform};
use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;
use tracing::info;
#[derive(Clone)]
pub struct OpenclBackendDevice {
    #[allow(dead_code)]
    pub(super) platform: Platform,
    #[allow(dead_code)]
    pub(super) platform_name: String,
    pub(super) device: Device,
    pub(super) device_name: String,
    #[allow(dead_code)]
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
        self.device_name.as_str()
    }

    fn device_type(&self) -> BackendDeviceType {
        BackendDeviceType::Gpu
    }

    fn props(&self) -> BackendDeviceProps {
        BackendDeviceProps {
            name: "opencl",
            description: self.device_name.clone(),
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

    fn init_backend(&self) -> Result<Box<dyn Backend>> {
        Ok(Box::new(OpenclBackend { context: self.backend_ctx.as_ref().unwrap().clone() }))
    }

    fn supports_op(&self, tensor: Tensor) -> Result<bool> {
        match tensor.get_op_type() {
            TensorOpType::TensorOpMul => Ok(true),
            _ => Ok(false),
        }
    }

    fn supports_buffer_allocator(
        &self,
        buffer_allocator: &dyn BackendBufferAllocator,
    ) -> Result<bool> {
        let ret = self.name() == "opencl"
            && self.as_any().is::<OpenclBackendDevice>()
            && buffer_allocator.as_any().is::<OpenclBackendBufferAllocator>();

        Ok(ret)
    }

    fn buffer_from_host_ptr(
        &self,
        _ptr: &mut [u8],
        _size: usize,
        _max_tensor_size: usize,
    ) -> Result<Box<dyn BackendBuffer>> {
        Err(Error::msg(format!("opencl: not support buffer_from_host_ptr"))
            .context("in OpenclBackendDevice::buffer_from_host_ptr"))
    }

    fn offload_op(&self, _tensor: Tensor) -> Result<bool> {
        Err(Error::msg(format!("opencl: not support offload_op trait"))
            .context("in OpenclBackendDevice::offload_op"))
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
        let ctx = Rc::new(RefCell::new(OpenclBackendContext::new(self)?));

        {
            let mut guard = ctx.borrow_mut();

            if self.device_name == "Intel" {
                guard.gpu_family = OpenclGpuFamlily::Intel;
                guard.wave_size = 64;
            } else if self.device_name == "Qualcomm" {
                guard.gpu_family = OpenclGpuFamlily::Intel;
                guard.wave_size = 64;
            } else {
                return Err(Error::msg(format!("Unsupported gpu {}", self.device_name))
                    .context("in OpenclBackendDevice::init"));
            }

            let max_alloc = ocl::core::get_device_info(
                guard.device,
                ocl::ocl_core::DeviceInfo::MaxMemAllocSize,
            )
            .map_err(|e| Error::msg(format!("Failed to get device info: {}", e)))?;
            info!("Opencl: max mem alloc size {}", max_alloc);
            let max_img_buf = ocl::core::get_device_info(
                guard.device,
                ocl::ocl_core::DeviceInfo::ImageMaxBufferSize,
            )
            .map_err(|e| Error::msg(format!("Failed to get device info: {}", e)))?;
            info!("Opencl: device max image buffer size {}", max_img_buf);
            let max_workgroup = ocl::core::get_device_info(
                guard.device,
                ocl::ocl_core::DeviceInfo::MaxWorkGroupSize,
            )
            .map_err(|e| Error::msg(format!("Failed to get device info: {}", e)))?;
            info!("Opencl: device max workgroup size: {}", max_workgroup);

            guard.load_cl_kernels()?;
            drop(guard);
        }

        self.backend_ctx = Some(ctx);

        Ok(())
    }
}
