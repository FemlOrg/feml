use super::backend::OpenclBackend;
use super::backend_context::OpenclBackendContext;
use super::backend_context::OpenclGpuFamlily;
use crate::backend::DeviceInfo;
#[allow(dead_code)]
use crate::backend::{
    Backend, BackendBuffer, BackendDevice, BackendDeviceCaps, BackendDeviceProps, BackendDeviceType,
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
    pub(super) context: Rc<Context>,
    pub(super) backend_ctx: Option<Rc<RefCell<OpenclBackendContext>>>,
}

impl BackendDevice for OpenclBackendDevice {
    fn info(&self) -> Result<DeviceInfo> {
        todo!()
    }

    fn init_backend(&self) -> Result<Box<dyn Backend>> {
        let ctx = self.backend_ctx.as_ref()
            .ok_or_else(|| Error::msg("device not initialized, call init_devices() first"))?;
        Ok(Box::new(OpenclBackend { backend_ctx: ctx.clone() }))
    }

    fn supports_op(&self, op_type: TensorOpType) -> Result<bool> {
        match op_type {
            TensorOpType::TensorOpMul => Ok(true),
            _ => Ok(false),
        }
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
        let ocl_ctx = OpenclBackendContext::new(self).expect("create opencl context failed!");
        let ctx = Rc::new(RefCell::new(ocl_ctx));

        let mut guard = ctx.borrow_mut();
        println!("{}", self.device_name);
        if self.device_name == "Intel" {
            guard.gpu_family = OpenclGpuFamlily::Intel;
            guard.wave_size = 64;
        } else if self.device_name == "Qualcomm" {
            guard.gpu_family = OpenclGpuFamlily::Intel;
            guard.wave_size = 64;
        } else if self.device_name == "Glenfly Arise-GT10C0t" {
            guard.gpu_family = OpenclGpuFamlily::Glenfly;
            guard.wave_size = 32;
        } else {
            return Err(Error::msg(format!("Unsupported gpu {}", self.device_name))
                .context("in OpenclBackendDevice::init"));
        }

        let max_alloc =
            ocl::core::get_device_info(guard.device, ocl::ocl_core::DeviceInfo::MaxMemAllocSize)
                .map_err(|e| Error::msg(format!("Failed to get device info: {}", e)))?;
        info!("Opencl: max mem alloc size {}", max_alloc);
        let max_img_buf =
            ocl::core::get_device_info(guard.device, ocl::ocl_core::DeviceInfo::ImageMaxBufferSize)
                .map_err(|e| Error::msg(format!("Failed to get device info: {}", e)))?;
        info!("Opencl: device max image buffer size {}", max_img_buf);
        let max_workgroup =
            ocl::core::get_device_info(guard.device, ocl::ocl_core::DeviceInfo::MaxWorkGroupSize)
                .map_err(|e| Error::msg(format!("Failed to get device info: {}", e)))?;
        info!("Opencl: device max workgroup size: {}", max_workgroup);

        guard.load_cl_kernels()?;
        drop(guard);

        self.backend_ctx = Some(ctx);

        Ok(())
    }
}
