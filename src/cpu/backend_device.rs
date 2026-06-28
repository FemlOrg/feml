use super::backend::CpuBackend;
use crate::backend::{
    Backend, BackendBuffer, BackendCapabilities, BackendDevice, BackendDeviceType, DeviceInfo,
    MemoryInfo,
};
use crate::data_type::TensorOpType;
use crate::error::{Error, ErrorKind, Result};
use crate::tensor::Tensor;
use std::any::Any;

#[derive(Clone)]
pub struct CpuBackendDevice {
    pub(super) description: String,
}

impl CpuBackendDevice {
    pub fn new() -> Self {
        Self { description: "CPU device".to_string() }
    }
}

impl BackendDevice for CpuBackendDevice {
    fn info(&self) -> Result<DeviceInfo> {
        Ok(DeviceInfo {
            name: "cpu".to_string(),
            description: self.description.clone(),
            memory: MemoryInfo { total: 0, free: 0 },
            device_type: BackendDeviceType::Cpu,
            caps: BackendCapabilities { async_compute: false, host_buffer: false, events: false },
        })
    }

    fn init_backend(&self) -> Result<Box<dyn Backend>> {
        Ok(Box::new(CpuBackend::new(self.clone())))
    }

    fn supports_op(&self, op_type: TensorOpType) -> Result<bool> {
        match op_type {
            TensorOpType::TensorOpMul => Ok(true),
            _ => Ok(false),
        }
    }

    fn offload_op(&self, _tensor: Tensor) -> Result<bool> {
        Ok(false)
    }

    fn buffer_from_host_ptr(
        &self,
        _ptr: &mut [u8],
        _size: usize,
        _max_tensor_size: usize,
    ) -> Result<Box<dyn BackendBuffer>> {
        Err(Error::new(ErrorKind::UnsupportedBackendOp {
            backend: "cpu",
            op: "buffer_from_host_ptr",
        }))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
