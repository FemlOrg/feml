//! CPU device discovery and registrar.
//!
//! A CPU device is always present; its score (100) is below GPU backends
//! (OpenCL: 1000) so `Registry::open_best` prefers accelerators when available.

use feml_core::backend::{
    Backend, BackendDevice, BackendRegistrar, BufferHandle, DeviceInfo, DeviceType,
};
use feml_core::dtype::DType;
use feml_core::error::{Error, Result};
use feml_core::op::Op;

use crate::backend::CpuBackend;

#[derive(Clone)]
pub struct CpuBackendDevice {
    info: DeviceInfo,
}

impl CpuBackendDevice {
    pub fn new() -> Self {
        Self {
            info: DeviceInfo {
                name: "CPU".into(),
                description: "generic CPU device".into(),
                memory_total: 0,
                memory_free: 0,
                device_type: DeviceType::Cpu,
                caps: Default::default(),
            },
        }
    }
}

impl Default for CpuBackendDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl BackendDevice for CpuBackendDevice {
    fn info(&self) -> Result<DeviceInfo> {
        Ok(self.info.clone())
    }

    fn init_backend(&self) -> Result<Box<dyn Backend>> {
        Ok(Box::new(CpuBackend::new()))
    }

    fn supports_op(&self, op: Op, src_dtypes: &[DType]) -> bool {
        matches!(op, Op::Mul) && src_dtypes.iter().all(|d| *d == DType::F32)
    }

    fn offload_op(&self, _op: Op) -> bool {
        false
    }

    fn buffer_from_host_ptr(
        &self,
        _ptr: &mut [u8],
        _size: usize,
        _max_tensor_size: usize,
    ) -> Result<BufferHandle> {
        Err(Error::msg("CPU: buffer_from_host_ptr not implemented yet"))
    }
}

pub struct CpuRegistrar {
    devices: Vec<CpuBackendDevice>,
}

impl CpuRegistrar {
    pub fn probe() -> Result<Self> {
        Ok(Self { devices: vec![CpuBackendDevice::new()] })
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
    }
}

impl BackendRegistrar for CpuRegistrar {
    fn name(&self) -> &str {
        "CPU"
    }

    fn device_count(&self) -> usize {
        self.devices.len()
    }

    fn device(&self, index: usize) -> Result<Box<dyn BackendDevice>> {
        self.devices
            .get(index)
            .cloned()
            .map(|d| Box::new(d) as Box<dyn BackendDevice>)
            .ok_or_else(|| Error::msg(format!("CPU: device {index} not found")))
    }

    fn score(&self) -> u32 {
        100
    }
}
