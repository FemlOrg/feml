//! CUDA device discovery and registrar.
//!
//! Devices are probed structurally (name, memory, compute capability) — no
//! name matching, no whitelist. Scoring: available CUDA devices outrank CPU
//! backends. A missing CUDA driver/device yields an empty registrar (score 0),
//! mirroring how a machine without a GPU reports "unavailable".

use std::sync::Arc;

use cudarc::driver::{CudaContext, DriverError};

use feml_core::backend::{
    Backend, BackendDevice, BackendRegistrar, BufferHandle, Capabilities, DeviceInfo, DeviceType,
};
use feml_core::dtype::DType;
use feml_core::error::{Error, Result};
use feml_core::op::Op;

use crate::backend::CudaBackend;

pub(crate) fn cuda_err(e: DriverError) -> Error {
    Error::backend("CUDA", e.to_string())
}

#[derive(Clone)]
pub struct CudaBackendDevice {
    ctx: Arc<CudaContext>,
    info: DeviceInfo,
}

impl CudaBackendDevice {
    fn probe(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal).map_err(cuda_err)?;
        let name = ctx.name().map_err(cuda_err)?;
        let (memory_free, memory_total) = ctx.mem_get_info().map_err(cuda_err)?;
        let info = DeviceInfo {
            name,
            description: format!("CUDA device {ordinal}"),
            memory_total,
            memory_free,
            device_type: DeviceType::Gpu,
            caps: Capabilities { async_compute: true, host_buffer: false, events: true },
        };
        Ok(Self { ctx, info })
    }
}

impl BackendDevice for CudaBackendDevice {
    fn info(&self) -> Result<DeviceInfo> {
        Ok(self.info.clone())
    }

    fn init_backend(&self) -> Result<Box<dyn Backend>> {
        Ok(Box::new(CudaBackend::new(self.ctx.clone())?))
    }

    fn supports_op(&self, op: Op, src_dtypes: &[DType]) -> bool {
        matches!(op, Op::Mul | Op::Add | Op::MulMat) && src_dtypes.iter().all(|d| *d == DType::F32)
    }

    fn offload_op(&self, op: Op) -> bool {
        matches!(op, Op::Mul | Op::Add | Op::MulMat)
    }

    fn buffer_from_host_ptr(
        &self,
        _ptr: &mut [u8],
        _size: usize,
        _max_tensor_size: usize,
    ) -> Result<BufferHandle> {
        Err(Error::msg("CUDA: buffer_from_host_ptr not implemented yet"))
    }
}

pub struct CudaRegistrar {
    devices: Vec<CudaBackendDevice>,
}

impl CudaRegistrar {
    /// Probe all CUDA devices on this machine. Missing driver/GPUs produce an
    /// empty registrar (score 0), never an error.
    pub fn probe() -> Result<Self> {
        let count = match CudaContext::device_count() {
            Ok(c) if c > 0 => c as usize,
            _ => 0,
        };
        let mut devices = Vec::new();
        for ordinal in 0..count {
            if let Ok(d) = CudaBackendDevice::probe(ordinal) {
                devices.push(d);
            }
        }
        Ok(Self { devices })
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
    }
}

impl BackendRegistrar for CudaRegistrar {
    fn name(&self) -> &str {
        "CUDA"
    }

    fn device_count(&self) -> usize {
        self.devices.len()
    }

    fn device(&self, index: usize) -> Result<Box<dyn BackendDevice>> {
        self.devices
            .get(index)
            .cloned()
            .map(|d| Box::new(d) as Box<dyn BackendDevice>)
            .ok_or_else(|| Error::msg(format!("CUDA: device {index} not found")))
    }

    fn score(&self) -> u32 {
        if self.devices.is_empty() { 0 } else { 1000 }
    }
}
