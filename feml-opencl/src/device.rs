//! OpenCL device discovery and registrar.
//!
//! Devices are probed structurally (type, name, memory) — no name matching,
//! no whitelist. Scoring: available OpenCL devices outrank CPU backends.

use feml_core::backend::{
    Backend, BackendDevice, BackendRegistrar, BufferHandle, DeviceInfo, DeviceType,
};
use feml_core::dtype::DType;
use feml_core::error::{Error, Result};
use feml_core::op::Op;

use crate::backend::OpenclBackend;
use ocl::core::get_device_info;

fn ocl_err(e: ocl::Error) -> feml_core::Error {
    feml_core::Error::backend("OpenCL", e.to_string())
}

fn core_err(e: ocl::core::Error) -> feml_core::Error {
    feml_core::Error::backend("OpenCL", e.to_string())
}

fn device_memory(device: &ocl::Device) -> Result<usize> {
    match get_device_info(device, ocl::core::DeviceInfo::GlobalMemSize).map_err(core_err)? {
        ocl::core::DeviceInfoResult::GlobalMemSize(v) => Ok(v as usize),
        _ => Err(feml_core::Error::msg("OpenCL: unexpected memory info result")),
    }
}

fn device_type(device: &ocl::Device) -> Result<DeviceType> {
    match get_device_info(device, ocl::core::DeviceInfo::Type).map_err(core_err)? {
        ocl::core::DeviceInfoResult::Type(t) => {
            if t.contains(ocl::core::DeviceType::GPU) {
                Ok(DeviceType::Gpu)
            } else if t.contains(ocl::core::DeviceType::CPU) {
                Ok(DeviceType::Cpu)
            } else {
                Ok(DeviceType::Accelerator)
            }
        }
        _ => Err(feml_core::Error::msg("OpenCL: unexpected type info result")),
    }
}

#[derive(Clone)]
pub struct OpenclBackendDevice {
    platform: ocl::Platform,
    device: ocl::Device,
    info: DeviceInfo,
}

impl OpenclBackendDevice {
    fn probe(platform: ocl::Platform, device: ocl::Device) -> Result<Self> {
        let name = device.name().map_err(ocl_err)?;
        let vendor = device.vendor().map_err(ocl_err)?;
        let memory_total = device_memory(&device)?;
        let device_type = device_type(&device)?;
        let info = DeviceInfo {
            name,
            description: vendor,
            memory_total,
            memory_free: memory_total,
            device_type,
            caps: Default::default(),
        };
        Ok(Self { platform, device, info })
    }
}

impl BackendDevice for OpenclBackendDevice {
    fn info(&self) -> Result<DeviceInfo> {
        Ok(self.info.clone())
    }

    fn init_backend(&self) -> Result<Box<dyn Backend>> {
        Ok(Box::new(OpenclBackend::new(self.device, self.platform)?))
    }

    fn supports_op(&self, op: Op, src_dtypes: &[DType]) -> bool {
        matches!(op, Op::Mul) && src_dtypes.iter().all(|d| *d == DType::F32)
    }

    fn offload_op(&self, op: Op) -> bool {
        matches!(op, Op::Mul | Op::MulMat)
    }

    fn buffer_from_host_ptr(
        &self,
        _ptr: &mut [u8],
        _size: usize,
        _max_tensor_size: usize,
    ) -> Result<BufferHandle> {
        Err(Error::msg("OpenCL: buffer_from_host_ptr not implemented yet"))
    }
}

pub struct OpenclRegistrar {
    devices: Vec<OpenclBackendDevice>,
}

impl OpenclRegistrar {
    /// Probe all OpenCL platforms/devices on this machine.
    pub fn probe() -> Result<Self> {
        let mut devices = Vec::new();
        for platform in ocl::Platform::list() {
            for device in ocl::Device::list_all(platform)
                .map_err(|e| Error::backend("OpenCL", format!("device list: {e}")))?
            {
                if let Ok(d) = OpenclBackendDevice::probe(platform, device) {
                    devices.push(d)
                }
            }
        }
        Ok(Self { devices })
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
    }
}

impl BackendRegistrar for OpenclRegistrar {
    fn name(&self) -> &str {
        "OpenCL"
    }

    fn device_count(&self) -> usize {
        self.devices.len()
    }

    fn device(&self, index: usize) -> Result<Box<dyn BackendDevice>> {
        self.devices
            .get(index)
            .cloned()
            .map(|d| Box::new(d) as Box<dyn BackendDevice>)
            .ok_or_else(|| Error::msg(format!("OpenCL: device {index} not found")))
    }

    fn score(&self) -> u32 {
        if self.devices.is_empty() { 0 } else { 1000 }
    }
}
