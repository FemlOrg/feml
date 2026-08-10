use super::backend_device::CpuBackendDevice;
use crate::backend::{BackendDevice, BackendRegister};
use crate::error::{Error, ErrorKind, Result};
use std::sync::OnceLock;

static CPU_BACKEND_REG: OnceLock<CpuBackendRegister> = OnceLock::new();

pub(super) struct CpuBackendRegister {
    devices: Vec<CpuBackendDevice>,
}

impl BackendRegister for CpuBackendRegister {
    fn name(&self) -> &str {
        "CPU"
    }

    fn device_count(&self) -> usize {
        self.devices.len()
    }

    fn device(&self, index: usize) -> Result<Box<dyn BackendDevice>> {
        Ok(Box::new(self.cpu_device(index)?))
    }
}

impl CpuBackendRegister {
    pub fn init() -> &'static Self {
        CPU_BACKEND_REG.get_or_init(|| {
            Self::try_new().unwrap_or_else(|err| {
                eprintln!("cpu: failed to initialize backend register: {err}");

                Self { devices: Vec::new() }
            })
        })
    }

    pub fn cpu_device(&self, index: usize) -> Result<CpuBackendDevice> {
        self.devices.get(index).cloned().ok_or_else(|| {
            Error::new(ErrorKind::DeviceNotFound {
                backend: "cpu",
                index,
                count: self.devices.len(),
            })
        })
    }

    fn try_new() -> Result<Self> {
        Ok(Self { devices: Self::probe_devices()? })
    }

    fn probe_devices() -> Result<Vec<CpuBackendDevice>> {
        let mut cpu_devices: Vec<CpuBackendDevice> = Vec::new();
        cpu_devices.push(CpuBackendDevice::new());
        Ok(cpu_devices)
    }
}
