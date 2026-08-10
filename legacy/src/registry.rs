use crate::backend::{Backend, BackendDevice, BackendRegister};
use crate::error::{Error, Result};

#[derive(Default)]
pub struct Registry {
    registers: Vec<Box<dyn BackendRegister>>,
}

impl Registry {
    pub fn discover() -> Result<Self> {
        let mut registers: Vec<Box<dyn BackendRegister>> = Vec::new();

        #[cfg(feature = "cpu")]
        {
            use crate::cpu::backend_register::CpuBackendRegister;
            let reg = CpuBackendRegister::new();
            reg.probe_devices()?;
            registers.push(Box::new(reg));
        }
        #[cfg(feature = "opencl")]
        {
            use crate::opencl::backend_register::OpenclBackendRegister;
            let reg = OpenclBackendRegister::new();
            reg.probe_devices()?;
            registers.push(Box::new(reg));
        }
        #[cfg(feature = "cuda")]
        {
            use crate::cuda::backend_register::CudaBackendRegister;
            let reg = CudaBackendRegister::new();
            reg.probe_devices()?;
            registers.push(Box::new(reg));
        }

        Ok(Registry { registers })
    }

    pub fn init_all(&self) -> Result<()> {
        for reg in &self.registers {
            reg.init_devices()?;
        }
        Ok(())
    }

    pub fn registers(&self) -> &[Box<dyn BackendRegister>] {
        &self.registers
    }

    pub fn device_count(&self) -> usize {
        self.registers.iter().map(|r| r.device_count()).sum()
    }

    pub fn find(&self, name: &str) -> Option<&dyn BackendRegister> {
        self.registers.iter().find(|r| r.name().eq_ignore_ascii_case(name)).map(|r| r.as_ref())
    }

    pub fn open_backend(
        &self,
        backend_name: &str,
        device_index: usize,
    ) -> Result<Box<dyn Backend>> {
        let reg = self
            .find(backend_name)
            .ok_or_else(|| Error::msg(format!("backend '{}' not found", backend_name)))?;
        let device = reg.device(device_index)?;
        device.init_backend()
    }

    pub fn open_device(
        &self,
        backend_name: &str,
        device_index: usize,
    ) -> Result<Box<dyn BackendDevice>> {
        let reg = self
            .find(backend_name)
            .ok_or_else(|| Error::msg(format!("backend '{}' not found", backend_name)))?;
        reg.device(device_index)
    }

    pub fn open_best(&self) -> Result<Box<dyn Backend>> {
        if let Some(reg) = self.find("CUDA") {
            if reg.device_count() > 0 {
                return self.open_backend("CUDA", 0);
            }
        }
        if let Some(reg) = self.find("OpenCL") {
            if reg.device_count() > 0 {
                return self.open_backend("OpenCL", 0);
            }
        }
        if let Some(reg) = self.find("cpu") {
            if reg.device_count() > 0 {
                return self.open_backend("cpu", 0);
            }
        }
        Err(Error::msg("no backend available"))
    }

    pub fn best(&self) -> Option<&dyn BackendRegister> {
        self.find("CUDA")
            .or_else(|| self.find("OpenCL"))
            .or_else(|| self.find("cpu"))
            .filter(|r| r.device_count() > 0)
    }
}
