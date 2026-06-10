use crate::backend::{Backend, DeviceInfo};

pub struct Runtime {
    entries: Vec<DeviceEntry>,
}

pub struct DeviceEntry {
    pub info: DeviceInfo,
    init_fn: Box<dyn FnOnce() -> Result<Box<dyn Backend>>>,
}

impl Runtime {
    pub fn discover() -> Self {
        let mut entries = Vec::new();
        #[cfg(feature = "cpu")]
        entries.extend(CpuBackend::probe());
        #[cfg(feature = "opencl")]
        entries.extend(OpenclBackend::probe());
        #[cfg(feature = "cuda")]
        entries.extend(CudaBackend::probe());
        Self { entries }
    }

    pub fn devices(&self) -> &[DeviceEntry] {
        &self.entries
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn device_by_name(&self, name: &str) -> Option<&DeviceEntry> {
        self.entries.iter().find(|d| d.info.name.eq_ignore_ascii_case(name))
    }

    pub fn devices_by_type(&self, ty: DeviceType) -> Vec<&DeviceEntry> {
        self.entries.iter().filter(|d| d.info.device_type == ty).collect()
    }

    pub fn gpu(&self) -> Option<&DeviceEntry> {
        self.devices_by_type(DeviceType::Gpu).into_iter().next()
    }

    pub fn cpu(&self) -> Option<&DeviceEntry> {
        self.device_by_name("cpu")
    }

    pub fn best(&self) -> Option<&DeviceEntry> {
        self.gpu().or_else(|| self.cpu())
    }

    pub fn init_named(&self, name: &str) -> Result<Box<dyn Backend>> {
        let entry = self.device_by_name(name).ok_or_else(|| Error::backend_not_found(name))?;
        (entry.init_fn)()
    }

    pub fn init_gpu(&self) -> Result<Box<dyn Backend>> {
        let entry = self.gpu().ok_or_else(|| Error::no_gpu_available())?;
        (entry.init_fn)()
    }

    pub fn init_best(&self) -> Result<Box<dyn Backend>> {
        let entry = self.best().ok_or_else(|| Error::no_backend_available())?;
        (entry.init_fn)()
    }

    // pub fn load(self, path: &str) -> Result<Self> { ... }
    // pub fn load_all_from(self, dir: &str) -> Result<Self> { ... }
}
