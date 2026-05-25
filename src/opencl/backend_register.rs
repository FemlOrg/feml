use super::backend_device::OpenclBackendDevice;
use crate::{backend::BackendDevice, opencl::backend_context::OpenclBackendContext};
use std::sync::OnceLock;

static OPENCL_BACKEND_REG: OnceLock<OpenclBackendRegister> = OnceLock::new();

pub(super) struct OpenclBackendRegister {
    devices: Vec<OpenclBackendDevice>,
}

impl BackendRegister for OpenclBackendRegister {
    fn name(&self) -> &str {
        "OpenCL"
    }

    fn device_count(&self) -> usize {
        self.devices.len()
    }

    fn device(&self, index: usize) -> Result<Box<dyn BackendDevice>> {
        Ok(Box::new(self.opencl_device(index)?))
    }
}

impl OpenclBackendRegister {
    pub fn init() -> &'static Self {
        OPENCL_BACKEND_REG.get_or_init(|| {
            Self::try_new().unwrap_or_else(|err| {
                eprintln!("opencl: failed to initialize backend register: {err}");

                Self { devices: Vec::new() }
            })
        })
    }

    pub fn opencl_device(&self, index: usize) -> Result<OpenclBackendDevice> {
        self.devices.get(index).cloned().ok_or_else(|| {
            Error::new(ErrorKind::DeviceNotFound {
                backend: "opencl",
                index,
                count: self.devices.len(),
            })
        })
    }

    fn try_new() -> Result<Self> {
        Ok(Self { devices: Self::probe_devices()? })
    }

    fn probe_devices() -> Result<Vec<OpenclBackendDevice>> {
        let mut opencl_devices: Vec<OpenclBackendDevice> = Vec::new();
        let platforms = ocl::Platform::list();

        if platforms.is_empty() {
            return Ok(opencl_devices);
        }

        for platform in platforms {
            let devices = ocl::Device::list_all(&platform)?;
            if devices.is_empty() {
                continue;
            }

            let context =
                ocl::Context::builder().platform(platform.clone()).devices(&devices).build()?;

            for device in devices {
                let mut ocl_device = OpenclBackendDevice {
                    platform: platform.clone(),
                    platform_name: platform.name()?,
                    device: device.clone(),
                    device_name: device.name()?,
                    device_version: device.version().map_err(ocl::Error::from)?,
                    context: context.clone(),
                    backend_ctx: OpenclBackendContext::default(),
                };
                if ocl_device.init().is_ok() {
                    opencl_devices.push(ocl_device);
                }
                opencl_devices.push(ocl_device);
            }
        }

        Ok(opencl_devices)
    }
}
