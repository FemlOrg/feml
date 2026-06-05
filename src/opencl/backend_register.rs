use super::backend_device::OpenclBackendDevice;
use crate::{
    backend::{BackendDevice, BackendRegister},
    error::{Error, ErrorKind, Result},
};
use std::any::Any;
use std::sync::Once;

static INIT: Once = Once::new();
static mut REG: Option<*const dyn BackendRegister> = None;

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
        Ok(Box::new(self.opencl_device(index)?.clone()))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl OpenclBackendRegister {
    pub fn init() -> &'static dyn BackendRegister {
        unsafe {
            INIT.call_once(|| {
                let reg = Self::try_new().unwrap_or_else(|err| {
                    eprintln!("opencl: failed to initialize backend register: {err}");
                    Self { devices: Vec::new() }
                });
                REG = Some(Box::into_raw(Box::new(reg)));
            });
            &*REG.unwrap()
        }
    }

    pub fn opencl_device(&self, index: usize) -> Result<&OpenclBackendDevice> {
        self.devices.get(index).ok_or_else(|| {
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
                    backend_ctx: None,
                };
                if ocl_device.init().is_ok() {
                    opencl_devices.push(ocl_device);
                }
            }
        }

        Ok(opencl_devices)
    }
}
