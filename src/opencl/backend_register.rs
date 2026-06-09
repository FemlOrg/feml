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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_returns_opencl() {
        let reg = OpenclBackendRegister { devices: Vec::new() };
        assert_eq!(reg.name(), "OpenCL");
    }

    #[test]
    fn device_count_without_devices() {
        let reg = OpenclBackendRegister { devices: Vec::new() };
        assert_eq!(reg.device_count(), 0);
    }

    #[test]
    fn device_out_of_bounds_returns_error() {
        let reg = OpenclBackendRegister { devices: Vec::new() };
        let result = reg.device(0);
        assert!(result.is_err());
        let err_msg = format!("{}", result.err().unwrap());
        assert!(err_msg.contains("device 0 not found") || err_msg.contains("DeviceNotFound"));
    }

    #[test]
    fn singleton_init_returns_same_pointer() {
        let ptr1 = OpenclBackendRegister::init() as *const dyn BackendRegister;
        let ptr2 = OpenclBackendRegister::init() as *const dyn BackendRegister;
        assert_eq!(ptr1, ptr2);
    }

    #[test]
    fn singleton_has_correct_name() {
        let reg = OpenclBackendRegister::init();
        assert_eq!(reg.name(), "OpenCL");
    }

    #[test]
    fn as_any_roundtrip() {
        let reg = OpenclBackendRegister { devices: Vec::new() };
        let any = reg.as_any();
        assert!(any.is::<OpenclBackendRegister>());
    }

    #[test]
    fn as_any_downcast() {
        let reg = OpenclBackendRegister { devices: Vec::new() };
        let any = reg.as_any();
        let downcast = any.downcast_ref::<OpenclBackendRegister>();
        assert!(downcast.is_some());
    }

    #[test]
    fn as_any_not_other_type() {
        let reg = OpenclBackendRegister { devices: Vec::new() };
        let any = reg.as_any();
        assert!(!any.is::<String>());
    }

    #[test]
    fn opencl_device_out_of_bounds() {
        let reg = OpenclBackendRegister { devices: Vec::new() };
        let result = reg.opencl_device(0);
        assert!(result.is_err());
        let err_msg = format!("{}", result.err().unwrap());
        assert!(err_msg.contains("opencl") || err_msg.contains("0"));
    }

    #[test]
    fn device_count_with_device_count() {
        // We can't create real OpenclBackendDevice without hardware,
        // but we can verify the method works with empty vec
        let reg = OpenclBackendRegister { devices: Vec::new() };
        assert_eq!(reg.device_count(), 0);
    }
}
