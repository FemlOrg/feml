use super::backend_device::OpenclBackendDevice;
use crate::{
    backend::{BackendDevice, BackendRegister},
    error::{Error, ErrorKind, Result},
};
use std::any::Any;
use std::cell::RefCell;
use std::sync::Once;

static INIT: Once = Once::new();
static mut REG: Option<*const dyn BackendRegister> = None;

pub struct OpenclBackendRegister {
    pub(super) devices: RefCell<Vec<OpenclBackendDevice>>,
}

impl BackendRegister for OpenclBackendRegister {
    fn name(&self) -> &str {
        "OpenCL"
    }

    fn device_count(&self) -> usize {
        self.devices.borrow().len()
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

    fn probe_devices(&self) -> Result<()> {
        let mut opencl_devices: Vec<OpenclBackendDevice> = Vec::new();
        let platforms = ocl::Platform::list();

        if platforms.is_empty() {
            *self.devices.borrow_mut() = opencl_devices;
            return Ok(());
        }

        for platform in platforms {
            let devices = ocl::Device::list_all(&platform)?;
            if devices.is_empty() {
                continue;
            }

            let context =
                ocl::Context::builder().platform(platform.clone()).devices(&devices).build()?;

            for device in devices {
                let ocl_device = OpenclBackendDevice {
                    platform: platform.clone(),
                    platform_name: platform.name()?,
                    device: device.clone(),
                    device_name: device.name()?,
                    device_version: device.version().map_err(ocl::Error::from)?,
                    context: context.clone(),
                    backend_ctx: None,
                };
                opencl_devices.push(ocl_device);
            }
        }

        *self.devices.borrow_mut() = opencl_devices;
        Ok(())
    }

    fn init_devices(&self) -> Result<()> {
        let mut devices = self.devices.borrow_mut();
        let drained: Vec<OpenclBackendDevice> = devices.drain(..).collect();
        let valid: Vec<OpenclBackendDevice> = drained
            .into_iter()
            .filter_map(|mut d| if d.init().is_ok() { Some(d) } else { None })
            .collect();
        *devices = valid;
        Ok(())
    }
}

impl OpenclBackendRegister {
    pub fn init() -> &'static dyn BackendRegister {
        unsafe {
            INIT.call_once(|| {
                let reg = Self { devices: RefCell::new(Vec::new()) };
                REG = Some(Box::into_raw(Box::new(reg)));
            });
            &*REG.unwrap()
        }
    }

    pub fn opencl_device(&self, index: usize) -> Result<OpenclBackendDevice> {
        self.devices.borrow().get(index).cloned().ok_or_else(|| {
            Error::new(ErrorKind::DeviceNotFound {
                backend: "opencl",
                index,
                count: self.devices.borrow().len(),
            })
        })
    }

    pub fn new() -> Self {
        Self { devices: RefCell::new(Vec::new()) }
    }
}
