use super::backend_context::OpenclBackendContext;
use super::backend_device::OpenclBackendDevice;
use crate::{
    backend::{BackendDevice, BackendRegister},
    error::{Error, ErrorKind, Result},
};
use std::cell::RefCell;
use std::sync::Once;
use std::{any::Any, rc::Rc};

static INIT: Once = Once::new();
static mut REG: Option<*const dyn BackendRegister> = None;

pub(crate) struct OpenclBackendRegister {
    pub(super) devices: RefCell<Vec<OpenclBackendDevice>>,
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

    fn probe_devices(&self) -> Result<()> {
        let mut opencl_devices: Vec<OpenclBackendDevice> = Vec::new();
        let platforms = ocl::Platform::list();

        if platforms.is_empty() {
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
                let mut ocl_device = OpenclBackendDevice {
                    platform: platform.clone(),
                    platform_name: platform.name()?,
                    device: device.clone(),
                    device_name: device.name()?,
                    device_version: device.version().map_err(ocl::Error::from)?,
                    context: context.clone(),
                    backend_ctx: None,
                };
                self.devices.push(ocl_device);
            }
        }

        Ok(())
    }

    fn init_devices(&self) -> Result<()> {
        let devices: Vec<OpenclBackendDevice> = std::mem::take(&mut *self.devices.borrow_mut());
        let ctx = self.backend_ctx.clone();

        let valid_devices: Vec<OpenclBackendDevice> =
            devices.into_iter().filter(|device| device.init().is_ok()).collect();

        *self.devices.borrow_mut() = valid_devices;

        Ok(())
    }
}

impl OpenclBackendRegister {
    pub fn init() -> &'static dyn BackendRegister {
        unsafe {
            INIT.call_once(|| {
                let reg = Rc::try_unwrap(Self::new()).ok().unwrap().into_inner();
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

    pub(crate) fn new() -> Rc<RefCell<Self>> {
        let reg = Rc::new(RefCell::new(Self { devices: RefCell::new(Vec::new()) }));
        reg.borrow_mut().backend_ctx.borrow_mut().register = Some(Rc::downgrade(&reg));
        reg
    }
}
