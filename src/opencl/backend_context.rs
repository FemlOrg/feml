use super::backend_device::OpenclBackendDevice;
use ocl::{Context, Device, Queue};

pub(super) struct OpenclBackendContext {
    pub(super) device: Option<ocl::Device>,
    pub(super) device_name: Option<String>,
    pub(super) context: Option<ocl::Context>,
    pub(super) queue: Option<ocl::Queue>,
}

impl Default for OpenclBackendContext {
    fn default() -> Self {
        OpenclBackendContext { device: None, device_name: None, context: None, queue: None }
    }
}

impl OpenclBackendContext {
    pub fn new(device: &OpenclBackendDevice) -> Result<Self> {
        Ok(Self {
            device: Some(device.device.clone()),
            device_name: Some(device.device_name.clone()),
            context: Some(device.context.clone()),
            queue: Some(ocl::Queue::new(&device.context, device.device.clone(), None)?),
        })
    }
}
