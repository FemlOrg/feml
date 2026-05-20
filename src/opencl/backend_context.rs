use super::backend_device::OpenclBackendDevice;
use ocl::{Context, Device, Queue};

pub(super) struct OpenclBackendContext {
    device: ocl::Device,
    device_name: String,
    context: ocl::Context,
    queue: ocl::Queue,
}

impl OpenclBackendContext {
    pub fn new(device: &OpenclBackendDevice) -> Result<Self> {
        Ok(Self {
            device: device.device.clone(),
            device_name: device.device_name.clone(),
            context: device.context.clone(),
            queue: ocl::Queue::new(&device.context, device.device.clone(), None)?,
        })
    }
}
