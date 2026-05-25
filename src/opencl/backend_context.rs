use super::backend_device::OpenclBackendDevice;
use ocl::{Context, Device, Queue};

pub(super) enum OpenclGpuFamlily {
    Intel,
    Qualcomm,
    Unknown,
}

pub(super) struct OpenclBackendContext {
    pub(super) device: ocl::Device,
    pub(super) device_name: String,
    pub(super) context: ocl::Context,
    pub(super) queue: ocl::Queue,
    pub(super) wave_size: i32,
    pub(super) gpu_family: OpenclGpuFamlily,
}

impl OpenclBackendContext {
    pub fn new(device: &OpenclBackendDevice) -> Result<Self> {
        let mut props = CommandQueueProperties::ON_DEVICE_DEFAULT;
        #[cfg(feature = "opencl-profiling")]
        {
            props.profiling();
        }

        Ok(Self {
            device: device.device.clone(),
            device_name: device.device_name.clone(),
            context: device.context.clone(),
            queue: ocl::Queue::new(&device.context, device.device, props)?,
            wave_size: 0,
            gpu_family: OpenclGpuFamlily::Unknown,
        })
    }
}
