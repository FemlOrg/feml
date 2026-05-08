use crate::backend::{
    Backend, BackendBufferAllocator, BackendDevice, BackendDeviceCaps, BackendDeviceProps,
    BackendDeviceType, BackendRegister,
};
use crate::compute_graph::ComputeGraph;
use crate::defs::Status;
use crate::tensor::Tensor;
use ocl::core::ContextProperties;
use ocl::{ocl_core, Context, Device, Event, Platform, Queue};
use std::sync::OnceLock;

static OPENCL_BACKEND_REG: OnceLock<OpenclBackendRegister> = OnceLock::new();

struct OpenclBackendContext {
    device: ocl::Device,
    device_name: String,
    context: ocl::Context,
    queue: ocl::Queue,
}

pub struct OpenclBackend {
    device: OpenclDevice,
    context: OpenclBackendContext,
}

struct OpenclBackendRegister {
    devices: Vec<OpenclDevice>,
}

#[derive(Clone)]
struct OpenclDevice {
    platform: ocl::Platform,
    platform_name: String,
    device: ocl::Device,
    device_name: String,
    device_version: ocl_core::OpenclVersion,
    context: ocl::Context,
}

impl Default for OpenclBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenclBackendContext {
    pub fn new(device: &OpenclDevice) -> Self {
        OpenclBackendContext {
            device: device.device.clone(),
            device_name: device.device_name,
            context: device.context.clone(),
            queue: ocl::Queue::new(&device.context, device.device, None)?,
        }
    }
}

impl Backend for OpenclBackend {
    type Device = OpenclDevice;

    fn get_name(&self) -> &str {
        "opencl"
    }

    fn synchronize(&self) {
        let mut event = self.context.queue.enqueue_marker(None)?;
        event.wait_for();
    }

    fn graph_compute(&self, _graph: &mut ComputeGraph) -> Status {
        Status::Aborted
    }

    fn memcpy_async(&self, _dst: *mut u8, _src: *const u8, _size: usize) {}

    fn init_tensor(&self, _tensor: Tensor) {}

    fn memset_tensor(&self, _tensor: Tensor, _value: u8, _offset: usize, _size: usize) {}

    fn set_tensor(&self, _tensor: Tensor, _data: *mut u8, _offset: usize, _size: usize) {}

    fn get_tensor(&self, _tensor: Tensor, _data: *mut u8, _offset: usize, _size: usize) {}

    fn copy_tensor(&self, _src: Tensor, _dst: Tensor) {}

    fn set_tensor_async(&self, _tensor: Tensor, _data: *mut u8, _offset: usize, _size: usize) {}

    fn get_tensor_async(&self, _tensor: Tensor, _data: *mut u8, _offset: usize, _size: usize) {}

    fn copy_tensor_async(&self, _src: Tensor, _dst: Tensor) {}
}

impl OpenclBackend {
    pub fn init() -> Self {
        let reg = OpenclBackendRegister::init();
        let dev = reg.opencl_device(0);
        let backend_ctx = OpenclBackendContext::new(&dev);
        OpenclBackend { device: dev, context: backend_ctx }
    }
}

impl BackendDevice for OpenclDevice {
    fn name(&self) -> &str {
        "opencl"
    }

    fn memory(&self) -> (usize, usize) {
        (0, 0)
    }

    fn description(&self) -> &str {
        "OpenCL device"
    }

    fn device_type(&self) -> BackendDeviceType {
        BackendDeviceType::Gpu
    }

    fn props(&self) -> BackendDeviceProps {
        BackendDeviceProps {
            name: "opencl",
            description: "OpenCL device",
            memory_free: 0,
            memory_total: 0,
            device_type: BackendDeviceType::Gpu,
            caps: BackendDeviceCaps {
                aysnc: false,
                host_buffer: false,
                buffer_from_host_ptr: false,
                events: false,
            },
        }
    }

    fn init(&self, _params: *mut u8) {}

    fn supports_op(&self, _tensor: Tensor) -> bool {
        false
    }

    fn supports_buffer_allocator(
        &self,
        _buffer_allocator: &Box<dyn BackendBufferAllocator>,
    ) -> bool {
        false
    }

    fn offload_op(&self, _tensor: Tensor) -> bool {
        false
    }
}

impl BackendRegister for OpenclBackendRegister {
    fn name(&self) -> &str {
        "OpenCL"
    }

    fn device_count(&self) -> usize {
        self.devices.len()
    }

    fn device(&self, index: usize) -> Box<dyn BackendDevice> {
        Box::new(self.devices[index].clone())
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

    pub fn opencl_device(&self, index: usize) -> OpenclDevice {
        self.devices[index].clone()
    }

    fn try_new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self { devices: Self::probe_devices()? })
    }

    fn probe_devices() -> Result<Vec<OpenclDevice>, Box<dyn std::error::Error>> {
        let mut opencl_devices: Vec<OpenclDevice> = Vec::new();

        let platforms = ocl::Platform::list();

        if platforms.is_empty() {
            eprintln!("opencl: cannot find any platform!");
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
                opencl_devices.push(OpenclDevice {
                    platform: platform.clone(),
                    platform_name: platform.name().ok()?,
                    device: device.clone(),
                    device_name: device.name().ok()?,
                    device_version: device.version().ok()?,
                    context: context.clone(),
                });
            }
        }

        Ok(opencl_devices)
    }
}
