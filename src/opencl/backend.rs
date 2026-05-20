use crate::backend::{
    Backend,
    BackendBufferAllocator,
    BackendDevice,
    BackendDeviceCaps,
    BackendDeviceProps,
    BackendDeviceType,
    BackendRegister,
};
use crate::compute_graph::ComputeGraph;
use crate::error::{ Error, ErrorKind, Result };
use crate::tensor::Tensor;
use ocl::ocl_core;
use std::rc::Rc;
use std::sync::OnceLock;

static OPENCL_BACKEND_REG: OnceLock<OpenclBackendRegister> = OnceLock::new();

pub struct OpenclBackend {
    device: OpenclDevice,
    context: OpenclBackendContext,
}

pub struct OpenclBackendBuffer {
    buffers: Vec<ocl::Buffer<u8>>,
}

struct OpenclBackendContext {
    device: ocl::Device,
    device_name: String,
    context: ocl::Context,
    queue: ocl::Queue,
}

struct OpenclBackendRegister {
    devices: Vec<OpenclDevice>,
}

#[derive(Clone)]
pub struct OpenclDevice {
    platform: ocl::Platform,
    platform_name: String,
    device: ocl::Device,
    device_name: String,
    device_version: ocl_core::OpenclVersion,
    context: ocl::Context,
}

impl OpenclBackendContext {
    pub fn new(device: &OpenclDevice) -> Result<Self> {
        Ok(Self {
            device: device.device.clone(),
            device_name: device.device_name.clone(),
            context: device.context.clone(),
            queue: ocl::Queue::new(&device.context, device.device.clone(), None)?,
        })
    }
}

impl BackendBuffer for OpenclBackendBuffer {
    fn init_tensor(&self, tensor: Tensor, offset: usize) -> Result<()> {
        match tensor.borrow().view_tensor.clone() {
            Some(view_tensor) => {
                let extra_storage = view_tensor
                    .borrow()
                    .extra_storage.as_ref()
                    .ok_or_else(|| Error::msg("view extra_storage is None"))?
                    .clone();

                tensor.borrow_mut().extra_storage = Some(extra_storage);
            }
            None => {
                tensor.borrow_mut().extra_storage = Some(TensorStorage::OpenCL {
                    buffer: Rc::new(self),
                    ocl_buffer_idx: 0,
                    offset: offset,
                    acutal_size: tensor.nbytes(),
                });
            }
        }
        Ok(())
    }

    fn memset_tensor(
        &self,
        _tensor: Tensor,
        _value: u8,
        _offset: usize,
        _size: usize
    ) -> Result<()> {
        Ok(())
    }

    fn set_tensor(
        &self,
        _tensor: Tensor,
        _data: *mut u8,
        _offset: usize,
        _size: usize
    ) -> Result<()> {
        Ok(())
    }

    fn get_tensor(
        &self,
        _tensor: Tensor,
        _data: *mut u8,
        _offset: usize,
        _size: usize
    ) -> Result<()> {
        Ok(())
    }

    fn copy_tensor(&self, _src: Tensor, _dst: Tensor) -> Result<()> {
        Ok(())
    }
}

impl Backend for OpenclBackend {
    type Device = OpenclDevice;

    fn name(&self) -> &str {
        "opencl"
    }

    fn synchronize(&self) -> Result<()> {
        let event = self.context.queue.enqueue_marker(None::<()>)?;
        event.wait_for().map_err(ocl::Error::from)?;
        Ok(())
    }

    fn graph_compute(&self, _graph: &mut ComputeGraph) -> Result<()> {
        Err(Error::new(ErrorKind::UnsupportedBackendOp { backend: "opencl", op: "graph_compute" }))
    }

    fn memcpy_async(&self, _dst: *mut u8, _src: *const u8, _size: usize) -> Result<()> {
        Ok(())
    }

    fn set_tensor_async(
        &self,
        _tensor: Tensor,
        _data: *mut u8,
        _offset: usize,
        _size: usize
    ) -> Result<()> {
        Ok(())
    }

    fn get_tensor_async(
        &self,
        _tensor: Tensor,
        _data: *mut u8,
        _offset: usize,
        _size: usize
    ) -> Result<()> {
        Ok(())
    }

    fn copy_tensor_async(&self, _src: Tensor, _dst: Tensor) -> Result<()> {
        Ok(())
    }
}

impl OpenclBackend {
    pub fn init() -> Result<Self> {
        let reg = OpenclBackendRegister::init();
        let device = reg.opencl_device(0)?;
        let context = OpenclBackendContext::new(&device)?;

        Ok(Self { device, context })
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

    fn init(&self, _params: *mut u8) -> Result<()> {
        Ok(())
    }

    fn supports_op(&self, _tensor: Tensor) -> bool {
        false
    }

    fn supports_buffer_allocator(&self, _buffer_allocator: &dyn BackendBufferAllocator) -> bool {
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

    pub fn opencl_device(&self, index: usize) -> Result<OpenclDevice> {
        self.devices
            .get(index)
            .cloned()
            .ok_or_else(|| {
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

    fn probe_devices() -> Result<Vec<OpenclDevice>> {
        let mut opencl_devices: Vec<OpenclDevice> = Vec::new();
        let platforms = ocl::Platform::list();

        if platforms.is_empty() {
            return Ok(opencl_devices);
        }

        for platform in platforms {
            let devices = ocl::Device::list_all(&platform)?;
            if devices.is_empty() {
                continue;
            }

            let context = ocl::Context
                ::builder()
                .platform(platform.clone())
                .devices(&devices)
                .build()?;

            for device in devices {
                opencl_devices.push(OpenclDevice {
                    platform: platform.clone(),
                    platform_name: platform.name()?,
                    device: device.clone(),
                    device_name: device.name()?,
                    device_version: device.version().map_err(ocl::Error::from)?,
                    context: context.clone(),
                });
            }
        }

        Ok(opencl_devices)
    }
}
