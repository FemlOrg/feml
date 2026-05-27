use crate::backend::{
    BackendBufferAllocator, BackendDevice, BackendDeviceCaps, BackendDeviceProps, BackendDeviceType,
};
use crate::error::Result;
use crate::tensor::Tensor;

#[derive(Clone)]
pub struct CpuBackendDevice {
    pub(super) description: String,
}

impl CpuBackendDevice {
    pub fn new() -> Self {
        Self { description: "CPU device".to_string() }
    }
}

impl BackendDevice for CpuBackendDevice {
    fn name(&self) -> &str {
        "cpu"
    }

    fn memory(&self) -> (usize, usize) {
        (0, 0)
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn device_type(&self) -> BackendDeviceType {
        BackendDeviceType::Cpu
    }

    fn props(&self) -> BackendDeviceProps {
        BackendDeviceProps {
            name: "cpu",
            description: "CPU device",
            memory_free: 0,
            memory_total: 0,
            device_type: BackendDeviceType::Cpu,
            caps: BackendDeviceCaps {
                aysnc: false,
                host_buffer: false,
                buffer_from_host_ptr: false,
                events: false,
            },
        }
    }

    fn init(&self, _params: &[u8]) -> Result<()> {
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
