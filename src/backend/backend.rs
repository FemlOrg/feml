use super::backend_trait::*;

pub enum FemlBackendBufferUsage {
    Any,
    Weights,
    Compute,
}

pub enum FemlBackendDeviceType {
    CPU,
    GPU,
    ACCEL,
}

pub struct FemlBackendDevCaps {
    pub is_async: bool,
    pub is_host_buffer: bool,
    pub is_buffer_from_host_ptr: bool,
    pub is_events: bool,
}

pub struct FemlBackendBufferType {
    pub interface: Box<dyn FemlBackendBufferTypeInterface>,
    pub device: FemlBackendDevice,
    pub context: *mut u8,
}

pub struct FemlBackendBuffer {
    pub interface: Box<dyn FemlBackendBufferInterface>,
    pub buffer_type: FemlBackendBufferType,
    pub context: *mut u8,
    pub size: usize,
    pub usage: FemlBackendBufferUsage,
}

pub struct FemlBackendDeviceProps {
    pub name: String,
    pub description: String,
    pub free: u64,
    pub total: u64,
    pub backend_type: FemlBackendDeviceType,
    pub caps: FemlBackendDevCaps,
}

pub struct FemlBackend {
    pub guid: Vec<u8>,
    pub interface: Box<dyn FemlBackendInterface>,
    pub device: FemlBackendDevice,
    pub context: *const u8,
}

pub struct FemlBackendEvent {
    pub interface: Box<dyn FemlBackendDeviceInterface>,
    pub context: *mut u8,
}

pub struct FemlBackendReg {
    pub interface: Box<dyn FemlBackendRegInterface>,
    pub context: *mut u8,
    pub api_version: i32,
}

pub struct FemlBackendDevice {
    pub interface: Box<dyn FemlBackendDeviceInterface>,
    pub reg: FemlBackendReg,
    pub context: *mut u8,
}

// TODO
impl FemlBackendBufferType {
    // fn feml_backend_buffer_init(&self, interface: Box<dyn FemlBackendBufferInterface>, context: *mut u8, size: usize) -> FemlBackendBufferType{}

    // fn feml_backend_buffer_is_multi_buffer(&self) -> bool {}

    // fn feml_backend_multi_buffer_set_usage(&self, usage: &FemlBackendBufferUsage);
}

fn feml_backend_reg_dev_get(reg: &FemlBackendReg, index: usize) -> FemlBackendDevice {
    return reg.interface.get_device(reg, index);
}
