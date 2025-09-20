use super::backend_trait::*;
use crate::common::def::FemlGuid;

pub enum FemlBackendBufferUsage {
    Any,
    Weights,
    Compute,
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

pub struct FemlBackend {
    pub guid: FemlGuid,
    pub interface: Box<dyn FemlBackendInterface>,
    pub device: FemlBackendDevice,
    pub context: *const u8,
}

pub struct FemlBackendDevice;

pub struct FemlBackendEvent;
// TODO
impl FemlBackendBufferType {
    // fn feml_backend_buffer_init(&self, interface: Box<dyn FemlBackendBufferInterface>, context: *mut u8, size: usize) -> FemlBackendBufferType{}

    // fn feml_backend_buffer_is_multi_buffer(&self) -> bool {}

    // fn feml_backend_multi_buffer_set_usage(&self, usage: &FemlBackendBufferUsage);
}

