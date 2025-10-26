use super::backend_trait::*;
use crate::common::context;
use crate::common::def::FemlGuid;
use std::any::Any;
use std::sync::Arc;

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
    pub device: Option<Arc<FemlBackendDevice>>,
    pub context: Option<Box<dyn Any>>,
}

pub struct FemlBackendBuffer {
    pub interface: Box<dyn FemlBackendBufferInterface>,
    pub buffer_type: Arc<FemlBackendBufferType>,
    pub context: Option<Box<dyn Any>>,
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
    pub guid: FemlGuid,
    pub interface: Box<dyn FemlBackendInterface>,
    pub device: Arc<FemlBackendDevice>,
    pub context: Option<Box<dyn Any>>,
}

pub struct FemlBackendEvent {
    pub interface: Box<dyn FemlBackendDeviceInterface>,
    pub context: Option<Box<dyn Any>>,
}
pub struct FemlBackendReg {
    pub interface: Box<dyn FemlBackendRegInterface>,
    pub context: Option<Box<dyn Any + Send + Sync>>,
    pub api_version: i32,
}

pub struct FemlBackendDevice {
    pub interface: Box<dyn FemlBackendDeviceInterface>,
    pub reg: Arc<FemlBackendReg>,
    pub context: Option<Box<dyn Any>>,
}

impl FemlBackend {
    pub fn new(
        guid: FemlGuid,
        interface: Box<dyn FemlBackendInterface>,
        device: Arc<FemlBackendDevice>,
        context: Option<Box<dyn Any>>,
    ) -> Self {
        FemlBackend { guid, interface, device, context }
    }

    pub fn set_context<T: 'static>(&mut self, context: T) {
        self.context = Some(Box::new(context));
    }

    pub fn get_context<T: 'static>(&mut self) -> Option<&mut T> {
        self.context.as_mut()?.downcast_mut::<T>()
    }
}

impl FemlBackendReg {
    pub fn new(
        interface: Box<dyn FemlBackendRegInterface>,
        context: Option<Box<dyn Any + Send + Sync>>,
        api_version: i32,
    ) -> Self {
        FemlBackendReg { interface, context, api_version }
    }

    pub fn set_context<T: 'static + Send + Sync>(&mut self, context: T) {
        self.context = Some(Box::new(context));
    }

    pub fn get_context<T: 'static>(&mut self) -> Option<&mut T> {
        self.context.as_mut()?.downcast_mut::<T>()
    }
}

impl FemlBackendDevice {
    pub fn new(
        interface: Box<dyn FemlBackendDeviceInterface>,
        reg: Arc<FemlBackendReg>,
        context: Option<Box<dyn Any>>,
    ) -> Self {
        FemlBackendDevice { interface, reg, context }
    }

    pub fn set_context<T: 'static>(&mut self, context: T) {
        self.context = Some(Box::new(context));
    }

    pub fn get_context<T: 'static>(&mut self) -> Option<&mut T> {
        self.context.as_mut()?.downcast_mut::<T>()
    }
}

impl FemlBackendBufferType {
    pub fn new(
        interface: Box<dyn FemlBackendBufferTypeInterface>,
        device: Option<Arc<FemlBackendDevice>>,
        context: Option<Box<dyn Any>>,
    ) -> Self {
        FemlBackendBufferType { interface, device, context }
    }

    pub fn set_context<T: 'static>(&mut self, context: T) {
        self.context = Some(Box::new(context));
    }

    pub fn get_context<T: 'static>(&mut self) -> Option<&mut T> {
        self.context.as_mut()?.downcast_mut::<T>()
    }
}

impl FemlBackendBuffer {
    pub fn new(
        interface: Box<dyn FemlBackendBufferInterface>,
        buffer_type: &Arc<FemlBackendBufferType>,
        context: Option<Box<dyn Any>>,
        size: usize,
    ) -> Self {
        FemlBackendBuffer {
            interface,
            buffer_type: buffer_type.clone(),
            context,
            size,
            usage: FemlBackendBufferUsage::Any,
        }
    }

    pub fn set_context<T: 'static>(&mut self, context: T) {
        self.context = Some(Box::new(context));
    }

    pub fn get_context<T: 'static>(&mut self) -> Option<&mut T> {
        self.context.as_mut()?.downcast_mut::<T>()
    }
}

pub(crate) fn feml_backend_buffer_init(
    buft: Arc<FemlBackendBufferType>,
    interface: &mut Option<Box<dyn FemlBackendBufferInterface>>,
    context: Option<Box<dyn Any>>,
    size: usize,
) -> FemlBackendBuffer {
    let iface = interface.take().expect("interface must be name");
    FemlBackendBuffer::new(iface, &buft, context, size)
}
