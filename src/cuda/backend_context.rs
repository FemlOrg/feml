use crate::cuda::backend_register::CudaBackendRegister;
use crate::error::Error;
use crate::error::Result;
use cuda_core::CudaContext;
use cuda_core::CudaStream;
use cuda_core::DeviceBuffer;
use std::cell::RefCell;
use std::rc::Weak;
use std::sync::Arc;

pub(super) struct CudaBackendContext {
    pub(super) current_device_id: Option<u32>,
    pub(super) current_context: Option<Arc<CudaContext>>,
    pub(super) current_stream: Option<Arc<CudaStream>>,
    pub(super) register: Option<Weak<RefCell<CudaBackendRegister>>>,
}

impl CudaBackendContext {
    pub(super) fn new() -> Self {
        Self {
            current_device_id: None,
            current_context: None,
            current_stream: None,
            register: None,
        }
    }

    pub(super) fn set_device(&mut self, device_id: u32) -> Result<()> {
        if self.current_device_id == Some(device_id) {
            return Ok(());
        }

        let reg = self
            .register
            .as_ref()
            .and_then(|w| w.upgrade())
            .ok_or_else(|| Error::msg("register unavailable"))?;
        let reg = reg.borrow();
        let device = reg
            .devices
            .borrow()
            .get(device_id as usize)
            .ok_or_else(|| Error::msg(format!("device {} not found", device_id)))?
            .clone();

        let ctx = Arc::clone(&device.context);
        let stream = Arc::clone(&device.stream);

        self.current_device_id = Some(device_id);
        self.current_context = Some(ctx);
        self.current_stream = Some(stream);
        Ok(())
    }

    pub(super) fn get_device(&self) -> Result<Arc<CudaContext>> {
        self.current_context.ok_or_else(|| {
            Error::msg("current context is None").context("CudaBackendContext::get_device")
        })
    }

    pub(super) fn get_device_id(&self) -> Result<u32> {
        self.current_device_id.ok_or_else(|| {
            Error::msg("current device id is None").context("CudaBackendContext::get_device_id")
        })
    }

    pub(super) fn device_malloc(&self, size: usize) -> Result<DeviceBuffer<u8>> {
        let buffer = DeviceBuffer::<u8>::zeroed(self.current_stream.unwrap().as_ref(), size)?;
        Ok(buffer)
    }

    pub(super) fn ensure_current_stream(&self) -> Result<Arc<CudaStream>> {
        let device_id =
            self.current_device_id.ok_or_else(|| Error::msg("current device id is none"))?;

        self.set_device(device_id)?;
        self.current_stream.as_ref().cloned().ok_or_else(|| Error::msg("current stream is none"))
    }
}
