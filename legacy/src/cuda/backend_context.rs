use super::common::CUDA_MAX_DEVICES;
use crate::error::Error;
use crate::error::Result;
use cuda_core::CudaContext;
use cuda_core::CudaStream;
use cuda_core::DeviceBuffer;
use std::sync::Arc;

pub(crate) struct CudaBackendContext {
    pub(super) current_device_id: Option<u32>,
    pub(super) current_context: Option<Arc<CudaContext>>,
    pub(super) current_stream: Option<Arc<CudaStream>>,
    context: [Option<Arc<CudaContext>>; CUDA_MAX_DEVICES],
}

impl CudaBackendContext {
    pub(super) fn new() -> Self {
        Self {
            current_device_id: None,
            current_context: None,
            current_stream: None,
            context: Default::default(),
        }
    }

    pub(super) fn set_device(&mut self, device_id: u32) -> Result<()> {
        if self.current_device_id == Some(device_id) {
            return Ok(());
        }

        let idx = device_id as usize;
        if idx >= CUDA_MAX_DEVICES {
            return Err(Error::msg(format!("device id {} out of bounds", device_id)));
        }

        let ctx = match &self.context[idx] {
            Some(ctx) => Arc::clone(ctx),
            None => {
                let ctx = CudaContext::new(idx)?;
                self.context[idx] = Some(Arc::clone(&ctx));
                ctx
            }
        };

        let stream = ctx.default_stream();

        self.current_device_id = Some(device_id);
        self.current_context = Some(ctx);
        self.current_stream = Some(stream);
        Ok(())
    }

    pub(super) fn get_device(&self) -> Result<Arc<CudaContext>> {
        self.current_context.clone().ok_or_else(|| {
            Error::msg("current context is None").context("CudaBackendContext::get_device")
        })
    }

    pub(super) fn get_device_id(&self) -> Result<u32> {
        self.current_device_id.ok_or_else(|| {
            Error::msg("current device id is None").context("CudaBackendContext::get_device_id")
        })
    }

    pub(super) fn device_malloc(&self, size: usize) -> Result<DeviceBuffer<u8>> {
        let stream =
            self.current_stream.clone().ok_or_else(|| Error::msg("current stream is None"))?;
        DeviceBuffer::<u8>::zeroed(stream.as_ref(), size)
            .map_err(|e| Error::msg(format!("DeviceBuffer::zeroed failed: {}", e)))
    }

    pub(super) fn ensure_current_stream(&mut self) -> Result<Arc<CudaStream>> {
        let device_id =
            self.current_device_id.ok_or_else(|| Error::msg("current device id is none"))?;

        self.set_device(device_id)?;
        self.current_stream.clone().ok_or_else(|| Error::msg("current stream is none"))
    }
}
