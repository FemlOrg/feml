use crate::error::Error;
use crate::error::Result;
use cuda_core::CudaContext;
use cuda_core::CudaStream;
use cuda_core::DeviceBuffer;
use std::sync::Arc;

const CUDA_MAX_DEVICES: usize = 10;

pub(super) struct CudaBackendContext {
    current_device_id: Option<u32>,
    current_context: Option<Arc<CudaContext>>,
    current_stream: Option<Arc<CudaStream>>,
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
            return Err(Error::msg(format!("device id {} out of bounds", device_id))
                .context("CudaBackendContext::set_device"));
        }

        let ctx = match &self.context[idx] {
            Some(ctx) => Arc::clone(ctx),
            None => {
                let ctx = CudaContext::new(idx)?;
                self.context[idx] = Some(Arc::clone(&ctx));
                ctx
            }
        };

        self.current_device_id = Some(device_id);
        self.current_context = Some(Arc::clone(&ctx));
        self.current_stream = Some(ctx.default_stream());
        Ok(())
    }

    pub(super) fn get_device(&self) -> Result<Arc<CudaContext>> {
        self.current_context.ok_or_else(|| {
            Error::msg("current context is None").context("CudaBackendContext::get_device")
        })
    }

    pub(super) fn device_malloc(&self, size: usize) -> Result<DeviceBuffer<u8>> {
        let buffer = DeviceBuffer::<u8>::zeroed(self.current_stream.unwrap().as_ref(), size)?;
        Ok(buffer)
    }
}
