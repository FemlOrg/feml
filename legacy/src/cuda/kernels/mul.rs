use super::common::get_cuda_tensor_ptr_and_size;
use crate::cuda::backend::CudaBackend;
use crate::error::Error;
use crate::error::Result;
use crate::tensor::Tensor;

use cuda_core::{DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use std::sync::Arc;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn mul(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[i] * b[i];
        }
    }
}

pub(crate) fn mul(backend: &CudaBackend, src0: &Tensor, src1: &Tensor, dst: &Tensor) -> Result<()> {
    let backend_ctx = backend
        .backend_ctx
        .borrow()
        .current_context
        .clone()
        .ok_or_else(|| Error::msg("current context is none"))?;
    let stream = backend.backend_ctx.borrow_mut().ensure_current_stream()?;

    let (src0_ptr, src0_size) = get_cuda_tensor_ptr_and_size(src0.clone())?;
    let (src1_ptr, src1_size) = get_cuda_tensor_ptr_and_size(src1.clone())?;
    let (dst_ptr, dst_size) = get_cuda_tensor_ptr_and_size(dst.clone())?;

    unsafe {
        let src0_buffer =
            DeviceBuffer::<f32>::from_raw_parts(src0_ptr, src0_size, Arc::clone(&backend_ctx));
        let src1_buffer =
            DeviceBuffer::<f32>::from_raw_parts(src1_ptr, src1_size, Arc::clone(&backend_ctx));
        let mut dst_buffer =
            DeviceBuffer::<f32>::from_raw_parts(dst_ptr, dst_size, Arc::clone(&backend_ctx));

        let module = kernels::load(&backend_ctx).expect("Failed to load embedded module");
        module
            .mul(
                &stream,
                LaunchConfig::for_num_elems(1024),
                &src0_buffer,
                &src1_buffer,
                &mut dst_buffer,
            )
            .unwrap();
    };

    Ok(())
}
