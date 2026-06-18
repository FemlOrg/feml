use std::cmp::min;

use super::super::backend::OpenclBackend;
use super::super::backend_context::ClKernelId;
use crate::error::{Error, Result};
use crate::tensor::Tensor;
use ocl::{core, core::ArgVal, Event, SpatialDims};

pub(crate) fn mul(
    backend: &OpenclBackend,
    src0: &Tensor,
    src1: &Tensor,
    dst: &Tensor,
) -> Result<()> {
    let mut backend_ctx = backend.backend_ctx.borrow_mut();

    let ne00 = src0.shape()[0];
    let ne01 = src0.shape()[1];
    let ne02 = src0.shape()[2];
    let ne03 = src0.shape()[3];

    let nb00 = src0.stride()[0];
    let nb01 = src0.stride()[1];
    let nb02 = src0.stride()[2];
    let nb03 = src0.stride()[3];

    let ne10 = src1.shape()[0];
    let ne11 = src1.shape()[1];
    let ne12 = src1.shape()[2];
    let ne13 = src1.shape()[3];

    let nb10 = src1.stride()[0];
    let nb11 = src1.stride()[1];
    let nb12 = src1.stride()[2];
    let nb13 = src1.stride()[3];

    let ne0 = dst.shape()[0];
    let ne1 = dst.shape()[1];
    let ne2 = dst.shape()[2];
    let ne3 = dst.shape()[3];

    let nb0 = dst.stride()[0];
    let nb1 = dst.stride()[1];
    let nb2 = dst.stride()[2];
    let nb3 = dst.stride()[3];

    let src0_storage = src0.storage()?;
    let src1_storage = src1.storage()?;
    let dst_storage = dst.storage()?;

    let offset0 = src0_storage.offset() + src0.view_offset();
    let offset1 = src1_storage.offset() + src1.view_offset();
    let offsetd = dst_storage.offset() + dst.view_offset();

    let src0_buffer = src0_storage
        .as_opencl()
        .ok_or_else(|| Error::msg("source 0 storage is not OpenCL type"))?;
    let src1_buffer = src1_storage
        .as_opencl()
        .ok_or_else(|| Error::msg("source 1 storage is not OpenCL type"))?;
    let dst_buffer = dst_storage
        .as_opencl()
        .ok_or_else(|| Error::msg("destination storage is not OpenCL type"))?;

    let nth = min(64, ne0);
    let global_work_size: [usize; 3] = [ne01 * nth, ne02, ne03];
    let local_work_size: [usize; 3] = [nth, 1, 1];
    let global_work_dims = SpatialDims::from(global_work_size);
    let local_work_dims = SpatialDims::from(local_work_size);
    let queue = backend_ctx.queue.clone();

    backend_ctx.with_kernel(ClKernelId::Mul, |kernel| {
        core::set_kernel_arg(&kernel, 0, ArgVal::mem(&src0_buffer.buffer))?;
        core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&offset0))?;
        core::set_kernel_arg(&kernel, 2, ArgVal::mem(&src1_buffer.buffer))?;
        core::set_kernel_arg(&kernel, 3, ArgVal::scalar(&offset1))?;
        core::set_kernel_arg(&kernel, 4, ArgVal::mem(&dst_buffer.buffer))?;
        core::set_kernel_arg(&kernel, 5, ArgVal::scalar(&offsetd))?;
        core::set_kernel_arg(&kernel, 6, ArgVal::scalar(&ne00))?;
        core::set_kernel_arg(&kernel, 7, ArgVal::scalar(&ne01))?;
        core::set_kernel_arg(&kernel, 8, ArgVal::scalar(&ne02))?;
        core::set_kernel_arg(&kernel, 9, ArgVal::scalar(&ne03))?;
        core::set_kernel_arg(&kernel, 10, ArgVal::scalar(&nb00))?;
        core::set_kernel_arg(&kernel, 11, ArgVal::scalar(&nb01))?;
        core::set_kernel_arg(&kernel, 12, ArgVal::scalar(&nb02))?;
        core::set_kernel_arg(&kernel, 13, ArgVal::scalar(&nb03))?;
        core::set_kernel_arg(&kernel, 14, ArgVal::scalar(&ne10))?;
        core::set_kernel_arg(&kernel, 15, ArgVal::scalar(&ne11))?;
        core::set_kernel_arg(&kernel, 16, ArgVal::scalar(&ne12))?;
        core::set_kernel_arg(&kernel, 17, ArgVal::scalar(&ne13))?;
        core::set_kernel_arg(&kernel, 18, ArgVal::scalar(&nb10))?;
        core::set_kernel_arg(&kernel, 19, ArgVal::scalar(&nb11))?;
        core::set_kernel_arg(&kernel, 20, ArgVal::scalar(&nb12))?;
        core::set_kernel_arg(&kernel, 21, ArgVal::scalar(&nb13))?;
        core::set_kernel_arg(&kernel, 22, ArgVal::scalar(&ne0))?;
        core::set_kernel_arg(&kernel, 23, ArgVal::scalar(&ne1))?;
        core::set_kernel_arg(&kernel, 24, ArgVal::scalar(&ne2))?;
        core::set_kernel_arg(&kernel, 25, ArgVal::scalar(&ne3))?;
        core::set_kernel_arg(&kernel, 26, ArgVal::scalar(&nb0))?;
        core::set_kernel_arg(&kernel, 27, ArgVal::scalar(&nb1))?;
        core::set_kernel_arg(&kernel, 28, ArgVal::scalar(&nb2))?;
        core::set_kernel_arg(&kernel, 29, ArgVal::scalar(&nb3))?;

        unsafe {
            core::enqueue_kernel(
                &queue,
                kernel,
                3,
                None,
                &global_work_size,
                Some(local_work_size),
                None::<Event>,
                None::<&mut Event>,
            )
            .expect("enqueue kernel_mul faield!");
        }
        Ok(())
    })?;

    Ok(())
}
