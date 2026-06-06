use std::cmp::min;

use super::super::backend::OpenclBackend;
use super::super::backend_context::ClKernelId;
use crate::error::{Error, Result};
use crate::opencl::backend_buffer::OpenclBackendBuffer;
use crate::tensor::Tensor;
use ocl::SpatialDims;

pub(crate) fn mul(
    backend: &OpenclBackend,
    src0: &Tensor,
    src1: &Tensor,
    dst: &Tensor,
) -> Result<()> {
    let backend_ctx = backend.context.borrow_mut();
    let kernel_mul = backend_ctx.get_kernel(ClKernelId::Mul)?;

    let ne00 = src0.get_shape()[0];
    let ne01 = src0.get_shape()[1];
    let ne02 = src0.get_shape()[2];
    let ne03 = src0.get_shape()[3];

    let nb00 = src0.get_stride()[0];
    let nb01 = src0.get_stride()[1];
    let nb02 = src0.get_stride()[2];
    let nb03 = src0.get_stride()[3];

    let ne10 = src1.get_shape()[0];
    let ne11 = src1.get_shape()[1];
    let ne12 = src1.get_shape()[2];
    let ne13 = src1.get_shape()[3];

    let nb10 = src1.get_stride()[0];
    let nb11 = src1.get_stride()[1];
    let nb12 = src1.get_stride()[2];
    let nb13 = src1.get_stride()[3];

    let ne0 = dst.get_shape()[0];
    let ne1 = dst.get_shape()[1];
    let ne2 = dst.get_shape()[2];
    let ne3 = dst.get_shape()[3];

    let nb0 = dst.get_stride()[0];
    let nb1 = dst.get_stride()[1];
    let nb2 = dst.get_stride()[2];
    let nb3 = dst.get_stride()[3];

    let src0_storage = src0.get_extra_storage()?;
    let src1_storage = src1.get_extra_storage()?;
    let dst_storage = dst.get_extra_storage()?;

    let offset0 = src0_storage.offset + src0.get_view_offset();
    let offset1 = src1_storage.offset + src1.get_view_offset();
    let offsetd = dst_storage.offset + dst.get_view_offset();

    let src0_buffer = src0_storage
        .as_opencl()
        .ok_or_else(|| Error::msg("source 0 storage is not OpenCL type"))?;
    let src1_buffer = src1_storage
        .as_opencl()
        .ok_or_else(|| Error::msg("source 1 storage is not OpenCL type"))?;
    let dst_buffer = dst_storage
        .as_opencl()
        .ok_or_else(|| Error::msg("destination storage is not OpenCL type"))?;

    let mut idx = 0;
    kernel_mul.set_arg(idx += 1, src0_buffer);
    kernel_mul.set_arg(idx += 1, offset0);
    kernel_mul.set_arg(idx += 1, src1_buffer);
    kernel_mul.set_arg(idx += 1, offset1);
    kernel_mul.set_arg(idx += 1, dst_buffer);
    kernel_mul.set_arg(idx += 1, offsetd);
    kernel_mul.set_arg(idx += 1, ne00);
    kernel_mul.set_arg(idx += 1, ne01);
    kernel_mul.set_arg(idx += 1, ne02);
    kernel_mul.set_arg(idx += 1, ne03);
    kernel_mul.set_arg(idx += 1, nb00);
    kernel_mul.set_arg(idx += 1, nb01);
    kernel_mul.set_arg(idx += 1, nb02);
    kernel_mul.set_arg(idx += 1, nb03);
    kernel_mul.set_arg(idx += 1, ne10);
    kernel_mul.set_arg(idx += 1, ne11);
    kernel_mul.set_arg(idx += 1, ne12);
    kernel_mul.set_arg(idx += 1, ne13);
    kernel_mul.set_arg(idx += 1, nb10);
    kernel_mul.set_arg(idx += 1, nb11);
    kernel_mul.set_arg(idx += 1, nb12);
    kernel_mul.set_arg(idx += 1, nb13);
    kernel_mul.set_arg(idx += 1, ne0);
    kernel_mul.set_arg(idx += 1, ne1);
    kernel_mul.set_arg(idx += 1, ne2);
    kernel_mul.set_arg(idx += 1, ne3);
    kernel_mul.set_arg(idx += 1, nb0);
    kernel_mul.set_arg(idx += 1, nb1);
    kernel_mul.set_arg(idx += 1, nb2);
    kernel_mul.set_arg(idx += 1, nb3);

    let mut nth = min(64, ne0);
    let global_work_size: [usize; 3] = [ne01 * nth, ne02, ne03];
    let local_work_size: [usize; 3] = [nth, 1, 1];
    let global_work_dims = SpatialDims::from(global_work_size);
    let local_work_dims = SpatialDims::from(local_work_size);

    kernel_mul.set_default_global_work_size(global_work_dims);
    kernel_mul.set_default_local_work_size(local_work_dims);

    backend_ctx.enqueue_ndrange_kernel(kernel_mul, &global_work_dims, &local_work_dims)?;
    Ok(())
}
