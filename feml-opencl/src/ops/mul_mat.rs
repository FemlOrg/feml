//! OpenCL mul_mat kernel invocation (naive f32).

use crate::backend::BackendInner;
use feml_core::error::{Error, Result};
use ocl::SpatialDims;

fn ocl_err(e: ocl::Error) -> Error {
    Error::backend("OpenCL", e.to_string())
}

pub(crate) struct MatArg<'a> {
    pub buf: &'a ocl::Buffer<u8>,
    pub offset: usize,
    pub stride1: usize,
}

pub(crate) fn mul_mat(
    inner: &mut BackendInner,
    a: &MatArg<'_>,
    b: &MatArg<'_>,
    c: &MatArg<'_>,
    k: usize,
    m: usize,
    n: usize,
) -> Result<()> {
    let kernel = inner
        .kernels
        .get_mut("kernel_mul_mat")
        .ok_or_else(|| Error::msg("mul_mat: kernel not loaded"))?;

    kernel.set_arg(0, a.buf).map_err(ocl_err)?;
    kernel.set_arg(1, a.offset as u64).map_err(ocl_err)?;
    kernel.set_arg(2, a.stride1 as u64).map_err(ocl_err)?;
    kernel.set_arg(3, b.buf).map_err(ocl_err)?;
    kernel.set_arg(4, b.offset as u64).map_err(ocl_err)?;
    kernel.set_arg(5, b.stride1 as u64).map_err(ocl_err)?;
    kernel.set_arg(6, c.buf).map_err(ocl_err)?;
    kernel.set_arg(7, c.offset as u64).map_err(ocl_err)?;
    kernel.set_arg(8, c.stride1 as u64).map_err(ocl_err)?;
    kernel.set_arg(9, k as i32).map_err(ocl_err)?;
    kernel.set_arg(10, m as i32).map_err(ocl_err)?;
    kernel.set_arg(11, n as i32).map_err(ocl_err)?;

    kernel.set_default_global_work_size(SpatialDims::from([m * n]));

    unsafe { kernel.enq() }.map_err(ocl_err)
}
