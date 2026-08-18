//! Byte-pattern fill kernel invocation.

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use feml_core::error::{Error, Result};

use crate::backend::BackendInner;
use crate::device::cuda_err;

pub(crate) fn fill(
    inner: &BackendInner,
    slice: &CudaSlice<u8>,
    offset: usize,
    value: u8,
    len: usize,
) -> Result<()> {
    let func =
        inner.funcs.get("kernel_fill").ok_or_else(|| Error::msg("fill: kernel not loaded"))?;

    // One grid dimension is 2^31-1 blocks; reject ranges a 1D grid cannot
    // address rather than silently truncating.
    let blocks = len.div_ceil(256);
    if blocks > i32::MAX as usize {
        return Err(Error::msg("fill: range too large for 1D grid"));
    }

    let view = slice.as_view();
    let off = offset as u64;
    let v = value;
    let l = len as u64;
    let cfg = LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: the caller bounds-checked `offset + len` against the slice; the
    // kernel writes exactly `len` bytes starting at `offset`.
    unsafe {
        inner
            .stream
            .launch_builder(func)
            .arg(&view)
            .arg(&off)
            .arg(&v)
            .arg(&l)
            .launch(cfg)
            .map_err(cuda_err)?;
    }
    Ok(())
}
