//! Strided elementwise mul/add kernels (4D broadcast, ggml semantics).
//!
//! Operands are passed as shared `&CudaSlice` references (the launch only
//! reads the device pointer); offsets/strides travel as kernel scalars (the
//! OpenCL backend model). Aliased operands pass the same slice reference
//! twice, which is safe: identical-range inplace ops read each element before
//! writing it within a thread, and partial overlap is rejected by the caller.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use feml_core::error::{Error, Result};
use feml_core::plan::{GraphPlan, PlanTensor};

use crate::backend::BackendInner;
use crate::device::cuda_err;

pub(crate) fn elementwise_binary(
    inner: &BackendInner,
    plan: &GraphPlan,
    sa: &PlanTensor,
    sb: &PlanTensor,
    dst: &PlanTensor,
    is_mul: bool,
) -> Result<()> {
    let func = inner
        .funcs
        .get(if is_mul { "kernel_mul" } else { "kernel_add" })
        .ok_or_else(|| Error::msg("elementwise: kernel not loaded"))?;

    // The planner packs several tensors into one buffer per usage class, so
    // aliasing is a byte-RANGE question, not a handle question.
    fn ranges_overlap(plan: &GraphPlan, a: &PlanTensor, b: &PlanTensor) -> bool {
        if plan.buffers[a.buffer].handle != plan.buffers[b.buffer].handle {
            return false;
        }
        let a_end = a.offset + a.nbytes();
        let b_end = b.offset + b.nbytes();
        a.offset < b_end && b.offset < a_end
    }

    // Partial overlap of dst with a source would race across threads; reject
    // it (identical ranges are read-before-write safe). Mirrors feml-cpu.
    if ranges_overlap(plan, sa, dst) && (sa.offset != dst.offset || sa.nbytes() != dst.nbytes()) {
        return Err(Error::msg("CUDA: partial-overlap alias not supported"));
    }
    if ranges_overlap(plan, sb, dst) && (sb.offset != dst.offset || sb.nbytes() != dst.nbytes()) {
        return Err(Error::msg("CUDA: partial-overlap alias not supported"));
    }

    let h_a = plan.buffers[sa.buffer].handle;
    let h_b = plan.buffers[sb.buffer].handle;
    let h_c = plan.buffers[dst.buffer].handle;
    let a = inner
        .buffers
        .get(&h_a)
        .ok_or_else(|| Error::msg(format!("elementwise: unknown buffer {h_a}")))?;
    let b = inner
        .buffers
        .get(&h_b)
        .ok_or_else(|| Error::msg(format!("elementwise: unknown buffer {h_b}")))?;
    let c = inner
        .buffers
        .get(&h_c)
        .ok_or_else(|| Error::msg(format!("elementwise: unknown buffer {h_c}")))?;

    let (ne00, ne01, ne02, ne03) = (
        sa.layout.shape[0] as i32,
        sa.layout.shape[1] as i32,
        sa.layout.shape[2] as i32,
        sa.layout.shape[3] as i32,
    );
    let (nb00, nb01, nb02, nb03) = (
        sa.layout.stride[0] as u64,
        sa.layout.stride[1] as u64,
        sa.layout.stride[2] as u64,
        sa.layout.stride[3] as u64,
    );
    let (ne10, ne11, ne12, ne13) = (
        sb.layout.shape[0] as i32,
        sb.layout.shape[1] as i32,
        sb.layout.shape[2] as i32,
        sb.layout.shape[3] as i32,
    );
    let (nb10, nb11, nb12, nb13) = (
        sb.layout.stride[0] as u64,
        sb.layout.stride[1] as u64,
        sb.layout.stride[2] as u64,
        sb.layout.stride[3] as u64,
    );
    let (ne0, ne1, ne2, ne3) = (
        dst.layout.shape[0] as i32,
        dst.layout.shape[1] as i32,
        dst.layout.shape[2] as i32,
        dst.layout.shape[3] as i32,
    );
    let (nb0, nb1, nb2, nb3) = (
        dst.layout.stride[0] as u64,
        dst.layout.stride[1] as u64,
        dst.layout.stride[2] as u64,
        dst.layout.stride[3] as u64,
    );

    let oa = sa.offset as u64;
    let ob = sb.offset as u64;
    let oc = dst.offset as u64;

    // Grid over the *dst* dims: feml-core broadcasts symmetrically (max of
    // both sources), so dst can be larger than src0 in any dim. Each source
    // is indexed with the block index modulo its own extent.
    let nth: u32 = 256.min(ne0.max(1) as u32);
    let cfg = LaunchConfig {
        grid_dim: (ne1.max(1) as u32, ne2.max(1) as u32, ne3.max(1) as u32),
        block_dim: (nth, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: slices are live for the whole launch; the kernel bounds-checks
    // element indices against the shapes above; identical-range aliasing is
    // read-before-write within a thread.
    unsafe {
        inner
            .stream
            .launch_builder(func)
            .arg(a)
            .arg(&oa)
            .arg(b)
            .arg(&ob)
            .arg(c)
            .arg(&oc)
            .arg(&ne00)
            .arg(&ne01)
            .arg(&ne02)
            .arg(&ne03)
            .arg(&nb00)
            .arg(&nb01)
            .arg(&nb02)
            .arg(&nb03)
            .arg(&ne10)
            .arg(&ne11)
            .arg(&ne12)
            .arg(&ne13)
            .arg(&nb10)
            .arg(&nb11)
            .arg(&nb12)
            .arg(&nb13)
            .arg(&ne0)
            .arg(&ne1)
            .arg(&ne2)
            .arg(&ne3)
            .arg(&nb0)
            .arg(&nb1)
            .arg(&nb2)
            .arg(&nb3)
            .launch(cfg)
            .map_err(cuda_err)?;
    }
    Ok(())
}
