//! CUDA mul_mat kernel invocation (naive f32, ggml semantics).

use cudarc::driver::{LaunchConfig, PushKernelArg};

use feml_core::error::{Error, Result};
use feml_core::plan::{GraphPlan, PlanTensor};

use crate::backend::BackendInner;
use crate::device::cuda_err;

pub(crate) fn mul_mat(
    inner: &BackendInner,
    plan: &GraphPlan,
    sa: &PlanTensor,
    sb: &PlanTensor,
    dst: &PlanTensor,
) -> Result<()> {
    let func = inner
        .funcs
        .get("kernel_mul_mat")
        .ok_or_else(|| Error::msg("mul_mat: kernel not loaded"))?;

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

    if ranges_overlap(plan, sa, dst) || ranges_overlap(plan, sb, dst) {
        return Err(Error::msg("CUDA: mul_mat dst aliases a source"));
    }

    let h_a = plan.buffers[sa.buffer].handle;
    let h_b = plan.buffers[sb.buffer].handle;
    let h_c = plan.buffers[dst.buffer].handle;
    let a = inner
        .buffers
        .get(&h_a)
        .ok_or_else(|| Error::msg(format!("mul_mat: unknown buffer {h_a}")))?;
    let b = inner
        .buffers
        .get(&h_b)
        .ok_or_else(|| Error::msg(format!("mul_mat: unknown buffer {h_b}")))?;
    let c = inner
        .buffers
        .get(&h_c)
        .ok_or_else(|| Error::msg(format!("mul_mat: unknown buffer {h_c}")))?;

    let oa = sa.offset as u64;
    let na1 = sa.layout.stride[1] as u64;
    let ob = sb.offset as u64;
    let nb1 = sb.layout.stride[1] as u64;
    let oc = dst.offset as u64;
    let nc1 = dst.layout.stride[1] as u64;
    let k = sa.layout.shape[0] as i32;
    let m = sa.layout.shape[1] as i32;
    let n = sb.layout.shape[1] as i32;

    let total = (m as usize) * (n as usize);
    if total > i32::MAX as usize {
        return Err(Error::msg("mul_mat: M*N exceeds kernel int range"));
    }
    let cfg = LaunchConfig {
        grid_dim: ((total as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: slices are live for the whole launch; the kernel indexes
    // exactly the [K, M] x [K, N] -> [M, N] ranges encoded by the strides
    // below; dst does not alias a source.
    unsafe {
        inner
            .stream
            .launch_builder(func)
            .arg(a)
            .arg(&oa)
            .arg(&na1)
            .arg(b)
            .arg(&ob)
            .arg(&nb1)
            .arg(c)
            .arg(&oc)
            .arg(&nc1)
            .arg(&k)
            .arg(&m)
            .arg(&n)
            .launch(cfg)
            .map_err(cuda_err)?;
    }
    Ok(())
}
