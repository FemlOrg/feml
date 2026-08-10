//! CPU execution stream.
//!
//! Buffers are per-buffer `Mutex<Vec<u8>>` inside a coarse map lock, so the
//! backend is `Send + Sync` and kernels lock their operands independently.
//! M1 executes plans single-threaded; the thread pool hooks are the per-op
//! work partitioning (see DESIGN.md §8).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use feml_core::backend::{Backend, BufferHandle, BufferUsage};
use feml_core::dtype::DType;
use feml_core::error::{Error, Result};
use feml_core::op::Op;
use feml_core::plan::{GraphPlan, PlanTensor};

pub(crate) type CpuBuffer = Arc<Mutex<Vec<u8>>>;

type LockedGuards<'a> = (usize, [usize; 3], [Option<std::sync::MutexGuard<'a, Vec<u8>>>; 3]);

pub(crate) struct CpuInner {
    pub(crate) buffers: HashMap<BufferHandle, CpuBuffer>,
    next_handle: u32,
}

#[derive(Clone)]
pub struct CpuBackend {
    inner: Arc<Mutex<CpuInner>>,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self { inner: Arc::new(Mutex::new(CpuInner { buffers: HashMap::new(), next_handle: 1 })) }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuBackend {
    fn ranges_overlap(plan: &GraphPlan, a: &PlanTensor, b: &PlanTensor) -> bool {
        if plan.buffers[a.buffer].handle != plan.buffers[b.buffer].handle {
            return false;
        }
        let a_end = a.offset + a.nbytes();
        let b_end = b.offset + b.nbytes();
        a.offset < b_end && b.offset < a_end
    }

    /// Clone the Arc to a buffer, dropping the map lock before the caller
    /// locks the buffer itself (kernels never hold the global map lock).
    fn buffer(&self, handle: BufferHandle, op: &str) -> Result<CpuBuffer> {
        let inner = self.inner.lock().unwrap();
        inner
            .buffers
            .get(&handle)
            .cloned()
            .ok_or_else(|| Error::msg(format!("{op}: unknown buffer {handle}")))
    }

    /// Lock the three operand buffers at most once per unique buffer
    /// (inplace-aliased operands share one lock), in deterministic
    /// pointer order. Uses fixed stack arrays: no allocation.
    fn lock_operands<'a>(a: &'a CpuBuffer, b: &'a CpuBuffer, c: &'a CpuBuffer) -> LockedGuards<'a> {
        let pa = Arc::as_ptr(a) as usize;
        let pb = Arc::as_ptr(b) as usize;
        let pc = Arc::as_ptr(c) as usize;
        let mut sorted = [pa, pb, pc];
        sorted.sort_unstable();
        let mut locked_p: [usize; 3] = [pa, pb, pc];
        let mut locked_g: [Option<std::sync::MutexGuard<'a, Vec<u8>>>; 3] = [None, None, None];
        let mut n = 0;
        'outer: for p in sorted {
            if locked_p[..n].contains(&p) {
                continue 'outer;
            }
            let arc: &'a CpuBuffer = if p == pa {
                a
            } else if p == pb {
                b
            } else {
                c
            };
            locked_p[n] = p;
            locked_g[n] = Some(arc.lock().unwrap());
            n += 1;
        }
        (n, locked_p, locked_g)
    }

    fn pos(locked_p: &[usize; 3], n: usize, p: usize) -> usize {
        locked_p[..n].iter().position(|&x| x == p).expect("operand handle present")
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "CPU"
    }

    fn create_buffer(&self, size: usize, usage: BufferUsage) -> Result<BufferHandle> {
        let mut inner = self.inner.lock().unwrap();
        let data = vec![0u8; size];
        let handle = BufferHandle::new(inner.next_handle);
        inner.next_handle += 1;
        inner.buffers.insert(handle, Arc::new(Mutex::new(data)));
        let _ = usage;
        Ok(handle)
    }

    fn release_buffer(&self, handle: BufferHandle) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        inner
            .buffers
            .remove(&handle)
            .map(|_| ())
            .ok_or_else(|| Error::msg(format!("release_buffer: unknown buffer {handle}")))
    }

    fn write(&self, handle: BufferHandle, offset: usize, data: &[u8]) -> Result<()> {
        let buffer = self.buffer(handle, "write")?;
        let mut buf = buffer.lock().unwrap();
        if offset + data.len() > buf.len() {
            return Err(Error::msg(format!(
                "write: {handle} overflow: offset {offset} + {} > buffer {}",
                data.len(),
                buf.len()
            )));
        }
        buf[offset..offset + data.len()].copy_from_slice(data);
        Ok(())
    }

    fn read(&self, handle: BufferHandle, offset: usize, out: &mut [u8]) -> Result<()> {
        let buffer = self.buffer(handle, "read")?;
        let buf = buffer.lock().unwrap();
        if offset + out.len() > buf.len() {
            return Err(Error::msg(format!(
                "read: {handle} overflow: offset {offset} + {} > buffer {}",
                out.len(),
                buf.len()
            )));
        }
        out.copy_from_slice(&buf[offset..offset + out.len()]);
        Ok(())
    }

    fn fill(&self, handle: BufferHandle, offset: usize, value: u8, len: usize) -> Result<()> {
        let buffer = self.buffer(handle, "fill")?;
        let mut buf = buffer.lock().unwrap();
        if offset + len > buf.len() {
            return Err(Error::msg(format!(
                "fill: {handle} overflow: offset {offset} + {len} > buffer {}",
                buf.len()
            )));
        }
        buf[offset..offset + len].fill(value);
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    fn graph_compute(&self, plan: &GraphPlan) -> Result<()> {
        for &ti in &plan.nodes {
            let dst = &plan.tensors[ti];
            let sa = &plan.tensors[dst.srcs[0]];
            let sb = &plan.tensors[dst.srcs[1]];
            let (a, b, c) = {
                let inner = self.inner.lock().unwrap();
                let get = |t: &PlanTensor| -> Result<CpuBuffer> {
                    let h = plan.buffers[t.buffer].handle;
                    inner
                        .buffers
                        .get(&h)
                        .cloned()
                        .ok_or_else(|| Error::msg(format!("graph_compute: unknown buffer {h}")))
                };
                (get(sa)?, get(sb)?, get(dst)?)
            };
            let (n, locked_p, locked_g) = Self::lock_operands(&a, &b, &c);
            let pa = Self::pos(&locked_p, n, Arc::as_ptr(&a) as usize);
            let pb = Self::pos(&locked_p, n, Arc::as_ptr(&b) as usize);
            let pc = Self::pos(&locked_p, n, Arc::as_ptr(&c) as usize);
            let guard = |i: usize| -> &std::sync::MutexGuard<'_, Vec<u8>> {
                locked_g[i].as_ref().expect("locked operand present")
            };
            match dst.op {
                Op::Mul | Op::Add => {
                    if Self::ranges_overlap(plan, sa, dst)
                        && (sa.offset != dst.offset || sa.nbytes() != dst.nbytes())
                    {
                        return Err(Error::msg("CPU: partial-overlap alias not supported"));
                    }
                    if Self::ranges_overlap(plan, sb, dst)
                        && (sb.offset != dst.offset || sb.nbytes() != dst.nbytes())
                    {
                        return Err(Error::msg("CPU: partial-overlap alias not supported"));
                    }
                    let a_op = super::ops::elementwise::Operand {
                        ptr: guard(pa).as_ptr(),
                        len: guard(pa).len(),
                        offset: sa.offset,
                        layout: &sa.layout,
                    };
                    let b_op = super::ops::elementwise::Operand {
                        ptr: guard(pb).as_ptr(),
                        len: guard(pb).len(),
                        offset: sb.offset,
                        layout: &sb.layout,
                    };
                    let c_op = super::ops::elementwise::OutOperand {
                        ptr: guard(pc).as_ptr() as *mut u8,
                        len: guard(pc).len(),
                        offset: dst.offset,
                        layout: &dst.layout,
                    };
                    // SAFETY: guards outlive the call; kernels bounds-check all
                    // offsets; identical-range aliasing is read-before-write safe.
                    unsafe {
                        super::ops::elementwise::elementwise_binary(
                            &a_op,
                            &b_op,
                            &c_op,
                            dst.op == Op::Mul,
                        )?;
                    }
                }
                Op::MulMat => {
                    if Self::ranges_overlap(plan, sa, dst) || Self::ranges_overlap(plan, sb, dst) {
                        return Err(Error::msg("CPU: mul_mat dst aliases a source"));
                    }
                    let a_op = super::ops::elementwise::Operand {
                        ptr: guard(pa).as_ptr(),
                        len: guard(pa).len(),
                        offset: sa.offset,
                        layout: &sa.layout,
                    };
                    let b_op = super::ops::elementwise::Operand {
                        ptr: guard(pb).as_ptr(),
                        len: guard(pb).len(),
                        offset: sb.offset,
                        layout: &sb.layout,
                    };
                    let c_op = super::ops::elementwise::OutOperand {
                        ptr: guard(pc).as_ptr() as *mut u8,
                        len: guard(pc).len(),
                        offset: dst.offset,
                        layout: &dst.layout,
                    };
                    // SAFETY: guards outlive the call; ranges are disjoint and
                    // bounds-checked inside the kernel.
                    unsafe {
                        super::ops::mul_mat::mul_mat(&a_op, &b_op, &c_op)?;
                    }
                }
                _ => return Err(Error::msg(format!("CPU: op {} not implemented", dst.op.name()))),
            }
        }
        Ok(())
    }

    fn supports_op(&self, op: Op, src_dtypes: &[DType]) -> bool {
        matches!(op, Op::Mul | Op::Add | Op::MulMat) && src_dtypes.iter().all(|d| *d == DType::F32)
    }
}
