//! Compiled graph plan: the immutable, `Send + Sync` execution artifact.
//!
//! `GraphBuilder::compile` produces a `GraphPlan`: every tensor gets a fixed
//! (buffer, offset) assignment; execution walks `nodes` in topological order
//! with zero allocation.

use crate::backend::{BufferHandle, BufferUsage};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::op::Op;
use crate::tensor::{TensorFlags, TensorId};

#[derive(Clone, Debug)]
pub struct PlanTensor {
    pub id: TensorId,
    pub op: Op,
    /// Indices into `GraphPlan::tensors` (sources of this op).
    pub srcs: Vec<usize>,
    /// Index into `GraphPlan::buffers`.
    pub buffer: usize,
    /// Byte offset within the buffer.
    pub offset: usize,
    pub dtype: DType,
    pub layout: Layout,
    pub flags: TensorFlags,
}

#[derive(Debug)]
pub struct PlanBuffer {
    pub handle: BufferHandle,
    pub size: usize,
    pub usage: BufferUsage,
}

#[derive(Debug)]
pub struct GraphPlan {
    /// Stable fingerprint of the graph structure (plan caching, M2).
    pub uid: u64,
    /// All reachable tensors (leafs first, then nodes in topological order).
    pub tensors: Vec<PlanTensor>,
    /// Indices into `tensors`, in execution order.
    pub nodes: Vec<usize>,
    /// Backend-owned buffers backing the plan.
    pub buffers: Vec<PlanBuffer>,
}

impl PlanTensor {
    pub fn nbytes(&self) -> usize {
        self.layout.nbytes(self.dtype)
    }
}

impl GraphPlan {
    pub fn find(&self, id: TensorId) -> Option<&PlanTensor> {
        self.tensors.iter().find(|t| t.id == id)
    }

    pub fn tensors(&self) -> &[PlanTensor] {
        &self.tensors
    }

    pub fn nodes(&self) -> impl Iterator<Item = &PlanTensor> {
        self.nodes.iter().map(|&i| &self.tensors[i])
    }

    /// Free the backend buffers owned by this plan. The plan becomes unusable.
    pub fn release(&self, backend: &dyn crate::backend::Backend) -> Result<()> {
        for b in &self.buffers {
            backend.release_buffer(b.handle)?;
        }
        Ok(())
    }

    /// Write tensor data through the backend. `data.len()` must equal the
    /// tensor's `nbytes()`.
    pub fn write_tensor(
        &self,
        backend: &dyn crate::backend::Backend,
        id: TensorId,
        data: &[u8],
    ) -> Result<()> {
        let t = self
            .find(id)
            .ok_or_else(|| Error::msg(format!("write_tensor: unknown tensor {id}")))?;
        if data.len() != t.nbytes() {
            return Err(Error::shape(format!(
                "write_tensor: tensor {id} needs {} bytes, got {}",
                t.nbytes(),
                data.len()
            )));
        }
        backend.write(self.buffers[t.buffer].handle, t.offset, data)
    }

    /// Read tensor data through the backend. `out.len()` must equal the
    /// tensor's `nbytes()`.
    pub fn read_tensor(
        &self,
        backend: &dyn crate::backend::Backend,
        id: TensorId,
        out: &mut [u8],
    ) -> Result<()> {
        let t =
            self.find(id).ok_or_else(|| Error::msg(format!("read_tensor: unknown tensor {id}")))?;
        if out.len() != t.nbytes() {
            return Err(Error::shape(format!(
                "read_tensor: tensor {id} needs {} bytes, got {}",
                t.nbytes(),
                out.len()
            )));
        }
        backend.read(self.buffers[t.buffer].handle, t.offset, out)
    }
}
