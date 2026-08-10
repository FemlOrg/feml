//! Graph-aware memory planner (ggml-alloc algorithm, Rust port).
//!
//! Mirrors `ggml_gallocr_alloc_graph`:
//! 1. Pre-allocate INPUT/WEIGHT leafs (their slots are reserved; inplace ops
//!    may share an input slot, which consumes the input data for that run).
//! 2. Count children and views per tensor.
//! 3. For each node in topological order: ensure sources are allocated,
//!    allocate the node (views share their source; inplace ops reuse a sole
//!    parent with identical layout), then decrement sources' children and
//!    free tensors whose last use has passed.
//!
//! Buffer kinds: WEIGHT tensors go to the weights buffer (never freed);
//! everything else goes to the compute buffer. OUTPUT tensors are never freed.

use crate::backend::BufferUsage;
use crate::error::{Error, Result};
use crate::op::Op;
use crate::tensor::{TensorFlags, TensorId, TensorMeta};
use std::collections::HashMap;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub(crate) enum BufferKind {
    Weights,
    Compute,
}

/// All buffer kinds in index order. Adding a kind (e.g. a persistent KV-cache
/// class) is one variant here, one entry in this array, and one arm in
/// `kind_of`/`usage_of`.
pub(crate) const BUFFER_KINDS: [BufferKind; 2] = [BufferKind::Weights, BufferKind::Compute];

#[derive(Debug, Copy, Clone)]
pub(crate) struct Assignment {
    pub kind: BufferKind,
    pub offset: usize,
}

pub(crate) struct PlannerOutput {
    pub assignments: HashMap<TensorId, Assignment>,
    pub weights_size: usize,
    pub compute_size: usize,
}

pub(crate) fn usage_of(kind: BufferKind) -> BufferUsage {
    match kind {
        BufferKind::Weights => BufferUsage::Weights,
        BufferKind::Compute => BufferUsage::Compute,
    }
}

/// Byte alignment for all planner allocations (ggml uses 16; 32 keeps SIMD
/// kernels and quantized loads well-aligned).
const ALIGNMENT: usize = 32;

#[derive(Debug)]
struct Talloc {
    used: usize,
    free: Vec<(usize, usize)>,
}

impl Talloc {
    fn new() -> Self {
        Self { used: 0, free: Vec::new() }
    }

    fn alloc(&mut self, size: usize) -> usize {
        let size = size.next_multiple_of(ALIGNMENT);
        if let Some((idx, &(offset, free_size))) =
            self.free.iter().enumerate().find(|(_, (_, s))| *s >= size)
        {
            self.free.remove(idx);
            if free_size > size {
                self.free.push((offset + size, free_size - size));
            }
            return offset;
        }
        let offset = self.used.next_multiple_of(ALIGNMENT);
        self.used = offset + size;
        offset
    }

    fn free(&mut self, offset: usize, size: usize) {
        self.free.push((offset, size.next_multiple_of(ALIGNMENT)));
        self.free.sort_unstable();
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(self.free.len());
        for (off, sz) in self.free.drain(..) {
            if let Some(last) = merged.last_mut() {
                if last.0 + last.1 == off {
                    last.1 += sz;
                    continue;
                }
            }
            merged.push((off, sz));
        }
        self.free = merged;
    }

    fn peak(&self) -> usize {
        self.used
    }
}

fn kind_of(meta: &TensorMeta) -> BufferKind {
    if meta.flags.has(TensorFlags::WEIGHT) { BufferKind::Weights } else { BufferKind::Compute }
}

fn can_reuse(node: &TensorMeta, parent: &TensorMeta, children: usize, views: usize) -> bool {
    Op::can_inplace(&node.op)
        && children == 1
        && views == 0
        && !parent.flags.has(TensorFlags::OUTPUT)
        && !parent.flags.has(TensorFlags::WEIGHT)
        && node.dtype == parent.dtype
        && node.layout.shape == parent.layout.shape
        && node.layout.stride == parent.layout.stride
}

fn kind_index(kind: BufferKind) -> usize {
    BUFFER_KINDS.iter().position(|&k| k == kind).expect("kind in BUFFER_KINDS")
}

fn tensor_at<'a>(
    tensors: &'a [TensorMeta],
    by_id: &HashMap<TensorId, usize>,
    id: TensorId,
) -> &'a TensorMeta {
    &tensors[by_id[&id]]
}

fn allocate(
    id: TensorId,
    tensors: &[TensorMeta],
    by_id: &HashMap<TensorId, usize>,
    tallocs: &mut [Talloc],
    assigned: &mut HashMap<TensorId, Assignment>,
    allocated: &mut HashMap<TensorId, bool>,
) -> Result<()> {
    if allocated[&id] {
        return Ok(());
    }
    let t = tensor_at(tensors, by_id, id);
    if let Some(src) = t.view_src {
        allocate(src, tensors, by_id, tallocs, assigned, allocated)?;
        let a = *assigned.get(&src).unwrap();
        assigned.insert(id, Assignment { kind: a.kind, offset: a.offset + t.layout.offset });
        allocated.insert(id, true);
        return Ok(());
    }
    let kind = kind_of(t);
    let offset = tallocs[kind_index(kind)].alloc(t.nbytes());
    assigned.insert(id, Assignment { kind, offset });
    allocated.insert(id, true);
    Ok(())
}

pub(crate) fn plan(tensors: &[TensorMeta], order: &[usize]) -> Result<PlannerOutput> {
    let by_id: HashMap<TensorId, usize> =
        tensors.iter().enumerate().map(|(i, t)| (t.id, i)).collect();

    let mut children: HashMap<TensorId, usize> = tensors.iter().map(|t| (t.id, 0)).collect();
    let mut views: HashMap<TensorId, usize> = tensors.iter().map(|t| (t.id, 0)).collect();
    for t in tensors {
        if let Some(src) = t.view_src {
            *views.get_mut(&src).unwrap() += 1;
        }
        for src in t.srcs().iter().flatten() {
            *children.get_mut(src).unwrap() += 1;
        }
    }

    let mut tallocs: Vec<Talloc> = BUFFER_KINDS.iter().map(|_| Talloc::new()).collect();
    let mut assigned: HashMap<TensorId, Assignment> = HashMap::new();
    let mut allocated: HashMap<TensorId, bool> = tensors.iter().map(|t| (t.id, false)).collect();
    let mut live_children = children.clone();
    let mut live_views = views.clone();

    for t in tensors {
        if t.flags.has(TensorFlags::INPUT) || t.flags.has(TensorFlags::WEIGHT) {
            allocate(t.id, tensors, &by_id, &mut tallocs, &mut assigned, &mut allocated)?;
        }
    }

    for &idx in order {
        let node = &tensors[idx];

        if let Some(src) = node.view_src {
            allocate(src, tensors, &by_id, &mut tallocs, &mut assigned, &mut allocated)?;
            let a = *assigned
                .get(&src)
                .ok_or_else(|| Error::msg("planner: view source not assigned"))?;
            assigned.insert(
                node.id,
                Assignment { kind: a.kind, offset: a.offset + node.layout.offset },
            );
            allocated.insert(node.id, true);
        } else {
            for src in node.srcs().iter().flatten() {
                if !allocated[src] {
                    allocate(*src, tensors, &by_id, &mut tallocs, &mut assigned, &mut allocated)?;
                }
            }
            let mut reused = false;
            if Op::can_inplace(&node.op) {
                for src in node.srcs().iter().flatten() {
                    let parent = tensor_at(tensors, &by_id, *src);
                    if allocated[src]
                        && live_children[src] == 1
                        && live_views[src] == 0
                        && can_reuse(node, parent, live_children[src], live_views[src])
                    {
                        let a = *assigned.get(src).unwrap();
                        assigned.insert(node.id, a);
                        allocated.insert(node.id, true);
                        allocated.insert(*src, false);
                        reused = true;
                        break;
                    }
                }
            }
            if !reused {
                let kind = kind_of(node);
                let offset = tallocs[kind_index(kind)].alloc(node.nbytes());
                assigned.insert(node.id, Assignment { kind, offset });
                allocated.insert(node.id, true);
            }
        }

        for src in node.srcs().iter().flatten() {
            *live_children.get_mut(src).unwrap() -= 1;
            let parent = tensor_at(tensors, &by_id, *src);
            let never_freed = parent.flags.has(TensorFlags::INPUT)
                || parent.flags.has(TensorFlags::WEIGHT)
                || parent.flags.has(TensorFlags::OUTPUT);
            if live_children[src] == 0 && live_views[src] == 0 && !never_freed {
                if allocated[src] {
                    let a = *assigned.get(src).unwrap();
                    tallocs[kind_index(a.kind)].free(a.offset, parent.nbytes());
                    allocated.insert(*src, false);
                } else if let Some(view_src) = parent.view_src {
                    *live_views.get_mut(&view_src).unwrap() -= 1;
                    if live_views[&view_src] == 0
                        && live_children[&view_src] == 0
                        && !tensor_at(tensors, &by_id, view_src).flags.has(TensorFlags::OUTPUT)
                        && allocated[&view_src]
                    {
                        let a = *assigned.get(&view_src).unwrap();
                        let t = tensor_at(tensors, &by_id, view_src);
                        tallocs[kind_index(a.kind)].free(a.offset, t.nbytes());
                        allocated.insert(view_src, false);
                    }
                }
            }
        }
    }

    Ok(PlannerOutput {
        assignments: assigned,
        weights_size: tallocs[kind_index(BufferKind::Weights)].peak(),
        compute_size: tallocs[kind_index(BufferKind::Compute)].peak(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::shape;
    use crate::shape::Shape;

    fn t(name: &str, dtype: DType, shape: Shape, flags: u8) -> TensorMeta {
        let mut m = TensorMeta::new(name, dtype, shape).unwrap();
        m.flags = TensorFlags::new().with(flags);
        m
    }

    fn mul_out(name: &str, a: TensorMeta, b: TensorMeta) -> TensorMeta {
        let mut m = TensorMeta::new(name, a.dtype, a.layout.shape).unwrap();
        m.op = Op::Mul;
        m = m.set_src(a.id).set_src(b.id);
        m
    }

    #[test]
    fn diamond_shared_source() {
        let a = t("a", DType::F32, shape![4], TensorFlags::INPUT);
        let b = t("b", DType::F32, shape![4], TensorFlags::INPUT);
        let (a_id, b_id) = (a.id, b.id);
        let c = mul_out("c", a.clone(), b.clone());
        let (c_id,) = (c.id,);
        let d = mul_out("d", a.clone(), c.clone());
        let d_id = d.id;
        let tensors = vec![a, b, c, d];
        let out = plan(&tensors, &[2, 3]).unwrap();
        let (a_off, b_off) = (
            out.assignments.get(&a_id).unwrap().offset,
            out.assignments.get(&b_id).unwrap().offset,
        );
        let (c_off, d_off) = (
            out.assignments.get(&c_id).unwrap().offset,
            out.assignments.get(&d_id).unwrap().offset,
        );
        assert_ne!(a_off, b_off);
        assert_eq!(b_off, c_off, "c reuses single-child b");
        assert_eq!(
            a_off, d_off,
            "d reuses a: ggml uses the live (decremented) child count; when d runs, a has 1 child left"
        );
        assert_eq!(out.compute_size, 64, "two 32-aligned slots only (a, b)");
    }

    #[test]
    fn inplace_op_reuses_sole_parent() {
        let x = t("x", DType::F32, shape![8], TensorFlags::INPUT);
        let mut s = TensorMeta::new("s", DType::F32, shape![8]).unwrap();
        s.op = Op::Silu;
        s = s.set_src(x.id);
        let (x_id, s_id) = (x.id, s.id);
        let tensors = vec![x, s];
        let out = plan(&tensors, &[1]).unwrap();
        assert_eq!(
            out.assignments.get(&x_id).unwrap().offset,
            out.assignments.get(&s_id).unwrap().offset,
            "silu must reuse its input slot"
        );
        assert_eq!(out.compute_size, 32, "one slot only");
    }

    #[test]
    fn inplace_reused_parent_not_freed() {
        let x = t("x", DType::F32, shape![8], TensorFlags::INPUT);
        let mut s = TensorMeta::new("s", DType::F32, shape![8]).unwrap();
        s.op = Op::Silu;
        s = s.set_src(x.id);
        let mut o = TensorMeta::new("o", DType::F32, shape![8]).unwrap();
        o.op = Op::Copy;
        o.flags = o.flags.with(TensorFlags::OUTPUT);
        o = o.set_src(s.id);
        let (x_id, s_id, o_id) = (x.id, s.id, o.id);
        let tensors = vec![x, s, o];
        let out = plan(&tensors, &[1, 2]).unwrap();
        let x_a = *out.assignments.get(&x_id).unwrap();
        let o_a = *out.assignments.get(&o_id).unwrap();
        assert_eq!(x_a.offset, out.assignments.get(&s_id).unwrap().offset);
        assert_ne!(o_a.offset, x_a.offset, "output must get its own slot");
    }

    #[test]
    fn non_inplace_op_gets_own_slot() {
        let w = t("w", DType::F32, shape![4, 4], TensorFlags::WEIGHT);
        let x = t("x", DType::F32, shape![4, 1], TensorFlags::INPUT);
        let mut m = TensorMeta::new("m", DType::F32, shape![4, 1]).unwrap();
        m.op = Op::MulMat;
        m = m.set_src(w.id).set_src(x.id);
        let (w_id, x_id, m_id) = (w.id, x.id, m.id);
        let tensors = vec![w, x, m];
        let out = plan(&tensors, &[2]).unwrap();
        let w_a = out.assignments.get(&w_id).unwrap();
        let x_a = out.assignments.get(&x_id).unwrap();
        let m_a = out.assignments.get(&m_id).unwrap();
        assert_eq!(w_a.kind, BufferKind::Weights);
        assert_eq!(x_a.kind, BufferKind::Compute);
        assert_ne!(m_a.offset, x_a.offset);
    }

    #[test]
    fn view_shares_source_buffer() {
        let x = t("x", DType::F32, shape![8], TensorFlags::INPUT);
        let mut v = TensorMeta::new("v", DType::F32, shape![4]).unwrap();
        v.view_src = Some(x.id);
        v.layout.offset = 16;
        let mut o = TensorMeta::new("o", DType::F32, shape![4]).unwrap();
        o.op = Op::Copy;
        o = o.set_src(v.id);
        let (x_id, v_id) = (x.id, v.id);
        let tensors = vec![x, v, o];
        let out = plan(&tensors, &[2]).unwrap();
        let x_a = out.assignments.get(&x_id).unwrap();
        let v_a = out.assignments.get(&v_id).unwrap();
        assert_eq!(v_a.kind, x_a.kind);
        assert_eq!(v_a.offset, x_a.offset + 16);
    }

    #[test]
    fn compute_tensor_reuses_freed_slot() {
        let a = t("a", DType::F32, shape![16], TensorFlags::INPUT);
        let b = t("b", DType::F32, shape![16], TensorFlags::INPUT);
        let (a_id, b_id) = (a.id, b.id);
        let n1 = mul_out("n1", a.clone(), b.clone());
        let n2 = mul_out("n2", n1.clone(), b.clone());
        let n2_id = n2.id;
        let tensors = vec![a, b, n1, n2];
        let out = plan(&tensors, &[2, 3]).unwrap();
        let a_off = out.assignments.get(&a_id).unwrap().offset;
        let b_off = out.assignments.get(&b_id).unwrap().offset;
        let n2_off = out.assignments.get(&n2_id).unwrap().offset;
        assert_eq!(out.compute_size, 2 * 64, "a + b slots only");
        assert!(n2_off == a_off || n2_off == b_off);
    }
}
