//! Graph builder: the build-phase API.
//!
//! Tensors are `TensorId` handles into the builder's arena. All shape/dtype
//! validation happens here and returns `Result`. `compile` runs topological
//! sorting + memory planning and produces the immutable `GraphPlan`.

use crate::backend::Backend;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::op::{Op, RopeParams, broadcast_shape};
use crate::plan::{GraphPlan, PlanBuffer, PlanTensor};
use crate::planner::{self, Assignment, BufferKind, PlannerOutput};
use crate::shape::Shape;
use crate::tensor::{TensorFlags, TensorId, TensorMeta};
use std::collections::{HashMap, HashSet};

/// Minimum byte alignment for view offsets (matches the f32 element size;
/// keeps raw `*const f32` kernel loads well-formed on all targets).
const ELEMENT_ALIGNMENT: usize = 4;

pub struct GraphBuilder {
    tensors: Vec<TensorMeta>,
    by_id: HashMap<TensorId, usize>,
    outputs: Vec<TensorId>,
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self { tensors: Vec::new(), by_id: HashMap::new(), outputs: Vec::new() }
    }

    fn push(&mut self, meta: TensorMeta) -> TensorId {
        let id = meta.id;
        self.by_id.insert(id, self.tensors.len());
        self.tensors.push(meta);
        id
    }

    pub fn tensor(&self, id: TensorId) -> Option<&TensorMeta> {
        self.by_id.get(&id).map(|&i| &self.tensors[i])
    }

    pub fn tensors(&self) -> &[TensorMeta] {
        &self.tensors
    }

    pub fn outputs(&self) -> &[TensorId] {
        &self.outputs
    }

    pub fn mark_output(&mut self, tensor: TensorId) -> Result<()> {
        let t = self.tensor(tensor).ok_or_else(|| Error::msg("mark_output: unknown tensor"))?;
        if !t.flags.has(TensorFlags::OUTPUT) {
            let idx = self.by_id[&tensor];
            self.tensors[idx].flags = self.tensors[idx].flags.with(TensorFlags::OUTPUT);
        }
        self.outputs.push(tensor);
        Ok(())
    }

    pub fn input(
        &mut self,
        name: impl Into<String>,
        dtype: DType,
        shape: Shape,
    ) -> Result<TensorId> {
        let mut meta = TensorMeta::new(name, dtype, shape)?;
        meta.flags = meta.flags.with(TensorFlags::INPUT);
        Ok(self.push(meta))
    }

    pub fn weight(
        &mut self,
        name: impl Into<String>,
        dtype: DType,
        shape: Shape,
    ) -> Result<TensorId> {
        let mut meta = TensorMeta::new(name, dtype, shape)?;
        meta.flags = meta.flags.with(TensorFlags::WEIGHT);
        Ok(self.push(meta))
    }

    pub fn view(&mut self, src: TensorId, offset: usize, shape: Shape) -> Result<TensorId> {
        let src_meta = self.tensor(src).ok_or_else(|| Error::msg("view: unknown source"))?;
        let nbytes = src_meta.nbytes();
        let mut meta = TensorMeta::new("view", src_meta.dtype, shape)?;
        if !offset.is_multiple_of(ELEMENT_ALIGNMENT) {
            return Err(Error::shape(format!(
                "view: offset {offset} is not {ELEMENT_ALIGNMENT}-byte aligned"
            )));
        }
        if offset + meta.nbytes() > nbytes {
            return Err(Error::shape(format!(
                "view: offset {offset} + {} > source bytes {nbytes}",
                meta.nbytes()
            )));
        }
        meta.op = Op::View;
        meta.view_src = Some(src);
        meta.layout.offset = offset;
        Ok(self.push(meta))
    }

    pub fn mul(&mut self, a: TensorId, b: TensorId) -> Result<TensorId> {
        let (ma, mb) = self.pair(a, b, "mul")?;
        let shape = broadcast_shape(&ma.layout.shape, &mb.layout.shape)?;
        self.binary(Op::Mul, "mul", a, b, shape)
    }

    pub fn add(&mut self, a: TensorId, b: TensorId) -> Result<TensorId> {
        let (ma, mb) = self.pair(a, b, "add")?;
        let shape = broadcast_shape(&ma.layout.shape, &mb.layout.shape)?;
        self.binary(Op::Add, "add", a, b, shape)
    }

    pub fn mul_mat(&mut self, a: TensorId, b: TensorId) -> Result<TensorId> {
        let (ma, mb) = self.pair(a, b, "mul_mat")?;
        if ma.dtype != DType::F32 || mb.dtype != DType::F32 {
            return Err(Error::unsupported_dtype(ma.dtype, "mul_mat"));
        }
        if ma.layout.shape[0] != mb.layout.shape[0] {
            return Err(Error::shape(format!(
                "mul_mat: inner dim mismatch: {} vs {}",
                ma.layout.shape[0], mb.layout.shape[0]
            )));
        }
        if ma.layout.shape.rank() > 2 || mb.layout.shape.rank() > 2 {
            return Err(Error::shape("mul_mat: rank > 2 not supported yet"));
        }
        let shape = Shape::new(&[ma.layout.shape[1], mb.layout.shape[1]])?;
        self.binary(Op::MulMat, "mul_mat", a, b, shape)
    }

    pub fn rms_norm(&mut self, x: TensorId, eps: f32) -> Result<TensorId> {
        self.unary(Op::RmsNorm { eps }, "rms_norm", x)
    }

    pub fn silu(&mut self, x: TensorId) -> Result<TensorId> {
        self.unary(Op::Silu, "silu", x)
    }

    /// Softmax along `axis` (last dim when axis < 0, ggml convention).
    pub fn softmax(&mut self, x: TensorId, axis: i32) -> Result<TensorId> {
        self.unary(Op::Softmax { axis }, "softmax", x)
    }

    /// Rotary position embedding: x applied with positions `pos` (i32).
    pub fn rope(&mut self, x: TensorId, pos: TensorId, params: RopeParams) -> Result<TensorId> {
        let mx = self.tensor(x).ok_or_else(|| Error::msg("rope: unknown tensor"))?.clone();
        let mp = self.tensor(pos).ok_or_else(|| Error::msg("rope: unknown pos tensor"))?;
        if mp.dtype != DType::I32 {
            return Err(Error::shape("rope: pos must be i32"));
        }
        let mut meta = TensorMeta::new("rope", mx.dtype, mx.layout.shape)?;
        meta.op = Op::Rope {
            n_dims: params.n_dims,
            mode: params.mode,
            n_ctx: params.n_ctx,
            freq_base: params.freq_base,
            freq_scale: params.freq_scale,
        };
        meta = meta.set_src(x).set_src(pos);
        Ok(self.push(meta))
    }

    /// Embedding lookup: rows of `table` indexed by `idx` (i32), ggml layout.
    pub fn get_rows(&mut self, table: TensorId, idx: TensorId) -> Result<TensorId> {
        let mt = self.tensor(table).ok_or_else(|| Error::msg("get_rows: unknown table"))?.clone();
        let mi = self.tensor(idx).ok_or_else(|| Error::msg("get_rows: unknown idx tensor"))?;
        if mi.dtype != DType::I32 {
            return Err(Error::shape("get_rows: idx must be i32"));
        }
        // ggml: dst[ne00, ne10] where ne10 = idx element count
        let shape = Shape::new(&[mt.layout.shape[0], mi.layout.shape.len()])?;
        self.binary(Op::GetRows, "get_rows", table, idx, shape)
    }

    /// Concatenate two tensors along `axis`.
    pub fn concat(&mut self, a: TensorId, b: TensorId, axis: i32) -> Result<TensorId> {
        let (ma, mb) = self.pair(a, b, "concat")?;
        if ma.layout.shape.rank() != mb.layout.shape.rank() {
            return Err(Error::shape("concat: rank mismatch"));
        }
        let rank = ma.layout.shape.rank() as i32;
        let axis = if axis < 0 { rank + axis } else { axis };
        if axis < 0 || axis >= rank {
            return Err(Error::shape(format!("concat: axis {axis} out of range (rank {rank})")));
        }
        let mut dims: Vec<usize> = ma.layout.shape.iter().copied().collect();
        for i in 0..rank as usize {
            if i != axis as usize && ma.layout.shape[i] != mb.layout.shape[i] {
                return Err(Error::shape(format!(
                    "concat: dim {i} differs off-axis ({} vs {})",
                    ma.layout.shape[i], mb.layout.shape[i]
                )));
            }
        }
        dims[axis as usize] += mb.layout.shape[axis as usize];
        let shape = Shape::new(&dims)?;
        self.binary(Op::Concat { axis }, "concat", a, b, shape)
    }

    /// Copy `src` into a fresh tensor (same shape/dtype).
    pub fn copy(&mut self, src: TensorId) -> Result<TensorId> {
        self.unary(Op::Copy, "copy", src)
    }

    /// Grouped matrix multiply for MoE: `weights` [K, M, n_experts], `ids` [n_experts, N].
    pub fn mul_mat_id(
        &mut self,
        weights: TensorId,
        x: TensorId,
        ids: TensorId,
        n_selected: u32,
    ) -> Result<TensorId> {
        let mw =
            self.tensor(weights).ok_or_else(|| Error::msg("mul_mat_id: unknown weights"))?.clone();
        let mx = self.tensor(x).ok_or_else(|| Error::msg("mul_mat_id: unknown x"))?.clone();
        let mi = self.tensor(ids).ok_or_else(|| Error::msg("mul_mat_id: unknown ids"))?;
        if mw.dtype != DType::F32 || mx.dtype != DType::F32 {
            return Err(Error::unsupported_dtype(mw.dtype, "mul_mat_id"));
        }
        if mi.dtype != DType::I32 {
            return Err(Error::shape("mul_mat_id: ids must be i32"));
        }
        if mw.layout.shape[0] != mx.layout.shape[0] {
            return Err(Error::shape("mul_mat_id: inner dim mismatch"));
        }
        let n_experts = mw.layout.shape[2];
        if mi.layout.shape[0] != n_experts {
            return Err(Error::shape(format!(
                "mul_mat_id: ids rows {} != experts {n_experts}",
                mi.layout.shape[0]
            )));
        }
        if mi.layout.shape[1] != mx.layout.shape[1] {
            return Err(Error::shape("mul_mat_id: ids cols != x cols"));
        }
        let shape = Shape::new(&[mw.layout.shape[1], mx.layout.shape[1]])?;
        let mut meta = TensorMeta::new("mul_mat_id", DType::F32, shape)?;
        meta.op = Op::MulMatId { n_experts: n_experts as u32, n_selected };
        meta = meta.set_src(weights).set_src(x).set_src(ids);
        Ok(self.push(meta))
    }

    /// Causal mask: set upper triangle of the last two dims to -inf.
    pub fn diag_mask_inf(&mut self, x: TensorId, n_past: i32) -> Result<TensorId> {
        self.unary(Op::DiagMaskInf { n_past }, "diag_mask_inf", x)
    }

    /// Element-wise scaling.
    pub fn scale(&mut self, x: TensorId, alpha: f32) -> Result<TensorId> {
        self.unary(Op::Scale { alpha }, "scale", x)
    }

    fn pair(&self, a: TensorId, b: TensorId, op: &str) -> Result<(TensorMeta, TensorMeta)> {
        let ma = self.tensor(a).ok_or_else(|| Error::msg(format!("{op}: unknown tensor {a}")))?;
        let mb = self.tensor(b).ok_or_else(|| Error::msg(format!("{op}: unknown tensor {b}")))?;
        if ma.dtype != mb.dtype {
            return Err(Error::shape(format!(
                "{op}: dtype mismatch: {} vs {}",
                ma.dtype.name(),
                mb.dtype.name()
            )));
        }
        Ok((ma.clone(), mb.clone()))
    }

    fn binary(
        &mut self,
        op: Op,
        name: &str,
        a: TensorId,
        b: TensorId,
        shape: Shape,
    ) -> Result<TensorId> {
        let da = self.tensor(a).unwrap().dtype;
        let mut meta = TensorMeta::new(name, da, shape)?;
        meta.op = op;
        meta = meta.set_src(a).set_src(b);
        Ok(self.push(meta))
    }

    fn unary(&mut self, op: Op, name: &str, x: TensorId) -> Result<TensorId> {
        let mx = self
            .tensor(x)
            .ok_or_else(|| Error::msg(format!("{name}: unknown tensor {x}")))?
            .clone();
        let mut meta = TensorMeta::new(name, mx.dtype, mx.layout.shape)?;
        meta.op = op;
        meta = meta.set_src(x);
        Ok(self.push(meta))
    }

    /// Iterative DFS post-order of tensors reachable from `outputs`.
    fn topo_order(&self) -> Result<(Vec<usize>, HashSet<TensorId>)> {
        let mut order = Vec::new();
        let mut reachable: HashSet<TensorId> = HashSet::new();
        let mut stack: Vec<(usize, bool)> = Vec::new();
        for &out in &self.outputs {
            let idx = self
                .by_id
                .get(&out)
                .ok_or_else(|| Error::msg(format!("output tensor {out} not in graph")))?;
            stack.push((*idx, false));
        }
        while let Some((idx, expanded)) = stack.pop() {
            let t = &self.tensors[idx];
            if expanded {
                if t.op != Op::None {
                    order.push(idx);
                }
                continue;
            }
            if reachable.contains(&t.id) {
                continue;
            }
            reachable.insert(t.id);
            stack.push((idx, true));
            for src in t.srcs().iter().flatten() {
                if let Some(&src_idx) = self.by_id.get(src) {
                    stack.push((src_idx, false));
                }
            }
        }
        Ok((order, reachable))
    }

    fn uid(&self, order: &[usize]) -> u64 {
        let mut h: u64 = 1469598103934665603;
        for &i in order {
            let t = &self.tensors[i];
            h ^= t.id.as_u32() as u64;
            h = h.wrapping_mul(1099511628211);
            for src in t.srcs().iter().flatten() {
                h ^= src.as_u32() as u64;
                h = h.wrapping_mul(1099511628211);
            }
        }
        h
    }

    pub fn compile(&self, backend: &dyn Backend) -> Result<GraphPlan> {
        if self.outputs.is_empty() {
            return Err(Error::msg("compile: no outputs marked"));
        }
        let (order, reachable) = self.topo_order()?;
        let uid = self.uid(&order);

        // Fail fast: reject ops the backend cannot execute before allocating
        // any buffers.
        for &i in &order {
            let node = &self.tensors[i];
            let src_dtypes: Vec<DType> = node
                .srcs()
                .iter()
                .flatten()
                .map(|id| self.by_id[id])
                .map(|i| self.tensors[i].dtype)
                .collect();
            if !backend.supports_op(node.op, &src_dtypes) {
                return Err(Error::msg(format!(
                    "compile: backend '{}' does not support op {}",
                    backend.name(),
                    node.op.name()
                )));
            }
        }

        let reachable_meta: Vec<TensorMeta> =
            self.tensors.iter().filter(|t| reachable.contains(&t.id)).cloned().collect();
        let order_rel: Vec<usize> = order
            .iter()
            .map(|&i| {
                let id = self.tensors[i].id;
                reachable_meta
                    .iter()
                    .position(|t| t.id == id)
                    .expect("reachable tensor must be in reachable_meta")
            })
            .collect();
        let planner_out: PlannerOutput = planner::plan(&reachable_meta, &order_rel)?;

        let mut buffer_sizes: HashMap<BufferKind, usize> = HashMap::new();
        buffer_sizes.insert(BufferKind::Weights, planner_out.weights_size);
        buffer_sizes.insert(BufferKind::Compute, planner_out.compute_size);

        let mut buffers: Vec<PlanBuffer> = Vec::new();
        let mut buffer_idx: HashMap<BufferKind, usize> = HashMap::new();
        for kind in planner::BUFFER_KINDS {
            let size = buffer_sizes[&kind];
            if size == 0 {
                continue;
            }
            let handle = backend.create_buffer(size, planner::usage_of(kind))?;
            buffer_idx.insert(kind, buffers.len());
            buffers.push(PlanBuffer { handle, size, usage: planner::usage_of(kind) });
        }
        if buffers.is_empty() {
            return Err(Error::msg("compile: graph has no memory requirements"));
        }

        let mut plan_tensors = Vec::with_capacity(reachable_meta.len());
        let mut tensor_idx: HashMap<TensorId, usize> = HashMap::new();

        let mut leafs: Vec<TensorMeta> =
            reachable_meta.iter().filter(|t| t.op == Op::None).cloned().collect();
        leafs.sort_by_key(|t| if t.flags.has(TensorFlags::WEIGHT) { 0 } else { 1 });
        let nodes: Vec<TensorMeta> =
            reachable_meta.iter().filter(|t| t.op != Op::None).cloned().collect();

        for t in leafs.into_iter().chain(nodes) {
            let src_indices: Vec<usize> =
                t.srcs().iter().flatten().filter_map(|s| tensor_idx.get(s).copied()).collect();
            let assignment: &Assignment = planner_out
                .assignments
                .get(&t.id)
                .ok_or_else(|| Error::msg(format!("compile: tensor {} has no allocation", t.id)))?;
            tensor_idx.insert(t.id, plan_tensors.len());
            plan_tensors.push(PlanTensor {
                id: t.id,
                op: t.op,
                srcs: src_indices,
                buffer: buffer_idx[&assignment.kind],
                offset: assignment.offset,
                dtype: t.dtype,
                layout: t.layout,
                flags: t.flags,
            });
        }

        let node_indices: Vec<usize> =
            order.iter().map(|&i| tensor_idx[&self.tensors[i].id]).collect();

        Ok(GraphPlan { uid, tensors: plan_tensors, nodes: node_indices, buffers })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BufferHandle, BufferUsage};
    use crate::plan::GraphPlan;
    use crate::shape;
    use std::collections::HashMap;

    struct DummyBackend {
        buffers: std::sync::Mutex<HashMap<BufferHandle, usize>>,
        next: std::sync::atomic::AtomicU32,
    }

    impl DummyBackend {
        fn new() -> Self {
            Self {
                buffers: std::sync::Mutex::new(HashMap::new()),
                next: std::sync::atomic::AtomicU32::new(1),
            }
        }
    }

    impl Backend for DummyBackend {
        fn name(&self) -> &str {
            "dummy"
        }
        fn create_buffer(&self, size: usize, _usage: BufferUsage) -> Result<BufferHandle> {
            let h = BufferHandle::new(self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
            self.buffers.lock().unwrap().insert(h, size);
            Ok(h)
        }
        fn release_buffer(&self, handle: BufferHandle) -> Result<()> {
            self.buffers
                .lock()
                .unwrap()
                .remove(&handle)
                .map(|_| ())
                .ok_or_else(|| Error::msg("release_buffer: unknown"))
        }
        fn write(&self, _h: BufferHandle, _o: usize, _d: &[u8]) -> Result<()> {
            Ok(())
        }
        fn read(&self, _h: BufferHandle, _o: usize, _out: &mut [u8]) -> Result<()> {
            Ok(())
        }
        fn fill(&self, _h: BufferHandle, _o: usize, _v: u8, _l: usize) -> Result<()> {
            Ok(())
        }
        fn synchronize(&self) -> Result<()> {
            Ok(())
        }
        fn graph_compute(&self, _plan: &GraphPlan) -> Result<()> {
            Ok(())
        }

        fn supports_op(&self, _op: Op, _dt: &[DType]) -> bool {
            true
        }
    }

    #[test]
    fn compile_rejects_unsupported_op() {
        struct Stub;
        impl Backend for Stub {
            fn name(&self) -> &str {
                "stub"
            }
            fn create_buffer(&self, _s: usize, _u: BufferUsage) -> Result<BufferHandle> {
                unreachable!("compile must reject before allocating")
            }
            fn release_buffer(&self, _h: BufferHandle) -> Result<()> {
                unreachable!()
            }
            fn write(&self, _h: BufferHandle, _o: usize, _d: &[u8]) -> Result<()> {
                unreachable!()
            }
            fn read(&self, _h: BufferHandle, _o: usize, _out: &mut [u8]) -> Result<()> {
                unreachable!()
            }
            fn fill(&self, _h: BufferHandle, _o: usize, _v: u8, _l: usize) -> Result<()> {
                unreachable!()
            }
            fn synchronize(&self) -> Result<()> {
                Ok(())
            }
            fn graph_compute(&self, _plan: &GraphPlan) -> Result<()> {
                unreachable!()
            }
        }

        let mut g = GraphBuilder::new();
        let x = g.input("x", DType::F32, shape![4]).unwrap();
        let s = g.softmax(x, -1).unwrap();
        g.mark_output(s).unwrap();
        let err = g.compile(&Stub).unwrap_err();
        assert!(err.to_string().contains("does not support op softmax"));
    }

    #[test]
    fn build_chain_and_compile() {
        let mut g = GraphBuilder::new();
        let a = g.input("a", DType::F32, shape![4]).unwrap();
        let b = g.input("b", DType::F32, shape![4]).unwrap();
        let c = g.mul(a, b).unwrap();
        g.mark_output(c).unwrap();
        assert_eq!(g.tensor(c).unwrap().op, Op::Mul);
        assert_eq!(g.outputs(), &[c]);

        let plan = g.compile(&DummyBackend::new()).unwrap();
        assert_eq!(plan.nodes.len(), 1);
        assert_eq!(plan.tensors.len(), 3);
        assert!(plan.find(c).is_some());
    }

    #[test]
    fn mul_mat_shape_check() {
        let mut g = GraphBuilder::new();
        let w = g.weight("w", DType::F32, shape![16, 8]).unwrap();
        let x = g.input("x", DType::F32, shape![16, 32]).unwrap();
        let out = g.mul_mat(w, x).unwrap();
        assert_eq!(g.tensor(out).unwrap().layout.shape, shape![8, 32]);
    }

    #[test]
    fn mul_mat_rejects_inner_mismatch() {
        let mut g = GraphBuilder::new();
        let w = g.weight("w", DType::F32, shape![16, 8]).unwrap();
        let x = g.input("x", DType::F32, shape![32, 32]).unwrap();
        assert!(g.mul_mat(w, x).is_err());
    }

    #[test]
    fn mul_rejects_dtype_mismatch() {
        let mut g = GraphBuilder::new();
        let a = g.input("a", DType::F32, shape![4]).unwrap();
        let b = g.input("b", DType::I32, shape![4]).unwrap();
        assert!(g.mul(a, b).is_err());
    }

    #[test]
    fn compile_requires_output() {
        let g = GraphBuilder::new();
        assert!(g.compile(&DummyBackend::new()).is_err());
    }

    #[test]
    fn view_bounds_checked() {
        let mut g = GraphBuilder::new();
        let a = g.input("a", DType::F32, shape![8]).unwrap();
        assert!(g.view(a, 24, shape![4]).is_err());
        let v = g.view(a, 16, shape![4]).unwrap();
        assert_eq!(g.tensor(v).unwrap().view_src, Some(a));
    }

    #[test]
    fn topo_order_respects_dependencies() {
        let mut g = GraphBuilder::new();
        let a = g.input("a", DType::F32, shape![4]).unwrap();
        let b = g.input("b", DType::F32, shape![4]).unwrap();
        let c = g.mul(a, b).unwrap();
        let d = g.mul(a, c).unwrap();
        g.mark_output(d).unwrap();
        let (order, _) = g.topo_order().unwrap();
        let pos = |id: TensorId| order.iter().position(|&i| g.tensors()[i].id == id).unwrap();
        assert!(pos(c) < pos(d));
    }

    #[test]
    fn diamond_executes_source_once() {
        let mut g = GraphBuilder::new();
        let a = g.input("a", DType::F32, shape![4]).unwrap();
        let b = g.input("b", DType::F32, shape![4]).unwrap();
        let c = g.mul(a, b).unwrap();
        let d = g.add(c, c).unwrap();
        g.mark_output(d).unwrap();
        let plan = g.compile(&DummyBackend::new()).unwrap();
        assert_eq!(plan.nodes.len(), 2);
    }

    #[test]
    fn softmax_keeps_shape() {
        let mut g = GraphBuilder::new();
        let x = g.input("x", DType::F32, shape![4, 64]).unwrap();
        let s = g.softmax(x, -1).unwrap();
        assert_eq!(g.tensor(s).unwrap().layout.shape, shape![4, 64]);
        assert_eq!(g.tensor(s).unwrap().op, Op::Softmax { axis: -1 });
    }

    #[test]
    fn rope_validates_pos_dtype() {
        let mut g = GraphBuilder::new();
        let x = g.input("x", DType::F32, shape![64, 8]).unwrap();
        let pos = g.input("pos", DType::I32, shape![1]).unwrap();
        let r = g.rope(x, pos, RopeParams::default()).unwrap();
        assert_eq!(g.tensor(r).unwrap().layout.shape, shape![64, 8]);
        assert_eq!(g.tensor(r).unwrap().srcs().len(), 2);

        let bad = g.input("bad", DType::F32, shape![1]).unwrap();
        assert!(g.rope(x, bad, RopeParams::default()).is_err());
    }

    #[test]
    fn get_rows_gathers_dim0() {
        let mut g = GraphBuilder::new();
        let table = g.weight("embd", DType::F32, shape![16, 4096]).unwrap();
        let idx = g.input("idx", DType::I32, shape![8]).unwrap();
        let rows = g.get_rows(table, idx).unwrap();
        assert_eq!(g.tensor(rows).unwrap().layout.shape, shape![16, 8]);

        let bad = g.input("bad", DType::F32, shape![8]).unwrap();
        assert!(g.get_rows(table, bad).is_err());
    }

    #[test]
    fn concat_joins_axis() {
        let mut g = GraphBuilder::new();
        let a = g.input("a", DType::F32, shape![4, 3]).unwrap();
        let b = g.input("b", DType::F32, shape![4, 5]).unwrap();
        let c = g.concat(a, b, 1).unwrap();
        assert_eq!(g.tensor(c).unwrap().layout.shape, shape![4, 8]);
        assert!(g.concat(a, b, 0).is_err(), "shapes differ off-axis");

        let mut g2 = GraphBuilder::new();
        let a2 = g2.input("a", DType::F32, shape![2]).unwrap();
        let b2 = g2.input("b", DType::F32, shape![3]).unwrap();
        let c2 = g2.concat(a2, b2, -1).unwrap();
        assert_eq!(g2.tensor(c2).unwrap().layout.shape, shape![5]);
    }

    #[test]
    fn mul_mat_id_validates_experts() {
        let mut g = GraphBuilder::new();
        let w = g.weight("w", DType::F32, shape![16, 8, 4]).unwrap();
        let x = g.input("x", DType::F32, shape![16, 32]).unwrap();
        let ids = g.input("ids", DType::I32, shape![4, 32]).unwrap();
        let out = g.mul_mat_id(w, x, ids, 2).unwrap();
        assert_eq!(g.tensor(out).unwrap().layout.shape, shape![8, 32]);

        let bad_ids = g.input("bad", DType::I32, shape![3, 32]).unwrap();
        assert!(g.mul_mat_id(w, x, bad_ids, 2).is_err());
    }

    #[test]
    fn unary_masks_and_scale() {
        let mut g = GraphBuilder::new();
        let x = g.input("x", DType::F32, shape![4, 4]).unwrap();
        let m = g.diag_mask_inf(x, 0).unwrap();
        assert_eq!(g.tensor(m).unwrap().op, Op::DiagMaskInf { n_past: 0 });
        let s = g.scale(x, 0.125).unwrap();
        assert_eq!(g.tensor(s).unwrap().op, Op::Scale { alpha: 0.125 });
        let c = g.copy(x).unwrap();
        assert_eq!(g.tensor(c).unwrap().op, Op::Copy);
    }
}
