use crate::context::Context;
use crate::data_type::TensorOpType;
use crate::data_type::TensorType;
use crate::error::{Error, Result};
use crate::tensor::TensorId;
use std::cell::{Ref, RefCell};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::rc::Rc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GraphId(usize);

impl GraphId {
    fn new() -> Self {
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }

    pub fn as_usize(&self) -> usize {
        self.0
    }
}

impl fmt::Display for GraphId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub struct ComputeGraphInner {
    pub(crate) id: GraphId,
    pub(crate) size: usize,
    node_count: usize,
    leaf_count: usize,
    pub(crate) nodes: Vec<TensorId>,
    pub(crate) leafs: Vec<TensorId>,
    node_use_count: HashMap<TensorId, usize>,
    visited_nodes: HashSet<TensorId>,
}

#[derive(Clone)]
pub struct ComputeGraph(pub(crate) Rc<RefCell<ComputeGraphInner>>);

impl Default for ComputeGraphInner {
    fn default() -> Self {
        Self {
            id: GraphId::new(),
            size: 0,
            node_count: 0,
            leaf_count: 0,
            nodes: Vec::new(),
            leafs: Vec::new(),
            node_use_count: HashMap::new(),
            visited_nodes: HashSet::new(),
        }
    }
}

impl ComputeGraph {
    pub fn new() -> Self {
        ComputeGraph(Rc::new(RefCell::new(ComputeGraphInner::default())))
    }

    pub fn clear(&self) {
        let mut inner = self.0.borrow_mut();
        inner.size = 0;
        inner.node_count = 0;
        inner.leaf_count = 0;
        inner.nodes.clear();
        inner.leafs.clear();
        inner.node_use_count.clear();
        inner.visited_nodes.clear();
    }

    pub fn id(&self) -> GraphId {
        self.0.borrow().id
    }

    pub fn size(&self) -> usize {
        self.0.borrow().size
    }

    pub fn node_count(&self) -> usize {
        self.0.borrow().node_count
    }

    pub fn leaf_count(&self) -> usize {
        self.0.borrow().leaf_count
    }

    pub fn nodes(&self) -> Ref<'_, [TensorId]> {
        Ref::map(self.0.borrow(), |inner| inner.nodes.as_slice())
    }

    pub fn leafs(&self) -> Ref<'_, [TensorId]> {
        Ref::map(self.0.borrow(), |inner| inner.leafs.as_slice())
    }

    pub fn use_count(&self, id: TensorId) -> usize {
        self.0.borrow().node_use_count.get(&id).copied().unwrap_or(0)
    }

    pub fn contains(&self, id: TensorId) -> bool {
        self.0.borrow().visited_nodes.contains(&id)
    }

    pub fn visit_parents(&self, context: &Context, input: TensorId) -> Result<()> {
        if self.0.borrow().visited_nodes.contains(&input) {
            return Ok(());
        }

        let tensor = context.get_tensor(input).map_err(|_err| {
            Error::msg(format!("tensor {} not found", input.as_usize()))
                .context("in ComputeGraph::visit_parents")
        })?;
        let src_tensors = tensor.src_tensor();

        self.0.borrow_mut().visited_nodes.insert(input);
        self.0.borrow_mut().size += src_tensors.len();

        for src_id in src_tensors {
            *self.0.borrow_mut().node_use_count.entry(src_id).or_insert(0) += 1;
            self.visit_parents(context, src_id)?;
        }
        if tensor.op_type() == TensorOpType::TensorNone
            && tensor.tensor_type() == TensorType::FlagParam
        {
            self.0.borrow_mut().leaf_count += 1;
            self.0.borrow_mut().leafs.push(input);
        } else {
            self.0.borrow_mut().node_count += 1;
            self.0.borrow_mut().nodes.push(input);
        }

        Ok(())
    }

    pub fn build_forward(&self, context: &Context, input: TensorId, expand: bool) -> Result<()> {
        if !expand {
            self.clear();
            return self.visit_parents(context, input);
        }

        let node_cnt_prev = self.0.borrow().node_count;
        let leaf_cnt_prev = self.0.borrow().leaf_count;
        self.visit_parents(context, input)?;
        let node_added = self.0.borrow().node_count > node_cnt_prev;
        let leaf_added = self.0.borrow().leaf_count > leaf_cnt_prev;
        if node_added || leaf_added {
            let new_nodes = &self.0.borrow().nodes[node_cnt_prev..self.0.borrow().node_count];
            let new_leafs = &self.0.borrow().leafs[leaf_cnt_prev..self.0.borrow().leaf_count];
            if !new_nodes.contains(&input) && !new_leafs.contains(&input) {
                return Err(Error::msg(format!(
                    "build_forward expected input tensor {} in newly added nodes or leafs",
                    input.as_usize()
                ))
                .context("in ComputeGraph::build_forward"));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Context;
    use crate::data_type::DataType;
    use crate::shape;
    use crate::shape::Shape;

    fn new_tensor_in_context() -> (Context, TensorId) {
        let mut ctx = Context::builder().tensor_pool_capacity(8).build();
        let shape = shape![2, 2, 1, 1];
        let tensor = ctx.new_tensor(DataType::F32, &shape).unwrap();
        (ctx, tensor.tensor_id())
    }

    fn new_test_tensor(ctx: &mut Context) -> crate::tensor::Tensor {
        let shape = shape![2, 2, 1, 1];
        ctx.new_tensor(DataType::F32, &shape).unwrap()
    }

    fn mark_as_param_leaf(tensor: &crate::tensor::Tensor) {
        tensor.set_op_type(TensorOpType::TensorNone);
        tensor.set_tensor_type(TensorType::FlagParam);
    }

    fn index_of(nodes: &[TensorId], id: TensorId) -> usize {
        nodes.iter().position(|node| *node == id).unwrap()
    }

    #[test]
    fn test_compute_graph_new_initial_state() {
        let graph = ComputeGraph::new();
        assert_eq!(graph.size(), 0);
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.leaf_count(), 0);
        assert!(graph.nodes().is_empty());
        assert!(graph.leafs().is_empty());
        assert_eq!(graph.use_count(TensorId::new()), 0);
    }

    #[test]
    fn test_visit_parents_adds_node_for_regular_tensor() {
        let (ctx, input) = new_tensor_in_context();
        let graph = ComputeGraph::new();
        graph.visit_parents(&ctx, input).unwrap();
        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.leaf_count(), 0);
        assert_eq!(&*graph.nodes(), &[input]);
        assert!(graph.leafs().is_empty());
        assert!(graph.contains(input));
    }

    #[test]
    fn test_visit_parents_adds_leaf_for_flag_param_tensor_none() {
        let (ctx, input) = new_tensor_in_context();
        {
            let t = ctx.get_tensor(input).unwrap();
            t.set_tensor_type(TensorType::FlagParam);
            t.set_op_type(TensorOpType::TensorNone);
        }
        let graph = ComputeGraph::new();
        graph.visit_parents(&ctx, input).unwrap();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.leaf_count(), 1);
        assert!(graph.nodes().is_empty());
        assert_eq!(&*graph.leafs(), &[input]);
    }

    #[test]
    fn test_visit_parents_deduplicates_visited_node() {
        let (ctx, input) = new_tensor_in_context();
        let graph = ComputeGraph::new();
        graph.visit_parents(&ctx, input).unwrap();
        graph.visit_parents(&ctx, input).unwrap();
        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.leaf_count(), 0);
        assert_eq!(graph.nodes().len(), 1);
        assert_eq!(graph.nodes()[0], input);
    }

    #[test]
    fn test_clear_resets_all_state() {
        let (ctx, input) = new_tensor_in_context();
        let graph = ComputeGraph::new();
        graph.visit_parents(&ctx, input).unwrap();
        assert_eq!(graph.node_count(), 1);
        assert!(graph.contains(input));
        graph.clear();
        assert_eq!(graph.size(), 0);
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.leaf_count(), 0);
        assert!(graph.nodes().is_empty());
        assert!(graph.leafs().is_empty());
        assert!(!graph.contains(input));
    }

    #[test]
    fn test_build_forward_expand_false_rebuilds_graph() {
        let (ctx, input) = new_tensor_in_context();
        let graph = ComputeGraph::new();
        graph.visit_parents(&ctx, input).unwrap();
        assert_eq!(graph.node_count(), 1);
        graph.build_forward(&ctx, input, false).unwrap();
        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.leaf_count(), 0);
        assert_eq!(&*graph.nodes(), &[input]);
        assert!(graph.contains(input));
    }

    #[test]
    fn test_visit_parents_returns_error_when_tensor_missing() {
        let ctx = Context::builder().tensor_pool_capacity(10).build();
        let graph = ComputeGraph::new();
        let err = graph.visit_parents(&ctx, TensorId::new()).unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn test_visit_parents_records_dfs_topology_and_use_counts() {
        let mut ctx = Context::builder().tensor_pool_capacity(10).build();
        let mut a = new_test_tensor(&mut ctx);
        let mut b = new_test_tensor(&mut ctx);
        let mut c = new_test_tensor(&mut ctx);
        let mut d = new_test_tensor(&mut ctx);

        mark_as_param_leaf(&a);
        b.set_src_tensor(a.tensor_id());
        c.set_src_tensor(a.tensor_id());
        d.set_src_tensor(b.tensor_id());
        d.set_src_tensor(c.tensor_id());

        let graph = ComputeGraph::new();
        graph.visit_parents(&ctx, d.tensor_id()).unwrap();

        assert_eq!(&*graph.leafs(), &[a.tensor_id()]);
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.use_count(a.tensor_id()), 2);
        assert_eq!(graph.use_count(b.tensor_id()), 1);
        assert_eq!(graph.use_count(c.tensor_id()), 1);

        let nodes = &*graph.nodes();
        let b_idx = index_of(nodes, b.tensor_id());
        let c_idx = index_of(nodes, c.tensor_id());
        let d_idx = index_of(nodes, d.tensor_id());
        assert!(b_idx < d_idx);
        assert!(c_idx < d_idx);
    }

    #[test]
    fn test_build_forward_expand_true_accepts_new_leaf_output() {
        let (ctx, input) = new_tensor_in_context();
        {
            let t = ctx.get_tensor(input).unwrap();
            t.set_op_type(TensorOpType::TensorNone);
            t.set_tensor_type(TensorType::FlagParam);
        }
        let graph = ComputeGraph::new();
        graph.build_forward(&ctx, input, true).unwrap();
        assert_eq!(&*graph.leafs(), &[input]);
        assert!(graph.nodes().is_empty());
    }

    #[test]
    fn test_build_forward_expand_true_does_not_duplicate_existing_nodes() {
        let (ctx, input) = new_tensor_in_context();
        let graph = ComputeGraph::new();
        graph.build_forward(&ctx, input, true).unwrap();
        let node_count = graph.node_count();
        graph.build_forward(&ctx, input, true).unwrap();
        assert_eq!(graph.node_count(), node_count);
        assert_eq!(&*graph.nodes(), &[input]);
    }
}
