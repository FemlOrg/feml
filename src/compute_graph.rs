/// Unique identifier for tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GraphId(usize);
use crate::context::Context;
use crate::data_type::TensorOpType;
use crate::data_type::TensorType;
use crate::error::{Error, Result};
use crate::tensor::TensorId;
use std::collections::{HashMap, HashSet};

impl GraphId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }

    pub fn as_usize(&self) -> usize {
        self.0
    }
}

pub struct ComputeGraph {
    id: GraphId,
    size: usize,
    node_count: usize,
    leaf_count: usize,
    nodes: Vec<TensorId>,
    leafs: Vec<TensorId>,
    node_use_count: HashMap<TensorId, usize>,
    visited_nodes: HashSet<TensorId>,
}

impl ComputeGraph {
    pub fn new() -> Self {
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

    pub fn clear(&mut self) {
        self.size = 0;
        self.node_count = 0;
        self.leaf_count = 0;
        self.nodes.clear();
        self.leafs.clear();
        self.node_use_count.clear();
        self.visited_nodes.clear();
    }

    pub fn id(&self) -> GraphId {
        self.id
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }

    pub fn leaf_count(&self) -> usize {
        self.leaf_count
    }

    pub fn nodes(&self) -> &[TensorId] {
        &self.nodes
    }

    pub fn leafs(&self) -> &[TensorId] {
        &self.leafs
    }

    pub fn use_count(&self, id: TensorId) -> usize {
        self.node_use_count.get(&id).copied().unwrap_or(0)
    }

    pub fn contains(&self, id: TensorId) -> bool {
        self.visited_nodes.contains(&id)
    }

    pub fn visit_parents(&mut self, context: &Context, input: TensorId) -> Result<()> {
        if self.visited_nodes.contains(&input) {
            return Ok(());
        }

        let mut tensor = context.borrow().tensor_tables.get(&input).cloned().ok_or_else(|| {
            Error::msg(format!("tensor {} not found in context.tensor_tables", input.as_usize()))
                .context("in ComputeGraph::visit_parents")
        })?;
        let src_tensors = tensor.get_src_tensor();

        self.visited_nodes.insert(input);
        self.size += src_tensors.len();

        for src_id in src_tensors {
            *self.node_use_count.entry(src_id).or_insert(0) += 1;
            self.visit_parents(context, src_id)?;
        }

        if tensor.borrow().op_type == TensorOpType::TensorNone
            && tensor.borrow().tensor_type == TensorType::FlagParam
        {
            self.leaf_count += 1;
            tensor.set_name(format!("{}_leaf_{}", self.id.0, self.leaf_count));
            self.leafs.push(input);
        } else {
            self.node_count += 1;
            tensor.set_name(format!("{}_node_{}", self.id.0, self.node_count));
            self.nodes.push(input);
        }

        Ok(())
    }

    pub fn build_forward(
        &mut self,
        context: &Context,
        input: TensorId,
        expand: bool,
    ) -> Result<()> {
        if !expand {
            self.clear();
            return self.visit_parents(context, input);
        }

        let node_cnt_prev = self.node_count;
        let leaf_cnt_prev = self.leaf_count;
        self.visit_parents(context, input)?;
        let node_added = self.node_count > node_cnt_prev;
        let leaf_added = self.leaf_count > leaf_cnt_prev;
        if node_added || leaf_added {
            let new_nodes = &self.nodes[node_cnt_prev..self.node_count];
            let new_leafs = &self.leafs[leaf_cnt_prev..self.leaf_count];
            if !new_nodes.contains(&input) && !new_leafs.contains(&input) {
                return Err(Error::msg(format!(
                    "build_forward expected input tensor {} in newly added nodes or leafs",
                    input.as_usize()
                ))
                .context("in ComputeGraph::build_forward"));
            }
        } else if !self.contains(input) {
            return Err(Error::msg(format!(
                "build_forward expected input tensor {} to be present in graph",
                input.as_usize()
            ))
            .context("in ComputeGraph::build_forward"));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_type::DataType;
    use crate::shape::Shape;

    fn new_tensor_in_context() -> (Context, TensorId) {
        let mut ctx = Context::new(8);
        let shape = Shape([2, 2, 1, 1]);
        let tensor = ctx.new_tensor(DataType::F32, &shape).unwrap();
        (ctx, tensor.get_tensor_id())
    }

    fn new_test_tensor(ctx: &mut Context) -> crate::tensor::Tensor {
        let shape = Shape([2, 2, 1, 1]);
        ctx.new_tensor(DataType::F32, &shape).unwrap()
    }

    fn mark_as_param_leaf(tensor: &mut crate::tensor::Tensor) {
        tensor.borrow_mut().op_type = TensorOpType::TensorNone;
        tensor.borrow_mut().tensor_type = TensorType::FlagParam;
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
        let mut graph = ComputeGraph::new();

        graph.visit_parents(&ctx, input).unwrap();

        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.leaf_count(), 0);
        assert_eq!(graph.nodes(), &[input]);
        assert!(graph.leafs().is_empty());
        assert!(graph.contains(input));

        let tensor_name = ctx.borrow().tensor_tables.get(&input).unwrap().get_name();
        assert_eq!(tensor_name, format!("{}_node_1", graph.id().as_usize()));
    }

    #[test]
    fn test_visit_parents_adds_leaf_for_flag_param_tensor_none() {
        let (ctx, input) = new_tensor_in_context();
        {
            let tensor = ctx.borrow().tensor_tables.get(&input).unwrap().clone();
            tensor.borrow_mut().op_type = TensorOpType::TensorNone;
            tensor.borrow_mut().tensor_type = TensorType::FlagParam;
        }

        let mut graph = ComputeGraph::new();
        graph.visit_parents(&ctx, input).unwrap();

        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.leaf_count(), 1);
        assert!(graph.nodes().is_empty());
        assert_eq!(graph.leafs(), &[input]);

        let tensor_name = ctx.borrow().tensor_tables.get(&input).unwrap().get_name();
        assert_eq!(tensor_name, format!("{}_leaf_1", graph.id().as_usize()));
    }

    #[test]
    fn test_visit_parents_deduplicates_visited_node() {
        let (ctx, input) = new_tensor_in_context();
        let mut graph = ComputeGraph::new();

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
        let mut graph = ComputeGraph::new();

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
        let mut graph = ComputeGraph::new();

        graph.visit_parents(&ctx, input).unwrap();
        assert_eq!(graph.node_count(), 1);

        graph.build_forward(&ctx, input, false).unwrap();

        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.leaf_count(), 0);
        assert_eq!(graph.nodes(), &[input]);
        assert!(graph.contains(input));
    }

    #[test]
    fn test_visit_parents_returns_error_when_tensor_missing() {
        let ctx = Context::new(8);
        let mut graph = ComputeGraph::new();

        let err = graph.visit_parents(&ctx, TensorId::new()).unwrap_err();

        assert!(err.to_string().contains("not found in context.tensor_tables"));
    }

    #[test]
    fn test_visit_parents_records_dfs_topology_and_use_counts() {
        let mut ctx = Context::new(8);
        let mut a = new_test_tensor(&mut ctx);
        let mut b = new_test_tensor(&mut ctx);
        let mut c = new_test_tensor(&mut ctx);
        let mut d = new_test_tensor(&mut ctx);

        mark_as_param_leaf(&mut a);
        b.set_src_tensor(a.get_tensor_id());
        c.set_src_tensor(a.get_tensor_id());
        d.set_src_tensor(b.get_tensor_id());
        d.set_src_tensor(c.get_tensor_id());

        let mut graph = ComputeGraph::new();
        graph.visit_parents(&ctx, d.get_tensor_id()).unwrap();

        assert_eq!(graph.leafs(), &[a.get_tensor_id()]);
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.use_count(a.get_tensor_id()), 2);
        assert_eq!(graph.use_count(b.get_tensor_id()), 1);
        assert_eq!(graph.use_count(c.get_tensor_id()), 1);

        let nodes = graph.nodes();
        let b_idx = index_of(nodes, b.get_tensor_id());
        let c_idx = index_of(nodes, c.get_tensor_id());
        let d_idx = index_of(nodes, d.get_tensor_id());
        assert!(b_idx < d_idx);
        assert!(c_idx < d_idx);
    }

    #[test]
    fn test_build_forward_expand_true_accepts_new_leaf_output() {
        let (ctx, input) = new_tensor_in_context();
        {
            let tensor = ctx.borrow().tensor_tables.get(&input).unwrap().clone();
            tensor.borrow_mut().op_type = TensorOpType::TensorNone;
            tensor.borrow_mut().tensor_type = TensorType::FlagParam;
        }
        let mut graph = ComputeGraph::new();

        graph.build_forward(&ctx, input, true).unwrap();

        assert_eq!(graph.leafs(), &[input]);
        assert!(graph.nodes().is_empty());
    }

    #[test]
    fn test_build_forward_expand_true_does_not_duplicate_existing_nodes() {
        let (ctx, input) = new_tensor_in_context();
        let mut graph = ComputeGraph::new();

        graph.build_forward(&ctx, input, true).unwrap();
        let node_count = graph.node_count();

        graph.build_forward(&ctx, input, true).unwrap();

        assert_eq!(graph.node_count(), node_count);
        assert_eq!(graph.nodes(), &[input]);
    }
}
