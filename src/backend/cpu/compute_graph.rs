use crate::common::tensor::FemlTensor;

enum FemlComputeGraphEvalOrder {
    LeftToRight,
    RightToLeft,
    Count,
}

pub(crate) struct FemlComputeGraph {
    size : i32,
    n_nodes: i32,
    n_leafs: i32,

    nodes: Vec<Vec<FemlTensor>>,
    grads: Vec<Vec<FemlTensor>>,
    grad_accs: Vec<Vec<FemlTensor>>,
    leafs: Vec<Vec<FemlTensor>>,
    use_counts: i32,

    // TODO: add FemlHashset

    eval_order: FemlComputeGraphEvalOrder,
}