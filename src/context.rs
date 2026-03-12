use crate::compute_graph::{ ComputeGraph, GraphId };
use crate::data_type::{ DataType, get_block_size, get_type_size };
use crate::object_pool::ObjectPool;
use crate::shape::Shape;
use crate::tensor::{ self, Tensor, Tensor_, TensorId };
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use crate::error::Result;

pub struct Context_ {
    pub tensor_pool: ObjectPool<Tensor_>,
    pub tensor_tables: HashMap<TensorId, Tensor>,
    pub graph_tables: HashMap<GraphId, ComputeGraph>,
}
pub struct Context(Arc<RefCell<Context_>>);

impl AsRef<Context> for Context {
    fn as_ref(&self) -> &Context {
        self
    }
}

impl std::ops::Deref for Context {
    type Target = RefCell<Context_>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Context_ {
    pub fn new(size: &usize) -> Option<Self> {
        (Self {
            tensor_pool: ObjectPool::with_capacity(Tensor_::default, *size),
            tensor_tables: HashMap::new(),
            graph_tables: HashMap::new(),
        }).into()
    }

    fn new_tensor_impl(
        self: &mut Self,
        dtype: DataType,
        shape: &Shape,
        view_src: Option<Tensor>
    ) -> Result<Tensor> {
        let mut tensor_ = self.tensor_pool.get();
        tensor_.dtype = dtype;
        tensor_.layout.shape = shape.clone();
        tensor_.id = TensorId::new();
        view_src.map(|src| {
            tensor_.storage = src.borrow().storage.clone();
        });
        tensor_.layout.stride[0] = get_type_size(dtype);
        tensor_.layout.stride[1] =
            tensor_.layout.stride[0] * (tensor_.layout.stride[0] / get_block_size(dtype));
        for i in 2..4 {
            tensor_.layout.stride[i] = tensor_.layout.stride[i - 1] * tensor_.layout.shape.0[i - 1];
        }
        let tensor = Tensor(Arc::new(RefCell::new(tensor_)));
        self.tensor_tables.insert(tensor.borrow().id, tensor.clone());

        Ok(tensor)
    }
}

impl Context {
    // TODO
    pub fn new_tensor(self: &mut Self, dtype: DataType, shape: &Shape) -> Result<Tensor> {
        self.0.borrow_mut().new_tensor_impl(dtype, shape, None)
    }
    pub fn new_graph(self: &Self) -> Result<ComputeGraph> {
        todo!();
    }
}
