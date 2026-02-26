use std::collections::HashMap;
use std::sync::Arc;
use crate::memory_manager::MemoryManager;
use crate::tensor::{TensorId, Tensor_};
use crate::compute_graph::{GraphId, ComputeGraph};
use crate::shape::Shape;
use crate::data_type::DataType;

pub struct Context_ {
  memory_manager: Arc<MemoryManager>,
  pub tensor_tables: HashMap<TensorId, Tensor_>,
  pub graph_tables: HashMap<GraphId, ComputeGraph>,
}
pub struct Context(Arc<Context_>);

impl AsRef<Context> for Context {
    fn as_ref(&self) -> &Context {
        self
    }
}

impl std::ops::Deref for Context {
  type Target = Context_;

  fn deref(&self) -> &Self::Target {
      self.0.as_ref()
  }
}

impl Context_ {
    pub fn new(size: &usize) -> Option<Self> {
        Self {
            memory_manager: MemoryManager::new(*size, 0),
            tensor_tables: HashMap::new(),
            graph_tables: HashMap::new(),
        }.into()
    }

    pub fn new_tensor(self: &Self, dtype: DataType, shape: &Shape) -> Result<Tensor> {
      todo!();
    }

    // TODO
    pub fn new_graph(self: &Self) -> Result<ComputeGraph> {
      todo!();
    }
}

