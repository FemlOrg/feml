use crate::context::Context;
use crate::data_type::DataType;
use crate::memory_manager::MemoryBlock;
use crate::shape::Shape;
use std::sync::Arc;

/// Unique identifier for tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(usize);

impl TensorId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

pub struct Tensor_ {
    id: TensorId,
    name: String,
    dtype: DataType,
    shape: Shape,
    storage: Option<Arc<MemoryBlock>>,
    context: Context,
}

impl Tensor_ {
    pub fn new(
        name: String,
        dtype: DataType,
        shape: Shape,
        storage: Option<Arc<MemoryBlock>>,
        context: Context,
        size: usize,
    ) -> Self {
        Self { id: TensorId::new(), name, dtype, shape, storage, context }
    }
}
