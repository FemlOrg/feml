use std::rc::Rc;

use crate::backend::BackendBuffer;

#[derive(Clone)]
pub(crate) struct TensorStorage {
    buffer: Option<Rc<dyn BackendBuffer>>,
    offset: usize,
    size: usize,
}
