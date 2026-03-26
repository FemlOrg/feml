use crate::defs::MAX_DIMS;

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct Shape(pub [usize; MAX_DIMS]);
