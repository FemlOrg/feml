use crate::defs::MAX_DIMS;

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct Shape {
    dims: [usize; MAX_DIMS],
    rank: usize,
}
