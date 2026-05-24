use crate::defs::MAX_DIMS;

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct Shape {
    dims: [usize; MAX_DIMS],
    rank: usize,
}

impl Shape {
    pub fn len(&self) -> usize {
        self.dims[..self.rank].iter().product()
    }
}
