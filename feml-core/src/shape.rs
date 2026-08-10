//! Fixed-rank (max 4) tensor shape.

use crate::error::{Error, Result};

pub const MAX_DIMS: usize = 4;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: [usize; MAX_DIMS],
    rank: usize,
}

impl Default for Shape {
    fn default() -> Self {
        Self { dims: [1; MAX_DIMS], rank: 1 }
    }
}

impl Shape {
    /// Build a shape from a slice of dimensions (rank <= 4, no zero dims).
    pub fn new(dims: &[usize]) -> Result<Self> {
        if dims.len() > MAX_DIMS {
            return Err(Error::shape(format!("rank {} exceeds MAX_DIMS = {MAX_DIMS}", dims.len())));
        }
        if dims.contains(&0) {
            return Err(Error::shape("dimensions must be non-zero"));
        }
        let mut storage = [1usize; MAX_DIMS];
        storage[..dims.len()].copy_from_slice(dims);
        Ok(Self { dims: storage, rank: dims.len() })
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn dim(&self, i: usize) -> usize {
        self.dims[i]
    }

    pub fn iter(&self) -> impl Iterator<Item = &usize> {
        self.dims[..self.rank].iter()
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.dims[..self.rank].iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl std::ops::Index<usize> for Shape {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        &self.dims[index]
    }
}

/// Convenience macro: `shape![2, 3]` == `Shape::new(&[2, 3]).unwrap()`.
#[macro_export]
macro_rules! shape {
    ($($dim:expr),* $(,)?) => {{
        const RANK: usize = <[()]>::len(&[$($crate::shape!(@sub $dim)),*]);
        assert!(RANK <= $crate::shape::MAX_DIMS, "rank exceeds MAX_DIMS");
        let mut dims = [1usize; $crate::shape::MAX_DIMS];
        let data = [$($dim as usize),*];
        dims[..RANK].copy_from_slice(&data);
        $crate::shape::Shape::new(&dims[..RANK]).expect("shape macro: invalid dims")
    }};
    (@sub $dim:expr) => { () };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape;

    #[test]
    fn new_and_accessors() {
        let s = shape![2, 3, 4];
        assert_eq!(s.rank(), 3);
        assert_eq!(s[0], 2);
        assert_eq!(s[1], 3);
        assert_eq!(s[2], 4);
        assert_eq!(s.len(), 24);
        assert_eq!(s.iter().sum::<usize>(), 9);
    }

    #[test]
    fn trailing_dims_default_to_one() {
        let s = shape![2];
        assert_eq!(s[1], 1);
        assert_eq!(s[3], 1);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn rejects_rank_gt_4() {
        assert!(Shape::new(&[1, 2, 3, 4, 5]).is_err());
    }

    #[test]
    fn rejects_zero_dim() {
        assert!(Shape::new(&[0, 2]).is_err());
    }

    #[test]
    fn equality() {
        assert_eq!(shape![1, 2], shape![1, 2]);
        assert_ne!(shape![1, 2], shape![2, 1]);
    }
}
