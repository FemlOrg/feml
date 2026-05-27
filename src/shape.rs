use crate::defs::MAX_DIMS;
use std::{collections::btree_set::Range, ops::Index};
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Shape {
    pub(crate) dims: [usize; MAX_DIMS],
    pub(crate) rank: usize,
}

impl Index<usize> for Shape {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.rank);
        &self.dims[index]
    }
}

impl Default for Shape {
    fn default() -> Self {
        Self { dims: [0; 4], rank: 4 }
    }
}

impl Shape {
    pub fn new(dims: &[usize]) -> Self {
        assert!(dims.len() <= MAX_DIMS);

        let mut storage = [0; MAX_DIMS];
        storage[..dims.len()].copy_from_slice(dims);

        Self { dims: storage, rank: dims.len() }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, usize> {
        self.dims[..self.rank].iter()
    }
}

#[macro_export]
macro_rules! shape {
    ($($dim:expr),* $(,)?) => {{
        const RANK: usize = <[()]>::len(&[$(shape!(@sub $dim)),*]);

        assert!(RANK <= $crate::defs::MAX_DIMS);

        let mut dims = [0usize; $crate::defs::MAX_DIMS];

        let data = [$($dim as usize),*];

        dims[..RANK].copy_from_slice(&data);

        $crate::shape::Shape {
            dims,
            rank: RANK,
        }
    }};

    (@sub $dim:expr) => { () };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_equality() {
        let s1 = shape![1, 2, 3, 4];
        let s2 = shape![1, 2, 3, 4];
        let s3 = shape![2, 3, 4, 5];

        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
    }

    #[test]
    fn test_shape_debug() {
        let shape = shape![1, 2, 3, 4];
        let debug_str = format!("{:?}", shape);
        assert!(debug_str.contains("1"));
        assert!(debug_str.contains("2"));
        assert!(debug_str.contains("3"));
        assert!(debug_str.contains("4"));
    }

    #[test]
    fn test_shape_default() {
        let shape = Shape::default();
        assert_eq!(shape, shape![0, 0, 0, 0]);
    }
}
