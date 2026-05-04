use crate::shape::Shape;

pub struct Layout {
    pub shape: Shape,
    pub stride: [usize; 4],
    pub start_offset: usize,
}

impl Layout {
    pub fn default() -> Self {
        Self { shape: Shape::default(), stride: [0; 4], start_offset: 0 }
    }
    pub fn new(shape: Shape, stride: [usize; 4], start_offset: usize) -> Self {
        Self { shape, stride, start_offset }
    }
}
