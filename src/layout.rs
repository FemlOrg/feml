use crate::shape::Shape;

pub struct Layout {
    pub shape: Shape,
    pub stride: Vec<usize>,
    pub start_offset: usize,
}
