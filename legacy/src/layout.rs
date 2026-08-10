use crate::data_type;
use crate::data_type::DataType;
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

    pub(crate) fn nbytes(&self, dtype: DataType) -> usize {
        if self.shape.iter().any(|&dim| dim == 0) {
            return 0;
        }

        let block_size = data_type::get_block_size(dtype);
        let type_size = data_type::get_type_size(dtype);

        if block_size == 1 {
            type_size
                + self
                    .shape
                    .iter()
                    .zip(self.stride.iter())
                    .map(|(&dim, &stride)| (dim - 1) * stride)
                    .sum::<usize>()
        } else {
            (self.shape[0] * self.stride[0]) / block_size
                + self
                    .shape
                    .iter()
                    .zip(self.stride.iter())
                    .skip(1)
                    .map(|(&dim, &stride)| (dim - 1) * stride)
                    .sum::<usize>()
        }
    }
}
