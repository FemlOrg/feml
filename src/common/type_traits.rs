use crate::types::TensorType;

pub fn feml_block_size(_tensor_type : TensorType) -> usize {
    return 1;
}

pub fn feml_type_size(_tensor_type : TensorType) -> usize {
    return 1;
}

pub fn feml_row_size(_tensor_type : TensorType, _block_size : usize) -> usize {
    return 1;
}