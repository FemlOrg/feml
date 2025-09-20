use crate::types::TensorType;
use crate::types::FEML_TYPE_COUNT;

struct FemlTypeTraits<'a> {
    type_name:  &'a str,
    blck_size: i64,
    blk_size_interleave: i64,
    type_size: usize,
    is_quantized: bool,
}

static TYPE_TRAITS: [FemlTypeTraits;FEML_TYPE_COUNT] = [
    FemlTypeTraits{
        type_name: "f32", 
        blck_size: 1, 
        blk_size_interleave: 1, 
        type_size: std::mem::size_of::<f32>(), 
        is_quantized : false
    },
    FemlTypeTraits{
        type_name: "f16", 
        blck_size: 1, 
        blk_size_interleave: 1, 
        type_size: std::mem::size_of::<u16>(), 
        is_quantized : false
    },
];


pub fn feml_block_size(tensor_type : TensorType) -> usize {
    TYPE_TRAITS[tensor_type as usize].blck_size as usize
}

pub fn feml_type_size(tensor_type : TensorType) -> usize {
    TYPE_TRAITS[tensor_type as usize].type_size
}

pub fn feml_row_size(tensor_type : TensorType, ne: usize) -> usize {
    assert!(ne % feml_block_size(tensor_type) == 0);
    (feml_type_size(tensor_type) * ne) / feml_block_size(tensor_type)
}