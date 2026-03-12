/// The different types of elements allowed in tensors.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DataType {
    // Unsigned 8 bits integer.
    U8,
    // Unsigned 32 bits integer.
    U32,
    // Signed 16 bits integer.
    I16,
    // Signed 32 bits integer.
    I32,
    // Signed 64 bits integer.
    I64,
    // Floating-point using half precision (16 bits).
    F16,
    // Floating-point using single precision (32 bits).
    F32,
    // Floating-point using double precision (64 bits).
    F64,
}

pub struct DataTypeTraits {
    pub name: &'static str,
    pub block_size: usize,
    pub type_size: usize,
    pub quantized: bool,
}

static DATA_TYPE_TRAITS: [DataTypeTraits; 8] = [
    DataTypeTraits { name: "U8", block_size: 1, type_size: 1, quantized: true },
    DataTypeTraits { name: "U32", block_size: 4, type_size: 4, quantized: true },
    DataTypeTraits { name: "I16", block_size: 2, type_size: 2, quantized: true },
    DataTypeTraits { name: "I32", block_size: 4, type_size: 4, quantized: true },
    DataTypeTraits { name: "I64", block_size: 8, type_size: 8, quantized: true },
    DataTypeTraits { name: "F16", block_size: 2, type_size: 2, quantized: false },
    DataTypeTraits { name: "F32", block_size: 4, type_size: 4, quantized: false },
    DataTypeTraits { name: "F64", block_size: 8, type_size: 8, quantized: false },
];

pub fn get_type_size(dtype: DataType) -> usize {
    DATA_TYPE_TRAITS[dtype as usize].type_size
}

pub fn get_block_size(dtype: DataType) -> usize {
    DATA_TYPE_TRAITS[dtype as usize].block_size
}

/// The different types of tensors.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum TensorType {
    UNKNOWN,
    InputParam,
    OutputParam,
}
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum TensorOpType {
    UNKNOWN,
    TensorOpView,
    TensorOpMul,
}
