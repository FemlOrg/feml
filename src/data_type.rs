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

pub fn get_size(dtype: DataType) -> usize {
    match dtype {
        DataType::U8 => 1,
        DataType::U32 => 4,
        DataType::I16 => 2,
        DataType::I32 => 8,
        DataType::I64 => 16,
        DataType::F16 => 2,
        DataType::F32 => 4,
        DataType::F64 => 8,
        _ => panic!("Invalid data type"),
    }
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
