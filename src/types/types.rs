#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FemlObjectType {
    FemlObjectTypeTensor,
    FemlObjectTypeGraph,
    FemlObjectTypeBuffer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorType {
    TensorUnknown,
    TensorTypeF32,
    TensorTypeF16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FemlOpType {
    FemlOpTypeUnknown,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FemlType {
   FemlTypeF32,
   FemlTypeF16 
}

pub const FEML_TYPE_COUNT: usize = 2;