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
    FemlOpReshape,
    FemlOpView,
    FemlOpPermute,
    FemlOpTranspose,
    FemlOpCpy,
    FemlOpSetRows,
    FemlOpMulMat,
    FemlOpSoftMaxBack,
    FemlOpIm2ColBack,
    FemlOpGetRowsBack,
    FemlOpOutProd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FemlType {
    FemlTypeF32,
    FemlTypeF16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FemlStatus {
    AllocFailed,
    Failed,
    Success,
    Aborted,
}

pub const FEML_TYPE_COUNT: usize = 2;
