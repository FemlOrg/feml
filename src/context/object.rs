use crate::types::FemlObjectType;

#[derive(Debug, Clone)]
struct FemlObject{
    offset : u32,
    size : u32,
    object_type : FemlObjectType,
}

