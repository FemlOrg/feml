#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FemlObjectType {
  FemlObjectTypeTensor,
  FemlObjectTypeGraph,
  FemlObjectTypeBuffer,
}