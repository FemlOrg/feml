//! feml-core: tensor types, layouts, ops, and backend traits for LLM inference.
//!
//! Zero backend dependencies — backends (feml-cpu, feml-opencl, ...) depend on
//! this crate, never the other way around.

pub mod backend;
pub mod dtype;
pub mod error;
pub mod graph;
pub mod layout;
pub mod op;
pub mod plan;
mod planner;
pub mod registry;
pub mod shape;
pub mod tensor;

pub use dtype::DType;
pub use error::{Error, Result};
pub use graph::GraphBuilder;
pub use layout::Layout;
pub use op::{Op, RopeParams};
pub use plan::{GraphPlan, PlanBuffer, PlanTensor};
pub use registry::Registry;
pub use shape::Shape;
pub use tensor::{TensorFlags, TensorId, TensorMeta};

/// Convenience prelude for typical usage.
pub mod prelude {
    pub use crate::backend::{
        Backend, BackendDevice, BackendRegistrar, BufferHandle, BufferUsage, Capabilities,
        DeviceInfo, DeviceType,
    };
    pub use crate::dtype::DType;
    pub use crate::error::{Error, Result};
    pub use crate::graph::GraphBuilder;
    pub use crate::layout::Layout;
    pub use crate::op::{Op, RopeParams};
    pub use crate::registry::Registry;
    pub use crate::shape;
    pub use crate::shape::{MAX_DIMS, Shape};
    pub use crate::tensor::{TensorFlags, TensorId, TensorMeta}; // shape![...] macro
}
