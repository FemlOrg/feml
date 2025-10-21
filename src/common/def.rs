use std::any::Any;

pub const FEML_MAX_DIMS: usize = 3;
pub const FEML_MAX_PARAMS: usize = 64;
pub const FEML_MAX_SRC: usize = 10;
pub const FEML_TENSOR_SIZE: usize = 336;
pub const FEML_MEM_ALIGN: usize = 16;
pub const FEML_DEFAULT_N_THREAD: i32 = 4;
pub const FEML_BACKEND_API_VERION: i32 = 1;
pub const FEML_TENSOR_ALIGNMENT: usize = 32;

pub type FemlGuid = [u8; 16];

pub type FemlAbortCallBack = Box<dyn FnMut(&mut dyn Any) -> bool>;
