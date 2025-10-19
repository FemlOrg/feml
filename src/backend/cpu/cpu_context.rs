use super::thread_pool::FemlThreadPool;
use crate::common::def::FemlAbortCallBack;
pub(crate) struct FemlBackendCpuContext {
    pub n_threads: i32,
    pub threadpool : *mut FemlThreadPool,
    pub work_data: *mut u8,
    pub work_size : u32,

    pub abort_callback: Option<FemlAbortCallBack>,
    pub abort_callback_data: *mut u8,
}

impl FemlBackendCpuContext {
    pub fn new(n_threads: i32) -> Self {
        FemlBackendCpuContext {
            n_threads,
            threadpool: std::ptr::null_mut(),
            work_data: std::ptr::null_mut(),
            work_size: 0,
            abort_callback: None,
            abort_callback_data: std::ptr::null_mut(),
        }
    }
}