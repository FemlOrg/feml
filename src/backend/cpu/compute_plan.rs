use super::thread_pool::FemlThreadPool;

pub struct ComputePlan {
    pub work_size: usize,
    pub work_data: *const u8,
    pub threadpool: *mut FemlThreadPool,
    pub n_threads: i32,

    // TODO: add feml abort_callback
}

