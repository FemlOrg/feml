pub(super) struct CpuBackendContext {
    n_threads: usize,
    data: Vec<u8>,
    abort_fn: Option<Box<dyn Fn() -> bool + Send + Sync>>,
}

impl CpuBackendContext {
    pub fn new() -> Self {
        Self { n_threads: 1, data: Vec::new(), abort_fn: None }
    }
}
