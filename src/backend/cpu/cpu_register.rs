use crate::backend::{backend::FemlBackend, *};
use super::cpu_context::FemlBackendCpuContext;
use std::collections::HashMap;

type SetNThreadsFn = fn(backend: &mut FemlBackend, n_threads: i32);

// 函数枚举
enum BackendFunction {
    SetNThreads(SetNThreadsFn),
}

struct BackendRegistry {
    functions: HashMap<String, BackendFunction>,
}

impl BackendRegistry {
    fn new() -> Self {
        let mut functions = HashMap::new();
        functions.insert(
            "ggml_backend_set_n_threads".to_string(),
            BackendFunction::SetNThreads(ggml_backend_cpu_set_n_threads),
        );
        BackendRegistry { functions }
    }

    fn get_function(&self, name: &str) -> Option<&BackendFunction> {
        self.functions.get(name)
    }
}

fn ggml_backend_cpu_set_n_threads(backend: &mut FemlBackend, n_threads: i32) {
    let ctx :&mut FemlBackendCpuContext= backend.get_context::<FemlBackendCpuContext>().unwrap();
    ctx.n_threads = n_threads;
}
