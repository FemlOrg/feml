use super::cpu_context::FemlBackendCpuContext;
use crate::backend::backend::{FemlBackendDevice, FemlBackendReg};
use crate::backend::backend_trait::{FemlBackendInterface, FemlBackendRegInterface};
use crate::backend::cpu::compute_graph::FemlComputeGraph;
use crate::feml_error;
use crate::types::FemlStatus;
use crate::{backend::backend::FemlBackend, common::def::FEML_DEFAULT_N_THREAD};

pub struct FemlBackendCpuImpl {}

impl FemlBackendInterface for FemlBackendCpuImpl {
    fn get_name(&self, backend: &FemlBackend) -> *const str {
        "CPU"
    }

    fn free(&self, backend: &mut FemlBackend) {
        // free resources
        let ctx: &mut FemlBackendCpuContext =
            backend.get_context::<FemlBackendCpuContext>().unwrap();
        let _ = ctx.work_data;
        let _ = ctx;
        let _ = backend;
    }

    fn set_tensor_async(
        &self,
        backend: &FemlBackend,
        tensor: &mut crate::common::tensor::FemlTensor,
        data: *const u8,
        offset: usize,
        size: usize,
    ) {
        // implement async set tensor data for CPU
        feml_error!("set_tensor_async is not implemented for CPU backend");
    }

    fn get_tensor_async(
        &self,
        backend: &FemlBackend,
        tensor: &mut crate::common::tensor::FemlTensor,
        data: *const u8,
        offset: usize,
        size: usize,
    ) {
        feml_error!("get_tensor_async is not implemented for CPU backend");
    }

    fn cpy_tensor_async(
        &self,
        bakend_src: &FemlBackend,
        backend_dst: &FemlBackend,
        src: &crate::common::tensor::FemlTensor,
        dst: &mut crate::common::tensor::FemlTensor,
    ) -> bool {
        // implement async copy tensor data for CPU
        feml_error!("cpy_tensor_async is not implemented for CPU backend");
        false
    }

    fn synchronize(&self, backend: &FemlBackend) {
        // implement synchronize for CPU
        feml_error!("synchronize is not implemented for CPU backend");
    }

    fn graph_plan_create(&self, backend: &FemlBackend, compute_graph: &FemlComputeGraph) {
        todo!()
    }

    fn graph_plan_free(&self, backend: &FemlBackend, plan: *const u8) {
        todo!()
    }

    fn graph_plan_unpdate(
        &self,
        backend: &FemlBackend,
        plan: *const u8,
        compute_graph: &FemlComputeGraph,
    ) {
        todo!()
    }

    fn graph_plan_compute(&self, backend: &FemlBackend, plan: *const u8) -> FemlStatus {
        todo!()
    }

    fn graph_compute(&self, backend: &FemlBackend, compute_graph: &FemlComputeGraph) {
        todo!()
    }

    fn event_record(
        &self,
        backend: &FemlBackend,
        event: &crate::backend::backend::FemlBackendEvent,
    ) {
        // implement event record for CPU
        feml_error!("event_record is not implemented for CPU backend");
    }

    fn event_wait(&self, backend: &FemlBackend, event: &crate::backend::backend::FemlBackendEvent) {
        // implement event wait for CPU
        feml_error!("event_wait is not implemented for CPU backend");
    }
}

// TODO: init feml cpu init
pub fn feml_cpu_init() {}

pub fn feml_backend_cpu_init() -> Option<FemlBackend> {
    todo!()
}
