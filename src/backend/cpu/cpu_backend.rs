use super::cpu_context::FemlBackendCpuContext;
use crate::backend::backend::FemlBackend;
use crate::backend::backend_trait::FemlBackendInterface;
use crate::backend::cpu::compute_graph::FemlComputeGraph;
use crate::feml_error;
use crate::types::FemlStatus;

pub struct FemlBackendCpuImpl {}

impl FemlBackendInterface for FemlBackendCpuImpl {
    fn get_name(&self, _backend: &FemlBackend) -> &'static str {
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
        _backend: &FemlBackend,
        _tensor: &mut crate::common::tensor::FemlTensor,
        _data: *const u8,
        _offset: usize,
        _size: usize,
    ) {
        // implement async set tensor data for CPU
        feml_error!("set_tensor_async is not implemented for CPU backend");
    }

    fn get_tensor_async(
        &self,
        _backend: &FemlBackend,
        _tensor: &mut crate::common::tensor::FemlTensor,
        _data: *const u8,
        _offset: usize,
        _size: usize,
    ) {
        feml_error!("get_tensor_async is not implemented for CPU backend");
    }

    fn cpy_tensor_async(
        &self,
        _bakend_src: &FemlBackend,
        _backend_dst: &FemlBackend,
        _src: &crate::common::tensor::FemlTensor,
        _dst: &mut crate::common::tensor::FemlTensor,
    ) -> bool {
        // implement async copy tensor data for CPU
        feml_error!("cpy_tensor_async is not implemented for CPU backend");
        false
    }

    fn synchronize(&self, _backend: &FemlBackend) {
        // implement synchronize for CPU
        feml_error!("synchronize is not implemented for CPU backend");
    }

    fn graph_plan_create(&self, _backend: &mut FemlBackend, _compute_graph: &FemlComputeGraph) {
        // feml_backend_cpu_graph_plan_create(backend, compute_graph);
    }

    fn graph_plan_free(&self, _backend: &FemlBackend, _plan: *const u8) {
        todo!()
    }

    fn graph_plan_unpdate(
        &self,
        _backend: &FemlBackend,
        _plan: *const u8,
        _compute_graph: &FemlComputeGraph,
    ) {
        todo!()
    }

    fn graph_plan_compute(&self, _backend: &FemlBackend, _plan: *const u8) -> FemlStatus {
        todo!()
    }

    fn graph_compute(&self, _backend: &FemlBackend, _compute_graph: &FemlComputeGraph) {
        todo!()
    }

    fn event_record(
        &self,
        _backend: &FemlBackend,
        _event: &crate::backend::backend::FemlBackendEvent,
    ) {
        // implement event record for CPU
        feml_error!("event_record is not implemented for CPU backend");
    }

    fn event_wait(
        &self,
        _backend: &FemlBackend,
        _event: &crate::backend::backend::FemlBackendEvent,
    ) {
        // implement event wait for CPU
        feml_error!("event_wait is not implemented for CPU backend");
    }
}
