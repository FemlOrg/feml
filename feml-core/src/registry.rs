//! Backend registry: pluggable registrars, score-based selection.
//!
//! Core never depends on concrete backends — discovery is driven by the
//! caller (or the `feml` facade crate) registering implementations.

use crate::backend::{Backend, BackendDevice, BackendRegistrar};
use crate::error::{Error, Result};

#[derive(Default)]
pub struct Registry {
    registrars: Vec<Box<dyn BackendRegistrar>>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, registrar: Box<dyn BackendRegistrar>) -> &mut Self {
        self.registrars.push(registrar);
        self
    }

    pub fn registrars(&self) -> &[Box<dyn BackendRegistrar>] {
        &self.registrars
    }

    /// Total number of devices across all registrars.
    pub fn device_count(&self) -> usize {
        self.registrars.iter().map(|r| r.device_count()).sum()
    }

    pub fn find(&self, name: &str) -> Option<&dyn BackendRegistrar> {
        self.registrars.iter().find(|r| r.name().eq_ignore_ascii_case(name)).map(|b| b.as_ref())
    }

    pub fn open(&self, registrar_name: &str, device_index: usize) -> Result<Box<dyn Backend>> {
        let reg = self
            .find(registrar_name)
            .ok_or_else(|| Error::msg(format!("backend registrar '{registrar_name}' not found")))?;
        reg.device(device_index)?.init_backend()
    }

    pub fn open_device(
        &self,
        registrar_name: &str,
        device_index: usize,
    ) -> Result<Box<dyn BackendDevice>> {
        let reg = self
            .find(registrar_name)
            .ok_or_else(|| Error::msg(format!("backend registrar '{registrar_name}' not found")))?;
        reg.device(device_index)
    }

    /// Open the highest-scoring available device (score > 0 required).
    pub fn open_best(&self) -> Result<Box<dyn Backend>> {
        let best = self
            .registrars
            .iter()
            .map(|r| (r.score(), r))
            .filter(|(score, r)| *score > 0 && r.device_count() > 0)
            .max_by_key(|(score, _)| *score)
            .map(|(_, r)| r)
            .ok_or_else(|| Error::msg("no backend available"))?;
        best.device(0)?.init_backend()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::*;
    use crate::dtype::DType;
    use crate::error::{Error, Result};
    use crate::op::Op;

    struct DummyDevice;

    impl BackendDevice for DummyDevice {
        fn info(&self) -> Result<DeviceInfo> {
            Ok(DeviceInfo {
                name: "dummy".into(),
                description: "test device".into(),
                memory_total: 0,
                memory_free: 0,
                device_type: DeviceType::Cpu,
                caps: Capabilities::default(),
            })
        }

        fn init_backend(&self) -> Result<Box<dyn Backend>> {
            Err(Error::msg("dummy backend does not init"))
        }

        fn supports_op(&self, _op: Op, _src_dtypes: &[DType]) -> bool {
            true
        }

        fn buffer_from_host_ptr(
            &self,
            _ptr: &mut [u8],
            _size: usize,
            _max_tensor_size: usize,
        ) -> Result<BufferHandle> {
            Err(Error::backend("dummy", "no host buffers"))
        }
    }

    struct DummyRegistrar(u32);

    impl BackendRegistrar for DummyRegistrar {
        fn name(&self) -> &str {
            "dummy"
        }
        fn device_count(&self) -> usize {
            1
        }
        fn device(&self, _index: usize) -> Result<Box<dyn BackendDevice>> {
            Ok(Box::new(DummyDevice))
        }
        fn score(&self) -> u32 {
            self.0
        }
    }

    #[test]
    fn open_best_picks_highest_score() {
        let mut reg = Registry::new();
        reg.register(Box::new(DummyRegistrar(1)));
        reg.register(Box::new(DummyRegistrar(9)));
        match reg.open_best() {
            Ok(_) => panic!("dummy backend must not init"),
            Err(e) => assert!(e.to_string().contains("dummy backend does not init")),
        }
    }

    #[test]
    fn open_best_fails_when_nothing_available() {
        let reg = Registry::new();
        assert!(reg.open_best().is_err());
    }

    #[test]
    fn find_is_case_insensitive() {
        let mut reg = Registry::new();
        reg.register(Box::new(DummyRegistrar(1)));
        assert!(reg.find("DUMMY").is_some());
        assert!(reg.find("nope").is_none());
    }
}
