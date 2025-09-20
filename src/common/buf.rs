#[derive(Debug, Clone)]
pub struct MemoryBuffer {
    pub size: usize,
    pub buf: Vec<u8>,
}

impl MemoryBuffer {
    pub fn new(size: usize) -> Self {
        Self {
            size: size,
            buf: vec![0u8; size],
        }
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.buf.as_ptr()
    }
}
