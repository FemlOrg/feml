pub struct ObjectPool<T> {
    free: Vec<T>,
    factory: fn() -> T,
}

impl<T> ObjectPool<T> {
    pub fn new(factory: fn() -> T) -> Self {
        Self {
            free: Vec::new(),
            factory,
        }
    }

    pub fn with_capacity(factory: fn() -> T, capacity: usize) -> Self {
        Self {
            free: Vec::with_capacity(capacity),
            factory,
        }
    }

    pub fn get(&mut self) -> T {
        self.free.pop().unwrap_or_else(|| (self.factory)())
    }

    pub fn put(&mut self, obj: T) {
        self.free.push(obj);
    }

    pub fn size(&self) -> usize {
        self.free.len()
    }
}