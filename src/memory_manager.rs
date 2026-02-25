use std::fmt;
use std::ptr;
use std::sync::{Arc, Mutex, Weak};

struct Region {
    ptr: *mut u8,
    size: usize,
}

impl Drop for Region {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                // println!("Freeing region of size {}", self.size);
                libc::free(self.ptr as *mut libc::c_void);
            }
        }
    }
}

unsafe impl Send for Region {}

#[derive(Debug, Clone, Copy)]
struct FreeSegment {
    region_id: usize,
    start: usize,
    len: usize,
}

impl FreeSegment {
    fn end(&self) -> usize {
        self.start + self.len
    }
}

struct ManagerState {
    regions: Vec<Region>,
    free_list: Vec<FreeSegment>,
    default_page_size: usize,
}

unsafe impl Send for ManagerState {}
unsafe impl Sync for ManagerState {}

pub struct MemoryManager {
    inner: Mutex<ManagerState>,
    size_compare_ratios: usize,
}

impl MemoryManager {
    pub fn new(initial_size: usize, size_compare_ratio: usize) -> Arc<Self> {
        let state = ManagerState {
            regions: Vec::new(),
            free_list: Vec::new(),
            default_page_size: initial_size,
        };

        let mgr =
            Arc::new(Self { inner: Mutex::new(state), size_compare_ratios: size_compare_ratio });

        mgr.expand_heap_internal(initial_size);

        mgr
    }

    fn expand_heap_internal(&self, min_size: usize) {
        let mut state = self.inner.lock().unwrap();

        let alloc_size = std::cmp::max(state.default_page_size, min_size);

        let new_ptr = unsafe {
            let ptr = libc::malloc(alloc_size) as *mut u8;
            if ptr.is_null() {
                panic!("System Out of Memory: failed to allocate {} bytes", alloc_size);
            }
            libc::memset(ptr as *mut libc::c_void, 0, alloc_size);
            ptr
        };

        let region_id = state.regions.len();

        state.regions.push(Region { ptr: new_ptr, size: alloc_size });

        state.free_list.push(FreeSegment { region_id, start: 0, len: alloc_size });

        println!("$$ 内存池扩容: 新增 Region {}, 大小 {} $$", region_id, alloc_size);
    }

    pub fn alloc(self: &Arc<Self>, size: usize, padding: usize) -> Option<Arc<MemoryBlock>> {
        let needed_size = if padding > 0 { (size + padding - 1) & !(padding - 1) } else { size };

        for _ in 0..2 {
            {
                let mut state = self.inner.lock().unwrap();
                let find_result = state.free_list.iter().position(|seg| seg.len >= needed_size);

                if let Some(index) = find_result {
                    let segment = &mut state.free_list[index];
                    let region_id = segment.region_id;
                    let alloc_start = segment.start;
                    let current_len = segment.len;

                    let threshold =
                        (current_len.checked_mul(self.size_compare_ratios).unwrap_or(usize::MAX))
                            >> 8;

                    if current_len == needed_size
                        || (current_len > needed_size && threshold <= needed_size)
                    {
                        state.free_list.remove(index);
                        println!(
                            ">> alloc: Region {} addr {}, len {} (whole/tolerate)",
                            region_id, alloc_start, current_len
                        );

                        return Some(Arc::new(MemoryBlock {
                            region_id,
                            start: alloc_start,
                            len: current_len,
                            manager: Arc::downgrade(self),
                        }));
                    } else {
                        segment.start += needed_size;
                        segment.len -= needed_size;
                        println!(
                            ">> alloc: Region {} addr {}, len {} (split)",
                            region_id, alloc_start, needed_size
                        );

                        return Some(Arc::new(MemoryBlock {
                            region_id,
                            start: alloc_start,
                            len: needed_size,
                            manager: Arc::downgrade(self),
                        }));
                    }
                }
            }
            println!("!! 内存不足，尝试扩容...");
            self.expand_heap_internal(needed_size);
        }

        None
    }

    fn release(&self, region_id: usize, start: usize, len: usize) {
        let mut state = self.inner.lock().unwrap();
        println!("<< release: Region {} addr {}, len {}", region_id, start, len);

        state.free_list.push(FreeSegment { region_id, start, len });

        state.free_list.sort_by(|a, b| match a.region_id.cmp(&b.region_id) {
            std::cmp::Ordering::Equal => a.start.cmp(&b.start),
            other => other,
        });

        let mut new_free_list = Vec::new();
        if let Some(first) = state.free_list.first() {
            let mut current = *first;
            for next in state.free_list.iter().skip(1) {
                if current.region_id == next.region_id && current.end() == next.start {
                    current.len += next.len;
                } else {
                    new_free_list.push(current);
                    current = *next;
                }
            }
            new_free_list.push(current);
        }
        state.free_list = new_free_list;
    }

    pub fn read_memory(&self, region_id: usize, start: usize, len: usize) -> Vec<u8> {
        let state = self.inner.lock().unwrap();
        if let Some(region) = state.regions.get(region_id) {
            assert!(start + len <= region.size, "read out of bounds within region");
            unsafe {
                let src = region.ptr.add(start);
                std::slice::from_raw_parts(src, len).to_vec()
            }
        } else {
            panic!("Invalid region id");
        }
    }

    pub fn write_memory(&self, region_id: usize, start: usize, data: &[u8]) {
        let state = self.inner.lock().unwrap();
        if let Some(region) = state.regions.get(region_id) {
            let end = start + data.len();
            assert!(end <= region.size, "write out of bounds within region");
            unsafe {
                let dst = region.ptr.add(start);
                ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
            }
        } else {
            panic!("Invalid region id");
        }
    }
}

pub struct MemoryBlock {
    pub region_id: usize,
    pub start: usize,
    pub len: usize,
    manager: Weak<MemoryManager>,
}

impl MemoryBlock {
    pub fn write(&self, data: &[u8]) {
        if let Some(mgr) = self.manager.upgrade() {
            mgr.write_memory(self.region_id, self.start, data);
        }
    }

    pub fn read(&self) -> Vec<u8> {
        if let Some(mgr) = self.manager.upgrade() {
            mgr.read_memory(self.region_id, self.start, self.len)
        } else {
            Vec::new()
        }
    }
}

impl Drop for MemoryBlock {
    fn drop(&mut self) {
        if let Some(manager) = self.manager.upgrade() {
            manager.release(self.region_id, self.start, self.len);
        }
    }
}

impl fmt::Debug for MemoryBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MemoryBlock {{ region: {}, start: {}, len: {} }}",
            self.region_id, self.start, self.len
        )
    }
}

// ----------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_grow() {
        // 初始只有 100 字节
        let mgr = MemoryManager::new(100, 256);

        // 分配 80，成功
        let b1 = mgr.alloc(80, 1).expect("Initial alloc failed");
        assert_eq!(b1.region_id, 0);

        // 再分配 80，原 Region 0 只剩 20，不够 -> 触发扩容
        // 扩容大小 = max(100, 80) = 100
        let b2 = mgr.alloc(80, 1).expect("Expansion alloc failed");

        // b2 应该在新的 Region 1 中
        assert_eq!(b2.region_id, 1);
        assert_eq!(b2.start, 0); // 新 Region 的起始位置

        // 验证读写互不干扰
        b1.write(&[0xAA; 10]);
        b2.write(&[0xBB; 10]);

        assert_eq!(b1.read()[0], 0xAA);
        assert_eq!(b2.read()[0], 0xBB);
    }

    #[test]
    fn test_large_allocation_trigger_grow() {
        let mgr = MemoryManager::new(100, 256);

        // 直接分配一个超大块 (1000 bytes)
        // 初始池只有 100，肯定不够 -> 扩容 -> 申请 1000 字节的新 Region
        let b1 = mgr.alloc(1000, 1).expect("Huge alloc failed");

        assert_eq!(b1.len, 1000);
        // Region 0 是初始的 100，Region 1 应该是 1000
        assert_eq!(b1.region_id, 1);
    }

    #[test]
    fn test_fragmentation_and_grow() {
        let mgr = MemoryManager::new(100, 256);

        let _b1 = mgr.alloc(50, 1).unwrap();
        let _b2 = mgr.alloc(30, 1).unwrap();
        // 剩余 20

        // 请求 50，不够 -> 扩容
        let b3 = mgr.alloc(50, 1).unwrap();
        assert_ne!(b3.region_id, 0);
    }

    #[test]
    fn test_cross_region_independence() {
        let mgr = MemoryManager::new(50, 256);
        let mut blocks = Vec::new();

        // 连续分配，迫使创建多个 Region
        for i in 0..5 {
            let b = mgr.alloc(40, 1).unwrap();
            b.write(&[i as u8; 40]);
            blocks.push(b);
        }

        // 验证每个块的数据
        for (i, b) in blocks.iter().enumerate() {
            let data = b.read();
            assert!(data.iter().all(|&x| x == i as u8));
            // 它们可能分布在不同的 region_id
            println!("Block {}: Region {}, Start {}", i, b.region_id, b.start);
        }
    }

    #[test]
    fn test_release_and_reuse_correct_region() {
        let mgr = MemoryManager::new(100, 256);

        let b1 = mgr.alloc(100, 1).unwrap(); // Region 0 满
        let b2 = mgr.alloc(100, 1).unwrap(); // Region 1 满

        let region_id_b1 = b1.region_id;
        drop(b1); // 释放 Region 0

        // 再分配 100，应该复用 Region 0，而不是开辟 Region 2
        let b3 = mgr.alloc(100, 1).unwrap();
        assert_eq!(b3.region_id, region_id_b1);

        drop(b2);
        drop(b3);
    }
}
