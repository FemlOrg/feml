use std::alloc::{alloc, dealloc, Layout};
use std::fmt;
use std::ptr;
use std::sync::{Arc, Mutex, Weak};

/// Represents a memory region in the pool.
struct Region {
    /// Pointer to the memory block.
    ptr: *mut u8,
    /// Size of the allocated memory.
    size: usize,
}

impl Drop for Region {
    /// Frees the allocated memory when the `Region` is dropped.
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let layout =
                    Layout::from_size_align(self.size, size_of::<u8>()).expect("Invalid layout");
                dealloc(self.ptr, layout);
            }
        }
    }
}

unsafe impl Send for Region {}

/// Represents a free memory segment in the pool.
#[derive(Debug, Clone, Copy)]
struct FreeSegment {
    /// The ID of the region to which the segment belongs.
    region_id: usize,
    /// The starting address of the free segment.
    start: usize,
    /// The length of the free segment.
    len: usize,
}

impl FreeSegment {
    /// Returns the end address of the free segment.
    fn end(&self) -> usize {
        self.start + self.len
    }
}

/// Holds the state of the memory manager, including allocated regions and free memory segments.
struct ManagerState {
    /// List of all memory regions allocated by the memory manager.
    regions: Vec<Region>,
    /// List of free memory segments available for allocation.
    free_list: Vec<FreeSegment>,
    /// Default size of the memory pages.
    default_page_size: usize,
}

unsafe impl Send for ManagerState {}
unsafe impl Sync for ManagerState {}

/// MemoryManager manages memory regions and handles allocation and deallocation.
pub struct MemoryManager {
    /// Mutex to protect the internal state of the memory manager.
    inner: Mutex<ManagerState>,
    /// The ratio used to determine whether to split a segment.
    size_compare_ratios: usize,
}

impl MemoryManager {
    /// Creates a new `MemoryManager` with a specified initial size and comparison ratio for size splitting.
    ///
    /// # Arguments
    ///
    /// * `initial_size` - The initial size of the memory pool.
    /// * `size_compare_ratio` - The ratio used to determine whether to split a segment.
    ///
    /// # Returns
    ///
    /// A new instance of `MemoryManager` wrapped in an `Arc` for shared ownership.
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

    /// Expands the heap by allocating a new memory region.
    ///
    /// # Arguments
    ///
    /// * `min_size` - The minimum size required for the new memory region.
    fn expand_heap_internal(&self, min_size: usize) {
        let mut state = self.inner.lock().unwrap();

        let alloc_size = std::cmp::max(state.default_page_size, min_size);

        // Allocate memory using `std::alloc::alloc`
        let new_ptr = unsafe {
            let layout =
                Layout::from_size_align(alloc_size, size_of::<u8>()).expect("Invalid layout");
            let ptr = alloc(layout) as *mut u8;
            if ptr.is_null() {
                panic!("System Out of Memory: failed to allocate {} bytes", alloc_size);
            }
            ptr::write_bytes(ptr, 0, alloc_size); // Initialize the allocated memory to zero.
            ptr
        };

        let region_id = state.regions.len();

        state.regions.push(Region { ptr: new_ptr, size: alloc_size });

        state.free_list.push(FreeSegment { region_id, start: 0, len: alloc_size });

        println!("$$ Memory pool expanded: New Region {}, Size {} $$", region_id, alloc_size);
    }

    /// Allocates a memory block of the specified size with optional padding.
    ///
    /// # Arguments
    ///
    /// * `size` - The size of the memory block to allocate.
    /// * `padding` - Padding used for memory alignment.
    ///
    /// # Returns
    ///
    /// An `Option<Arc<MemoryBlock>>` representing the allocated memory block.
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

                    // Check if the current segment is large enough to allocate
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
            println!("!! Memory exhausted, attempting to expand...");
            self.expand_heap_internal(needed_size);
        }

        None
    }

    /// Releases the memory block and adds it back to the free list.
    ///
    /// # Arguments
    ///
    /// * `region_id` - The ID of the region.
    /// * `start` - The start address of the memory block.
    /// * `len` - The length of the memory block to release.
    fn release(&self, region_id: usize, start: usize, len: usize) {
        let mut state = self.inner.lock().unwrap();
        println!("<< release: Region {} addr {}, len {}", region_id, start, len);

        state.free_list.push(FreeSegment { region_id, start, len });

        // Sort the free list by region_id and start address
        state.free_list.sort_by(|a, b| match a.region_id.cmp(&b.region_id) {
            std::cmp::Ordering::Equal => a.start.cmp(&b.start),
            other => other,
        });

        // Merge adjacent free segments
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

    /// Reads data from the memory region.
    ///
    /// # Arguments
    ///
    /// * `region_id` - The ID of the region.
    /// * `start` - The starting address from where to read.
    /// * `len` - The number of bytes to read.
    ///
    /// # Returns
    ///
    /// A `Vec<u8>` containing the read data.
    pub fn read_memory(&self, region_id: usize, start: usize, len: usize) -> Vec<u8> {
        let state = self.inner.lock().unwrap();
        if let Some(region) = state.regions.get(region_id) {
            assert!(start + len <= region.size, "Read out of bounds within region");
            unsafe {
                let src = region.ptr.add(start);
                std::slice::from_raw_parts(src, len).to_vec()
            }
        } else {
            panic!("Invalid region ID");
        }
    }

    /// Writes data to the memory region.
    ///
    /// # Arguments
    ///
    /// * `region_id` - The ID of the region.
    /// * `start` - The starting address where to write.
    /// * `data` - The data to write to the memory.
    pub fn write_memory(&self, region_id: usize, start: usize, data: &[u8]) {
        let state = self.inner.lock().unwrap();
        if let Some(region) = state.regions.get(region_id) {
            let end = start + data.len();
            assert!(end <= region.size, "Write out of bounds within region");
            unsafe {
                let dst = region.ptr.add(start);
                ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
            }
        } else {
            panic!("Invalid region ID");
        }
    }
}

/// Represents a memory block allocated from the pool.
pub struct MemoryBlock {
    /// The ID of the region.
    pub region_id: usize,
    /// The start address of the memory block.
    pub start: usize,
    /// The length of the memory block.
    pub len: usize,
    /// Weak reference to the memory manager for deallocation.
    manager: Weak<MemoryManager>,
}

impl MemoryBlock {
    /// Writes data to the memory block.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to write.
    pub fn write(&self, data: &[u8]) {
        if let Some(mgr) = self.manager.upgrade() {
            mgr.write_memory(self.region_id, self.start, data);
        }
    }

    /// Reads data from the memory block.
    ///
    /// # Returns
    ///
    /// A `Vec<u8>` containing the data read from the memory block.
    pub fn read(&self) -> Vec<u8> {
        if let Some(mgr) = self.manager.upgrade() {
            mgr.read_memory(self.region_id, self.start, self.len)
        } else {
            Vec::new()
        }
    }
}

impl Drop for MemoryBlock {
    /// Releases the memory block when it is dropped.
    fn drop(&mut self) {
        if let Some(manager) = self.manager.upgrade() {
            manager.release(self.region_id, self.start, self.len);
        }
    }
}

impl fmt::Debug for MemoryBlock {
    /// Custom formatter for `MemoryBlock` for debugging.
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
        // Initially only 100 bytes available.
        let mgr = MemoryManager::new(100, 256);

        // Allocate 80 bytes, successful
        let b1 = mgr.alloc(80, 1).expect("Initial alloc failed");
        assert_eq!(b1.region_id, 0);

        // Allocate 80 more, but Region 0 only has 20 left, not enough -> triggers expansion.
        let b2 = mgr.alloc(80, 1).expect("Expansion alloc failed");

        // b2 should be in the new Region 1
        assert_eq!(b2.region_id, 1);
        assert_eq!(b2.start, 0); // Start at the beginning of the new region

        // Verify that writes and reads do not interfere
        b1.write(&[0xAA; 10]);
        b2.write(&[0xBB; 10]);

        assert_eq!(b1.read()[0], 0xAA);
        assert_eq!(b2.read()[0], 0xBB);
    }

    #[test]
    fn test_large_allocation_trigger_grow() {
        let mgr = MemoryManager::new(100, 256);

        // Allocate a huge block (1000 bytes), initial pool is only 100, expansion happens.
        let b1 = mgr.alloc(1000, 1).expect("Huge alloc failed");

        assert_eq!(b1.len, 1000);
        assert_eq!(b1.region_id, 1); // Region 1 should be allocated.
    }

    #[test]
    fn test_fragmentation_and_grow() {
        let mgr = MemoryManager::new(100, 256);

        let _b1 = mgr.alloc(50, 1).unwrap();
        let _b2 = mgr.alloc(30, 1).unwrap();
        // Remaining 20 bytes

        // Request 50 bytes, not enough -> triggers expansion
        let b3 = mgr.alloc(50, 1).unwrap();
        assert_ne!(b3.region_id, 0);
    }

    #[test]
    fn test_cross_region_independence() {
        let mgr = MemoryManager::new(50, 256);
        let mut blocks = Vec::new();

        // Allocate multiple blocks to force multiple regions
        for i in 0..5 {
            let b = mgr.alloc(40, 1).unwrap();
            b.write(&[i as u8; 40]);
            blocks.push(b);
        }

        // Verify data in each block
        for (i, b) in blocks.iter().enumerate() {
            let data = b.read();
            assert!(data.iter().all(|&x| x == i as u8));
        }
    }

    #[test]
    fn test_release_and_reuse_correct_region() {
        let mgr = MemoryManager::new(100, 256);

        let b1 = mgr.alloc(100, 1).unwrap(); // Region 0 full
        let b2 = mgr.alloc(100, 1).unwrap(); // Region 1 full

        let region_id_b1 = b1.region_id;
        drop(b1); // Release Region 0

        // Reallocate 100 bytes, should reuse Region 0 instead of creating a new region.
        let b3 = mgr.alloc(100, 1).unwrap();
        assert_eq!(b3.region_id, region_id_b1);

        drop(b2);
        drop(b3);
    }
}
