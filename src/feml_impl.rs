use std::alloc::{Layout, alloc, dealloc};
use std::ptr::NonNull;

pub(crate) fn feml_aligned_free(ptr: *mut u8, size: usize) {
    let alignment: usize = 64;
    let layout = Layout::from_size_align(size, alignment).unwrap();
    unsafe {
        dealloc(ptr, layout);
    };
}

pub(crate) fn feml_aligned_malloc(size: usize) -> NonNull<u8> {
    let alignment: usize = 64;
    let layout = Layout::from_size_align(size, alignment).unwrap();
    unsafe {
        let ptr = alloc(layout);
        NonNull::new(ptr).expect("allocation falied")
    }
}
