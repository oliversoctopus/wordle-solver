//! The browser's side of the solver: one JSON request in, one JSON reply out.
//!
//! The page speaks to the solver over `/api/*` with a JSON body and sends the
//! whole board every submit, so that boundary is a pure function of (path,
//! body). This exposes exactly that, and `web/wasm-api.js` routes `fetch`
//! into it. Memory is marshaled by hand: JS calls `alloc`, writes UTF-8,
//! calls `handle`, reads the reply out of linear memory, and calls `dealloc`
//! on both. That is the one unsafe pattern here, confined to this file.

use std::alloc::{alloc as raw_alloc, dealloc as raw_dealloc, Layout};
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

/// A buffer of `len` bytes for JS to write into. Pair every call with
/// `dealloc`, or the page leaks a request's worth of memory per call.
///
/// # Safety
/// The caller must not write past `len`.
#[no_mangle]
pub unsafe extern "C" fn alloc(len: usize) -> *mut u8 {
    if len == 0 {
        return std::ptr::null_mut();
    }
    raw_alloc(Layout::from_size_align_unchecked(len, 1))
}

/// Releases a buffer from `alloc`, or a reply from `handle`.
///
/// # Safety
/// `ptr` must have come from `alloc` or `handle` with this same `len`.
#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len != 0 {
        raw_dealloc(ptr, Layout::from_size_align_unchecked(len, 1));
    }
}

/// Where the last reply landed, since a wasm export returns one value and a
/// string needs two. JS reads these straight after `handle`.
///
/// Atomics rather than `static mut`: reading a mutable static is unsafe for a
/// race that cannot happen in a single-threaded module, and the 2024 edition
/// rejects the pattern outright. These compile to the same plain loads here.
static REPLY_PTR: AtomicPtr<u8> = AtomicPtr::new(std::ptr::null_mut());
static REPLY_LEN: AtomicUsize = AtomicUsize::new(0);

#[no_mangle]
pub extern "C" fn reply_ptr() -> *mut u8 {
    REPLY_PTR.load(Ordering::Relaxed)
}

#[no_mangle]
pub extern "C" fn reply_len() -> usize {
    REPLY_LEN.load(Ordering::Relaxed)
}

/// Answers one request. `path` and `body` are UTF-8 in linear memory; the reply
/// is left where `reply_ptr` and `reply_len` point, and is the caller's to free.
///
/// # Safety
/// Both pointers must be readable for their given lengths.
#[no_mangle]
pub unsafe extern "C" fn handle(
    path_ptr: *const u8,
    path_len: usize,
    body_ptr: *const u8,
    body_len: usize,
) {
    let path = slice_to_str(path_ptr, path_len);
    let body = slice_to_str(body_ptr, body_len);
    let reply = wordle_core::api::handle(&path, &body);

    let bytes = reply.into_bytes().into_boxed_slice();
    REPLY_LEN.store(bytes.len(), Ordering::Relaxed);
    REPLY_PTR.store(Box::into_raw(bytes).cast::<u8>(), Ordering::Relaxed);
}

unsafe fn slice_to_str(ptr: *const u8, len: usize) -> String {
    if ptr.is_null() || len == 0 {
        return String::new();
    }
    String::from_utf8_lossy(std::slice::from_raw_parts(ptr, len)).into_owned()
}
