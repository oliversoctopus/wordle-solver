//! The browser's side of the solver: one JSON request in, one JSON reply out.
//!
//! The page already speaks to the solver over `/api/*` with a JSON body, and
//! sends the whole board on every submit rather than a delta, so that boundary
//! is already a pure function of (path, body). This exposes exactly that, and
//! `web/wasm-api.js` routes `fetch` into it -- which is why `web/app.js` needs
//! no changes at all and its tests keep loading it unmodified.
//!
//! Memory is marshalled by hand. JS calls `alloc` for a buffer, writes UTF-8
//! into it, calls `handle`, reads the reply out of linear memory, and calls
//! `dealloc` on both. That is the one unsafe pattern here and it is confined to
//! this file.

use std::alloc::{alloc as raw_alloc, dealloc as raw_dealloc, Layout};

mod api;
mod state;

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
static mut REPLY_PTR: *mut u8 = std::ptr::null_mut();
static mut REPLY_LEN: usize = 0;

#[no_mangle]
pub extern "C" fn reply_ptr() -> *mut u8 {
    unsafe { REPLY_PTR }
}

#[no_mangle]
pub extern "C" fn reply_len() -> usize {
    unsafe { REPLY_LEN }
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
    let reply = api::handle(&path, &body);

    let bytes = reply.into_bytes().into_boxed_slice();
    REPLY_LEN = bytes.len();
    REPLY_PTR = Box::into_raw(bytes) as *mut u8;
}

unsafe fn slice_to_str(ptr: *const u8, len: usize) -> String {
    if ptr.is_null() || len == 0 {
        return String::new();
    }
    String::from_utf8_lossy(std::slice::from_raw_parts(ptr, len)).into_owned()
}
