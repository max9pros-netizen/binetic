//! FFI bindings to llama.cpp's C API (from backends/llama-cpp/include/llama.h).
//!
//! bindings verified against llama.cpp HEAD as of the cloned revision.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(improper_ctypes_definitions)]
#![allow(dead_code)]

use std::ffi::c_char;
use std::os::raw::{c_bool, c_double, c_float, c_int, c_int64_t, c_size_t, c_uint, c_void};

// ── ggml_tensor ──────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum ggml_tensor_type {
    GGML_TENSOR_NONE = 0,
    GGML_TENSOR_F32 = 1,
    GGML_TENSOR_F16 = 2,
    GGML_TENSOR_Q4_0 = 3,
    GGML_TENSOR_Q4_1 = 4,
    GGML_TENSOR_Q5_0 = 5,
    GGML_TENSOR_Q5_1 = 6,
    GGML_TENSOR_Q8_0 = 7,
    GGML_TENSOR_Q8_1 = 8,
    GGML_TENSOR_Q4_2 = 9,
    GGML_TENSOR_Q4_3 = 10,
    GGML_TENSOR_Q8_2 = 11,
    GGML_TENSOR_Q8_3 = 12,
    GGML_TENSOR_Q4_4 = 13,
    GGML_TENSOR_Q4_5 = 14,
    GGML_TENSOR_Q4_6 = 15,
    GGML_TENSOR_Q4_7 = 16,
    GGML_TENSOR_Q8_4 = 17,
    GGML_TENSOR_Q8_5 = 18,
    GGML_TENSOR_Q8_6 = 19,
    GGML_TENSOR_Q8_7 = 20,
    GGML_TENSOR_Q4_K = 21,
    GGML_TENSOR_Q4_K_S = 22,
    GGML_TENSOR_Q8_K = 23,
    GGML_TENSOR_Q8_K_S = 24,
    GGML_TENSOR_IQ2_XXS = 25,
    GGML_TENSOR_IQ2_XS = 26,
    GGML_TENSOR_IQ2_S = 27,
    GGML_TENSOR_IQ3_XXS = 28,
    GGML_TENSOR_IQ3_XS = 29,
    GGML_TENSOR_IQ3_S = 30,
    GGML_TENSOR_IQ1_S = 31,
    GGML_TENSOR_IQ4_XXS = 32,
    GGML_TENSOR_IQ4_XS = 33,
    GGML_TENSOR_IQ4_S = 34,
    GGML_TENSOR_IQ5_XXS = 35,
    GGML_TENSOR_IQ5_XS = 36,
    GGML_TENSOR_IQ5_S = 37,
    GGML_TENSOR_IQ6_XXS = 38,
    GGML_TENSOR_IQ6_XS = 39,
    GGML_TENSOR_IQ6_S = 40,
    GGML_TENSOR_BF16 = 41,
    GGML_TENSOR_I8 = 42,
    GGML_TENSOR_I16 = 43,
    GGML_TENSOR_I32 = 44,
    GGML_TENSOR_I64 = 45,
    GGML_TENSOR_F64 = 46,
    GGML_TENSOR_COUNT,
}

#[repr(C)]
pub struct ggml_tensor {
    pub type_: ggml_tensor_type,
    pub nelements: c_size_t,
    pub bytes_per_element: usize,
    pub offset: usize,
    pub wdim: u32,
    pub n_dims: u32,
    pub dims: [u32; 3],
    pub opaque: *mut c_void,
    pub name: *const c_char,
}

// ── Opaque llama types ───────────────────────────────────────────────────────

#[repr(C)]
pub struct llama_model {
    _private: [u8; 0],
}

#[repr(C)]
pub struct llama_context {
    _private: [u8; 0],
}

#[repr(C)]
pub struct llama_vocab {
    _private: [u8; 0],
}

pub type llama_memory_t = *mut c_void;
pub type llama_token = c_int;
pub type llama_pos = c_int;
pub type llama_seq_id = c_int;

// ── llama_model_params ───────────────────────────────────────────────────────
// Matches the C struct from llama.h (lines 314-351).

#[repr(C)]
pub struct llama_model_params {
    pub devices: *mut c_void,
    pub tensor_buft_overrides: *mut c_void,
    pub n_gpu_layers: c_int,
    pub split_mode: c_int,
    pub load_mode: c_int,
    pub lazy_mode: c_int,
    pub main_gpu: c_int,
    pub tensor_split: *mut c_float,
    pub progress_callback: *mut c_void,
    pub progress_callback_user_data: *mut c_void,
    pub kv_overrides: *mut c_void,
    pub vocab_only: c_bool,
    pub check_tensors: c_bool,
    pub use_extra_bufts: c_bool,
    pub no_host: c_bool,
    pub no_alloc: c_bool,
    pub load_mtp: c_bool,
}

// ── llama_context_params ─────────────────────────────────────────────────────
// Matches the C struct from llama.h (lines 360-409+).

#[repr(C)]
pub struct llama_context_params {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_rs_seq: u32,
    pub n_outputs_max: u32,
    pub n_outputs_max_per_seq: u32,
    pub n_threads: c_int,
    pub n_threads_batch: c_int,
    pub ctx_type: c_int,
    pub rope_scaling_type: c_int,
    pub pooling_type: c_int,
    pub attention_type: c_int,
    pub flash_attn_type: c_int,
    pub rope_freq_base: c_float,
    pub rope_freq_scale: c_float,
    pub yarn_ext_factor: c_float,
    pub yarn_attn_factor: c_float,
    pub yarn_beta_fast: c_float,
    pub yarn_beta_slow: c_float,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: c_float,
    pub cb_eval: *mut c_void,
    pub cb_eval_user_data: *mut c_void,
    pub type_k: c_int,
    pub type_v: c_int,
    pub abort_callback: *mut c_void,
    pub abort_callback_data: *mut c_void,
    pub embeddings: c_bool,
    pub offload_kqv: c_bool,
    pub no_perf: c_bool,
    pub op_offload: c_bool,
    pub swa_full: c_bool,
    pub kv_unified: c_bool,
}

// ── llama_batch ──────────────────────────────────────────────────────────────

#[repr(C)]
pub struct llama_batch {
    pub n_tokens: c_int,
    pub tokens: *const c_int,
    pub embd: *const c_float,
    pub pos: *const c_int,
    pub seq_id: *const c_int,
    pub logits: *mut c_int8_t,
}

// ── llama_token_data ─────────────────────────────────────────────────────────

#[repr(C)]
pub struct llama_token_data {
    pub id: c_int,
    pub logit: c_float,
    pub prob: c_float,
}

#[repr(C)]
pub struct llama_token_data_array {
    pub n_tokens: c_int,
    pub data: *mut llama_token_data,
}

// ── llama_sampler ─────────────────────────────────────────────────────────────

#[repr(C)]
pub struct llama_sampler {
    _private: [u8; 0],
}

#[repr(C)]
pub struct llama_sampler_chain_params {
    pub samplers: *const *const llama_sampler,
    pub n_samplers: c_int,
}

// ── llama_model functions ────────────────────────────────────────────────────

extern "C" {
    pub fn llama_model_default_params() -> llama_model_params;
    pub fn llama_model_load_from_file(
        path_model: *const c_char,
        params: llama_model_params,
    ) -> *mut llama_model;
    pub fn llama_model_free(model: *mut llama_model);
    pub fn llama_model_n_ctx(model: *const llama_model) -> c_uint;
    pub fn llama_model_n_embd(model: *const llama_model) -> c_uint;
    pub fn llama_model_n_layer(model: *const llama_model) -> c_uint;
    pub fn llama_model_n_head(model: *const llama_model) -> c_uint;
    pub fn llama_model_n_vocab(model: *const llama_model) -> c_uint;
    pub fn llama_model_name(model: *const llama_model) -> *const c_char;
    pub fn llama_model_path(model: *const llama_model) -> *const c_char;
    pub fn llama_model_get_vocab(model: *const llama_model) -> *const llama_vocab;
    pub fn llama_model_rope_type(model: *const llama_model) -> c_int;
    pub fn llama_model_n_ctx_train(model: *const llama_model) -> c_int;
    pub fn llama_model_n_embd_inp(model: *const llama_model) -> c_int;
    pub fn llama_model_n_embd_out(model: *const llama_model) -> c_int;
    pub fn llama_model_n_head_kv(model: *const llama_model) -> c_int;
    pub fn llama_model_n_swa(model: *const llama_model) -> c_int;
    pub fn llama_model_n_layer_nextn(model: *const llama_model) -> c_int;
}

// ── llama_context functions ──────────────────────────────────────────────────

extern "C" {
    pub fn llama_context_default_params() -> llama_context_params;
    pub fn llama_init_from_model(
        model: *mut llama_model,
        params: llama_context_params,
    ) -> *mut llama_context;
    pub fn llama_free(ctx: *mut llama_context);
    pub fn llama_get_model(ctx: *const llama_context) -> *const llama_model;
    pub fn llama_get_memory(ctx: *const llama_context) -> llama_memory_t;
    pub fn llama_get_vocab(ctx: *const llama_context) -> *const llama_vocab;
    pub fn llama_n_ctx(ctx: *const llama_context) -> c_uint;
    pub fn llama_n_ctx_seq(ctx: *const llama_context) -> c_uint;
    pub fn llama_n_batch(ctx: *const llama_context) -> c_uint;
    pub fn llama_n_ubatch(ctx: *const llama_context) -> c_uint;
    pub fn llama_n_seq_max(ctx: *const llama_context) -> c_uint;
    pub fn llama_get_tensor(
        ctx: *const llama_context,
        name: *const c_char,
    ) -> *const ggml_tensor;
    pub fn llama_eval(
        ctx: *mut llama_context,
        tokens: *const c_int,
        n_tokens: c_int,
        n_past: c_int,
        params: *const llama_batch,
    ) -> c_int;
    pub fn llama_reset(ctx: *mut llama_context);
    pub fn llama_time_us() -> c_int64_t;
    pub fn llama_supports_mmap() -> c_bool;
    pub fn llama_supports_mlock() -> c_bool;
    pub fn llama_supports_gpu_offload() -> c_bool;
    pub fn llama_backend_init();
    pub fn llama_backend_free();
}

// ── llama_vocab functions ────────────────────────────────────────────────────

extern "C" {
    pub fn llama_vocab_get_token(vocab: *const llama_vocab, index: c_int) -> *const c_char;
    pub fn llama_vocab_get_token_count(vocab: *const llama_vocab) -> c_int;
    pub fn llama_vocab_get_eos_token_id(vocab: *const llama_vocab) -> c_int;
    pub fn llama_vocab_get_bos_token_id(vocab: *const llama_vocab) -> c_int;
    pub fn llama_vocab_n_tokens(vocab: *const llama_vocab) -> c_int;
    pub fn llama_tokenize(
        vocab: *const llama_vocab,
        text: *const c_char,
        tokens: *mut c_int,
        n_max_tokens: c_int,
        apply_special: c_bool,
        parse_special: c_bool,
    ) -> c_int;
}

// ── llama_sampler functions ──────────────────────────────────────────────────

extern "C" {
    pub fn llama_sampler_chain_default_params() -> llama_sampler_chain_params;
    pub fn llama_sampler_chain_init(
        params: llama_sampler_chain_params,
    ) -> *mut llama_sampler;
    pub fn llama_sampler_free(sampler: *mut llama_sampler);
    pub fn llama_sampler_sample(
        sampler: *mut llama_sampler,
        ctx: *const llama_context,
        logits: *const c_float,
        n_past: c_int,
    ) -> c_int;
    pub fn llama_sampler_get_type(sampler: *const llama_sampler) -> c_int;
}

// ── llama_batch helpers ──────────────────────────────────────────────────────

extern "C" {
    pub fn llama_batch_get_one(
        token: llama_token,
        pos: llama_pos,
        n_seq_id: c_int,
    ) -> llama_batch;
    pub fn llama_batch_init(
        n_tokens: c_int,
        n_seq: c_int,
        Flags: c_uint,
    ) -> llama_batch;
    pub fn llama_batch_free(batch: llama_batch);
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Convert a C string pointer to an owned String, returning None if null.
pub unsafe fn cstr_to_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        None
    } else {
        let cstr = std::ffi::CStr::from_ptr(ptr);
        cstr.to_str().ok().map(|s| s.to_owned())
    }
}

/// Convert a C string pointer to an owned String, panicking on null/invalid.
pub unsafe fn cstr_to_string_unwrap(ptr: *const c_char) -> String {
    cstr_to_string(ptr).expect("llama.cpp returned null string")
}