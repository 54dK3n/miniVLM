//! TinyVLM custom GPU operators, written in Rust.
//!
//! Kernels are authored as CUDA C, JIT-compiled at runtime with nvrtc (no nvcc
//! needed), and launched on the *existing* torch CUDA context via cudarc. We
//! operate directly on raw device pointers taken from torch tensors
//! (`tensor.data_ptr()`), so no host<->device copies are introduced.

use std::sync::{Mutex, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

const SRC: &str = r#"
extern "C" __global__ void vadd(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = a[i] + b[i]; }
}

// Fused scaled-dot-product attention forward.
// q,k,v: [B,H,Lq,d] / [B,H,Lk,d] contiguous float32.
// mask : [Lq*Lk] uint8 (1 = attend, 0 = masked), broadcast over (B,H); may be unused.
// out  : [B,H,Lq,d].
// One thread computes one output row (b,h,qi) using an online-softmax pass,
// so the [B,H,Lq,Lk] score matrix is never materialized in global memory.
extern "C" __global__ void fused_attn_fwd(
        const float* q, const float* k, const float* v,
        const unsigned char* mask, float* out,
        int B, int H, int Lq, int Lk, int d, float scale, int has_mask) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)B * H * Lq;
    if (idx >= total) return;

    int qi = (int)(idx % Lq);
    long bh = idx / Lq;                      // b*H + h
    const float* qrow = q + (bh * Lq + qi) * d;
    const float* kbase = k + bh * Lk * d;
    const float* vbase = v + bh * Lk * d;
    float* orow = out + (bh * Lq + qi) * d;

    float m = -INFINITY;   // running max
    float l = 0.0f;        // running denominator
    float acc[64];
    for (int c = 0; c < d; c++) acc[c] = 0.0f;

    for (int j = 0; j < Lk; j++) {
        if (has_mask && mask[(long)qi * Lk + j] == 0) continue;
        const float* krow = kbase + j * d;
        float s = 0.0f;
        for (int c = 0; c < d; c++) s += qrow[c] * krow[c];
        s *= scale;
        float m_new = fmaxf(m, s);
        float corr = __expf(m - m_new);      // m=-inf on first hit -> 0
        float p = __expf(s - m_new);
        l = l * corr + p;
        const float* vrow = vbase + j * d;
        for (int c = 0; c < d; c++) acc[c] = acc[c] * corr + p * vrow[c];
        m = m_new;
    }
    float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int c = 0; c < d; c++) orow[c] = acc[c] * inv;
}
"#;

struct Kernels {
    stream: std::sync::Arc<CudaStream>,
    vadd: CudaFunction,
    attn_fwd: CudaFunction,
}

// cudarc context/stream/function are only touched behind a Mutex below.
unsafe impl Send for Kernels {}

static KERNELS: OnceLock<Mutex<Kernels>> = OnceLock::new();

fn kernels() -> PyResult<&'static Mutex<Kernels>> {
    if let Some(k) = KERNELS.get() {
        return Ok(k);
    }
    let ctx = CudaContext::new(0).map_err(to_py)?;
    let stream = ctx.default_stream();
    let ptx = compile_ptx(SRC).map_err(to_py)?;
    let module = ctx.load_module(ptx).map_err(to_py)?;
    let vadd = module.load_function("vadd").map_err(to_py)?;
    let attn_fwd = module.load_function("fused_attn_fwd").map_err(to_py)?;
    let _ = KERNELS.set(Mutex::new(Kernels {
        stream,
        vadd,
        attn_fwd,
    }));
    Ok(KERNELS.get().unwrap())
}

fn to_py<E: std::fmt::Debug>(e: E) -> PyErr {
    PyRuntimeError::new_err(format!("{:?}", e))
}

/// out[i] = a[i] + b[i] for i in 0..n, on raw device pointers (float32).
#[pyfunction]
fn vector_add(a_ptr: u64, b_ptr: u64, out_ptr: u64, n: i64) -> PyResult<()> {
    let guard = kernels()?.lock().unwrap();
    let n_i32 = n as i32;
    let block = 256u32;
    let grid = ((n as u32) + block - 1) / block;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = guard.stream.launch_builder(&guard.vadd);
    b.arg(&a_ptr).arg(&b_ptr).arg(&out_ptr).arg(&n_i32);
    unsafe { b.launch(cfg).map_err(to_py)? };
    guard.stream.synchronize().map_err(to_py)?;
    Ok(())
}

/// Fused scaled-dot-product attention forward on raw float32 device pointers.
/// Tensors must be contiguous [B,H,Lq,d] / [B,H,Lk,d]. `mask_ptr` points at a
/// [Lq*Lk] uint8 buffer (1=attend); pass has_mask=0 to ignore it.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fused_attention(
    q_ptr: u64,
    k_ptr: u64,
    v_ptr: u64,
    mask_ptr: u64,
    out_ptr: u64,
    b: i64,
    h: i64,
    lq: i64,
    lk: i64,
    d: i64,
    scale: f32,
    has_mask: i64,
) -> PyResult<()> {
    if d > 64 {
        return Err(PyRuntimeError::new_err("head_dim must be <= 64"));
    }
    let guard = kernels()?.lock().unwrap();
    let (bi, hi, lqi, lki, di) = (b as i32, h as i32, lq as i32, lk as i32, d as i32);
    let hm = has_mask as i32;
    let total = (b * h * lq) as u32;
    let block = 128u32;
    let grid = (total + block - 1) / block;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut bld = guard.stream.launch_builder(&guard.attn_fwd);
    bld.arg(&q_ptr)
        .arg(&k_ptr)
        .arg(&v_ptr)
        .arg(&mask_ptr)
        .arg(&out_ptr)
        .arg(&bi)
        .arg(&hi)
        .arg(&lqi)
        .arg(&lki)
        .arg(&di)
        .arg(&scale)
        .arg(&hm);
    unsafe { bld.launch(cfg).map_err(to_py)? };
    guard.stream.synchronize().map_err(to_py)?;
    Ok(())
}

#[pymodule]
fn tinyvlm_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(vector_add, m)?)?;
    m.add_function(wrap_pyfunction!(fused_attention, m)?)?;
    Ok(())
}
