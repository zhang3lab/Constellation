// mlp_b4_clean.cu
//
// Clean MLP orchestration for the current row-major single-route pipeline.
//
// Pipeline:
//   up    = matvec_decode(input_half,  w_up)    -> half output
//   gate  = matvec_decode(input_half,  w_gate)  -> half output
//   up_f  = cast(up_half   -> float)            [implemented inside vector op kernel path if needed]
//   gate_f= cast(gate_half -> float)            [here we directly use float workspace via local cast]
//   fused = up_f * silu(gate_f)
//   out_f = matvec_decode(fused_float, w_down)
//   out   = cast(out_f -> half)
//
// Workspace layout in expert.h:
//   [up(float)][gate(float)][fused(float)][out_f(float)]

#include "expert.h"


#include <cstdio>
#include <cstdint>

namespace {


__global__ void cast_float_to_half_kernel_local(const float* in, __half* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __float2half(in[idx]);
}

bool launch_cast_float_to_half_local(const float* d_in,
                                     __half* d_out,
                                     int n,
                                     cudaStream_t stream) {
    if (!d_in || !d_out || n < 0) return false;
    if (n == 0) return true;
    const int threads = 256;
    const int blocks = ceil_div_int(n, threads);
    cast_float_to_half_kernel_local<<<blocks, threads, 0, stream>>>(d_in, d_out, n);
    return cudaGetLastError() == cudaSuccess;
}

__global__ void silu_mul_kernel_local(const float* up,
                                      const float* gate,
                                      float* out,
                                      int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = gate[idx];
        float s = g / (1.0f + expf(-g));
        out[idx] = up[idx] * s;
    }
}

bool launch_silu_mul_local(const float* d_up,
                           const float* d_gate,
                           float* d_out,
                           int num_tokens,
                           int inter_dim,
                           cudaStream_t stream) {
    if (!d_up || !d_gate || !d_out || num_tokens < 0 || inter_dim < 0) return false;
    int n = num_tokens * inter_dim;
    if (n == 0) return true;
    const int threads = 256;
    const int blocks = ceil_div_int(n, threads);
    silu_mul_kernel_local<<<blocks, threads, 0, stream>>>(d_up, d_gate, d_out, n);
    return cudaGetLastError() == cudaSuccess;
}

inline float* workspace_ptr(float* d_workspace, size_t byte_offset) {
    return reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(d_workspace) + byte_offset);
}

bool validate_matrix(
    const PackedRowMajorMatrix& W,
    int expected_rows,
    int expected_cols,
    const char* name) {
    if (!W.weights || !W.scales) {
        std::fprintf(stderr, "%s has null device pointer\n", name);
        return false;
    }
    if (W.rows != expected_rows || W.cols != expected_cols) {
        std::fprintf(stderr,
                     "%s shape mismatch: got [%d, %d], expected [%d, %d]\n",
                     name, W.rows, W.cols, expected_rows, expected_cols);
        return false;
    }
    if (W.k_chunk <= 0) {
        std::fprintf(stderr, "%s has invalid k_chunk=%d\n", name, W.k_chunk);
        return false;
    }
    if (W.num_k_chunks != ceil_div_int(W.cols, W.k_chunk)) {
        std::fprintf(stderr,
                     "%s has invalid num_k_chunks=%d for cols=%d k_chunk=%d\n",
                     name, W.num_k_chunks, W.cols, W.k_chunk);
        return false;
    }
    if (W.fp8_format != Fp8Format::E4M3 && W.fp8_format != Fp8Format::E5M2) {
        std::fprintf(stderr,
                     "%s has invalid fp8_format=%d\n",
                     name,
                     fp8_format_to_int(W.fp8_format));
        return false;
    }
    return true;
}

bool validate_shape_for_matrix(
    const MlpShape& shape,
    const PackedRowMajorMatrix& W,
    const char* shape_name,
    const char* matrix_name) {
    if (shape.num_tokens <= 0) {
        std::fprintf(stderr, "%s has invalid num_tokens=%d\n",
                     shape_name, shape.num_tokens);
        return false;
    }
    if (shape.rows_per_cta <= 0) {
        std::fprintf(stderr, "%s has invalid rows_per_cta=%d\n",
                     shape_name, shape.rows_per_cta);
        return false;
    }
    if (shape.k_chunk <= 0) {
        std::fprintf(stderr, "%s has invalid k_chunk=%d\n",
                     shape_name, shape.k_chunk);
        return false;
    }
    if (shape.k_chunk != W.k_chunk) {
        std::fprintf(stderr,
                     "%s k_chunk=%d does not match %s.k_chunk=%d\n",
                     shape_name, shape.k_chunk, matrix_name, W.k_chunk);
        return false;
    }
    if (shape.fp8_format != W.fp8_format) {
        std::fprintf(stderr,
                     "%s fp8_format=%d does not match %s.fp8_format=%d\n",
                     shape_name,
                     fp8_format_to_int(shape.fp8_format),
                     matrix_name,
                     fp8_format_to_int(W.fp8_format));
        return false;
    }
    return true;
}

__global__ void cast_half_to_float_kernel(const __half* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __half2float(in[idx]);
}

bool launch_cast_half_to_float(const __half* d_in,
                               float* d_out,
                               int n,
                               cudaStream_t stream) {
    if (!d_in || !d_out || n < 0) return false;
    if (n == 0) return true;
    const int threads = 256;
    const int blocks = ceil_div_int(n, threads);
    cast_half_to_float_kernel<<<blocks, threads, 0, stream>>>(d_in, d_out, n);
    return cudaGetLastError() == cudaSuccess;
}

#if defined(EXPERT_ENABLE_MLP_PROFILE) && EXPERT_ENABLE_MLP_PROFILE
struct MlpProfileEvents {
    cudaEvent_t e0 = nullptr, e1 = nullptr, e2 = nullptr, e3 = nullptr, e4 = nullptr, e5 = nullptr, e6 = nullptr;
    bool create() {
        return cudaEventCreate(&e0) == cudaSuccess &&
               cudaEventCreate(&e1) == cudaSuccess &&
               cudaEventCreate(&e2) == cudaSuccess &&
               cudaEventCreate(&e3) == cudaSuccess &&
               cudaEventCreate(&e4) == cudaSuccess &&
               cudaEventCreate(&e5) == cudaSuccess &&
               cudaEventCreate(&e6) == cudaSuccess;
    }
    void destroy() {
        if (e0) cudaEventDestroy(e0);
        if (e1) cudaEventDestroy(e1);
        if (e2) cudaEventDestroy(e2);
        if (e3) cudaEventDestroy(e3);
        if (e4) cudaEventDestroy(e4);
        if (e5) cudaEventDestroy(e5);
        if (e6) cudaEventDestroy(e6);
        e0 = e1 = e2 = e3 = e4 = e5 = e6 = nullptr;
    }
    ~MlpProfileEvents() { destroy(); }
};
#endif

} // namespace

#if defined(EXPERT_ENABLE_MLP_PROFILE) && EXPERT_ENABLE_MLP_PROFILE
#define EXPERT_MLP_PROFILE 1
#else
#define EXPERT_MLP_PROFILE 0
#endif

bool launch_mlp(const DeviceMlpView& mlp,
                const __half* d_input,
                __half* d_output,
                float* d_workspace,
                const MlpShape& shape,
                cudaStream_t stream) {
    if (shape.num_tokens <= 0) {
        std::fprintf(stderr, "launch_mlp: invalid num_tokens=%d\n", shape.num_tokens);
        return false;
    }
    if (shape.hidden_dim <= 0 || shape.inter_dim <= 0) {
        std::fprintf(stderr,
                     "launch_mlp: invalid dims hidden=%d inter=%d\n",
                     shape.hidden_dim, shape.inter_dim);
        return false;
    }
    if (!d_input || !d_output || !d_workspace) {
        std::fprintf(stderr, "launch_mlp: null pointer\n");
        return false;
    }

    if (!validate_matrix(mlp.w_up,   shape.inter_dim,  shape.hidden_dim, "w_up"))   return false;
    if (!validate_matrix(mlp.w_gate, shape.inter_dim,  shape.hidden_dim, "w_gate")) return false;
    if (!validate_matrix(mlp.w_down, shape.hidden_dim, shape.inter_dim,  "w_down")) return false;

    if (!validate_shape_for_matrix(shape, mlp.w_up,   "shape", "w_up"))   return false;
    if (!validate_shape_for_matrix(shape, mlp.w_gate, "shape", "w_gate")) return false;
    if (!validate_shape_for_matrix(shape, mlp.w_down, "shape", "w_down")) return false;

    if (!initialize(stream)) {
        std::fprintf(stderr, "launch_mlp: initialize failed\n");
        return false;
    }

    const int num_tokens = shape.num_tokens;
    const int hidden_dim = shape.hidden_dim;
    const int inter_dim = shape.inter_dim;

    float* d_up = workspace_ptr(d_workspace, workspace_up_offset_bytes(shape));
    float* d_gate = workspace_ptr(d_workspace, workspace_gate_offset_bytes(shape));
    float* d_fused = workspace_ptr(d_workspace, workspace_fused_offset_bytes(shape));
    float* d_outf = workspace_ptr(d_workspace, workspace_outf_offset_bytes(shape));

    __half* d_up_half = nullptr;
    __half* d_gate_half = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_up_half),
                                 static_cast<size_t>(num_tokens) * inter_dim * sizeof(__half));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "launch_mlp: cudaMalloc d_up_half failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&d_gate_half),
                     static_cast<size_t>(num_tokens) * inter_dim * sizeof(__half));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "launch_mlp: cudaMalloc d_gate_half failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_up_half);
        return false;
    }

#if EXPERT_MLP_PROFILE
    MlpProfileEvents prof;
    if (!prof.create()) {
        std::fprintf(stderr, "launch_mlp: failed to create profile events\n");
        cudaFree(d_up_half);
        cudaFree(d_gate_half);
        return false;
    }
    cudaEventRecord(prof.e0, stream);
#endif

    if (!launch_matvec_decode(mlp.w_up, d_input, d_up_half, shape, stream)) {
        std::fprintf(stderr, "launch_mlp: launch_matvec_decode(w_up) failed\n");
        cudaFree(d_up_half);
        cudaFree(d_gate_half);
        return false;
    }
#if EXPERT_MLP_PROFILE
    cudaEventRecord(prof.e1, stream);
#endif

    if (!launch_matvec_decode(mlp.w_gate, d_input, d_gate_half, shape, stream)) {
        std::fprintf(stderr, "launch_mlp: launch_matvec_decode(w_gate) failed\n");
        cudaFree(d_up_half);
        cudaFree(d_gate_half);
        return false;
    }
#if EXPERT_MLP_PROFILE
    cudaEventRecord(prof.e2, stream);
#endif

    if (!launch_cast_half_to_float(d_up_half, d_up, num_tokens * inter_dim, stream)) {
        std::fprintf(stderr, "launch_mlp: cast up half->float failed\n");
        cudaFree(d_up_half);
        cudaFree(d_gate_half);
        return false;
    }
    if (!launch_cast_half_to_float(d_gate_half, d_gate, num_tokens * inter_dim, stream)) {
        std::fprintf(stderr, "launch_mlp: cast gate half->float failed\n");
        cudaFree(d_up_half);
        cudaFree(d_gate_half);
        return false;
    }

    if (!launch_silu_mul_local(d_up, d_gate, d_fused, num_tokens, inter_dim, stream)) {
        std::fprintf(stderr, "launch_mlp: launch_silu_mul failed\n");
        cudaFree(d_up_half);
        cudaFree(d_gate_half);
        return false;
    }
#if EXPERT_MLP_PROFILE
    cudaEventRecord(prof.e3, stream);
#endif

    MlpShape down_shape = shape;
    down_shape.hidden_dim = shape.inter_dim;

    if (!launch_matvec_decode_from_float(mlp.w_down, d_fused, d_outf, down_shape, stream)) {
        std::fprintf(stderr, "launch_mlp: launch_matvec_decode_from_float(w_down) failed\n");
        cudaFree(d_up_half);
        cudaFree(d_gate_half);
        return false;
    }
#if EXPERT_MLP_PROFILE
    cudaEventRecord(prof.e4, stream);
#endif

    if (!launch_cast_float_to_half_local(d_outf, d_output, num_tokens * hidden_dim, stream)) {
        std::fprintf(stderr, "launch_mlp: launch_cast_float_to_half failed\n");
        cudaFree(d_up_half);
        cudaFree(d_gate_half);
        return false;
    }

#if EXPERT_MLP_PROFILE
    cudaEventRecord(prof.e5, stream);
    cudaEventSynchronize(prof.e5);

    float up_ms = 0.0f, gate_ms = 0.0f, fuse_ms = 0.0f, down_ms = 0.0f, cast_ms = 0.0f, total_ms = 0.0f;
    cudaEventElapsedTime(&up_ms,   prof.e0, prof.e1);
    cudaEventElapsedTime(&gate_ms, prof.e1, prof.e2);
    cudaEventElapsedTime(&fuse_ms, prof.e2, prof.e3);
    cudaEventElapsedTime(&down_ms, prof.e3, prof.e4);
    cudaEventElapsedTime(&cast_ms, prof.e4, prof.e5);
    cudaEventElapsedTime(&total_ms, prof.e0, prof.e5);

    std::fprintf(stderr,
                 "[mlp] up_ms=%.6f gate_ms=%.6f fuse_ms=%.6f down_ms=%.6f cast_ms=%.6f total_ms=%.6f\n",
                 up_ms, gate_ms, fuse_ms, down_ms, cast_ms, total_ms);
#endif

    cudaFree(d_up_half);
    cudaFree(d_gate_half);
    return true;
}
