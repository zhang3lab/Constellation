// vector_ops.cu
//
// Simple vector primitives for the new clean MLP pipeline.
//
// Implements:
//   - launch_fuse_silu_mul()
//   - launch_cast_float_to_half()
//   - workspace sizing / offsets
//
// Workspace layout:
//   [up][gate][fused][out_f]
//
// All offsets are byte offsets.

#include "expert.h"

#include <cstdio>
#include <cmath>
#include <cstddef>
#include <climits>

namespace expert {
namespace {

constexpr int THREADS = 256;

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void fuse_silu_mul_kernel(
    const float* __restrict__ up,
    const float* __restrict__ gate,
    float* __restrict__ fused,
    size_t n) {
    const size_t idx =
        static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) +
        static_cast<size_t>(threadIdx.x);
    if (idx < n) {
        fused[idx] = up[idx] * silu(gate[idx]);
    }
}

__global__ void cast_float_to_half_kernel(
    const float* __restrict__ x,
    half* __restrict__ y,
    size_t n) {
    const size_t idx =
        static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) +
        static_cast<size_t>(threadIdx.x);
    if (idx < n) {
        y[idx] = __float2half(x[idx]);
    }
}

inline size_t float_buffer_bytes(int num_tokens, int dim) {
    return static_cast<size_t>(num_tokens) * static_cast<size_t>(dim) * sizeof(float);
}

} // namespace

size_t mlp_workspace_bytes(
    int max_tokens,
    int hidden_dim,
    int inter_dim) {
    if (max_tokens <= 0 || hidden_dim <= 0 || inter_dim <= 0) return 0;

    const size_t up_bytes    = float_buffer_bytes(max_tokens, inter_dim);
    const size_t gate_bytes  = float_buffer_bytes(max_tokens, inter_dim);
    const size_t fused_bytes = float_buffer_bytes(max_tokens, inter_dim);
    const size_t outf_bytes  = float_buffer_bytes(max_tokens, hidden_dim);

    return up_bytes + gate_bytes + fused_bytes + outf_bytes;
}

size_t workspace_up_offset_bytes(
    int max_tokens,
    int hidden_dim,
    int inter_dim) {
    (void)max_tokens;
    (void)hidden_dim;
    (void)inter_dim;
    // up buffer starts at workspace base
    return 0;
}

size_t workspace_gate_offset_bytes(
    int max_tokens,
    int hidden_dim,
    int inter_dim) {
    return workspace_up_offset_bytes(max_tokens, hidden_dim, inter_dim)
         + float_buffer_bytes(max_tokens, inter_dim);
}

size_t workspace_fused_offset_bytes(
    int max_tokens,
    int hidden_dim,
    int inter_dim) {
    return workspace_gate_offset_bytes(max_tokens, hidden_dim, inter_dim)
         + float_buffer_bytes(max_tokens, inter_dim);
}

size_t workspace_outf_offset_bytes(
    int max_tokens,
    int hidden_dim,
    int inter_dim) {
    return workspace_fused_offset_bytes(max_tokens, hidden_dim, inter_dim)
         + float_buffer_bytes(max_tokens, inter_dim);
}

bool launch_fuse_silu_mul(
    const float* d_up,
    const float* d_gate,
    float* d_fused,
    int num_tokens,
    int dim,
    cudaStream_t stream) {
    if (!d_up || !d_gate || !d_fused) {
        std::fprintf(stderr, "launch_fuse_silu_mul: null pointer\n");
        return false;
    }
    if (num_tokens <= 0 || dim <= 0) {
        std::fprintf(stderr, "launch_fuse_silu_mul: invalid dims\n");
        return false;
    }

    const size_t n = static_cast<size_t>(num_tokens) * static_cast<size_t>(dim);
    if (n > static_cast<size_t>(INT_MAX) * THREADS) {
        std::fprintf(stderr, "launch_fuse_silu_mul: size too large\n");
        return false;
    }

    const dim3 block(THREADS, 1, 1);
    const dim3 grid(ceil_div_int(static_cast<int>(n), THREADS), 1, 1);

    fuse_silu_mul_kernel<<<grid, block, 0, stream>>>(d_up, d_gate, d_fused, n);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "fuse_silu_mul_kernel launch failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool launch_cast_float_to_half(
    const float* d_x,
    half* d_y,
    int num_tokens,
    int dim,
    cudaStream_t stream) {
    if (!d_x || !d_y) {
        std::fprintf(stderr, "launch_cast_float_to_half: null pointer\n");
        return false;
    }
    if (num_tokens <= 0 || dim <= 0) {
        std::fprintf(stderr, "launch_cast_float_to_half: invalid dims\n");
        return false;
    }

    const size_t n = static_cast<size_t>(num_tokens) * static_cast<size_t>(dim);
    if (n > static_cast<size_t>(INT_MAX) * THREADS) {
        std::fprintf(stderr, "launch_cast_float_to_half: size too large\n");
        return false;
    }

    const dim3 block(THREADS, 1, 1);
    const dim3 grid(ceil_div_int(static_cast<int>(n), THREADS), 1, 1);

    cast_float_to_half_kernel<<<grid, block, 0, stream>>>(d_x, d_y, n);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "cast_float_to_half_kernel launch failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }
    return true;
}

} // namespace expert
