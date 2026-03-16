#include "expert.h"

#include <cmath>
#include <cstdio>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace expert {
namespace {

constexpr int N_TILE = kPackedNTile;         // 128
constexpr int K_TILE = kPackedKTile;         // 64
constexpr int P = 8;                         // K partitions for tiny-batch path
constexpr int B_MAX = kTinyBatchMaxTokens;   // 8

__device__ __forceinline__ float decode_fp8_e4m3(uint8_t x) {
    const int sign = (x >> 7) & 0x1;
    const int exp  = (x >> 3) & 0xF;
    const int mant = x & 0x7;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        const float frac = static_cast<float>(mant) / 8.0f;
        const float val = ldexpf(frac, -6);
        return sign ? -val : val;
    }

    if (exp == 0xF) {
        if (mant == 0) return sign ? -CUDART_INF_F : CUDART_INF_F;
        return CUDART_NAN_F;
    }

    const float frac = 1.0f + static_cast<float>(mant) / 8.0f;
    const float val = ldexpf(frac, exp - 7);
    return sign ? -val : val;
}

__device__ __forceinline__ float decode_fp8_e5m2(uint8_t x) {
    const int sign = (x >> 7) & 0x1;
    const int exp  = (x >> 2) & 0x1F;
    const int mant = x & 0x3;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        const float frac = static_cast<float>(mant) / 4.0f;
        const float val = ldexpf(frac, -14);
        return sign ? -val : val;
    }

    if (exp == 0x1F) {
        if (mant == 0) return sign ? -CUDART_INF_F : CUDART_INF_F;
        return CUDART_NAN_F;
    }

    const float frac = 1.0f + static_cast<float>(mant) / 4.0f;
    const float val = ldexpf(frac, exp - 15);
    return sign ? -val : val;
}

__device__ __forceinline__ float decode_fp8_byte(uint8_t packed, int fp8_format) {
    if (fp8_format == FP8_E5M2) {
        return decode_fp8_e5m2(packed);
    }
    return decode_fp8_e4m3(packed);
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__host__ __device__ inline int ceil_div_int(int a, int b) {
    return (a + b - 1) / b;
}

__host__ bool check_packed_tile_matrix(
    const PackedTileMatrix& w,
    int rows,
    int cols,
    const char* name) {
    if (w.scales == nullptr || w.weights == nullptr) {
        std::fprintf(stderr, "%s has null pointer\n", name);
        return false;
    }
    if (w.rows != rows || w.cols != cols) {
        std::fprintf(stderr,
                     "%s shape mismatch: got [%d, %d], expected [%d, %d]\n",
                     name, w.rows, w.cols, rows, cols);
        return false;
    }
    if (w.fp8_format != FP8_E4M3 && w.fp8_format != FP8_E5M2) {
        std::fprintf(stderr, "%s bad fp8_format=%d\n", name, w.fp8_format);
        return false;
    }
    return true;
}

__device__ __forceinline__ size_t packed_tile_base(int tile_id) {
    return static_cast<size_t>(tile_id) * N_TILE * K_TILE;
}

__device__ __forceinline__ int tile_id_of(int out_tile, int k_tile_id, int num_k_tiles) {
    return out_tile * num_k_tiles + k_tile_id;
}

// -----------------------------------------------------------------------------
// Workspace layout:
//
// partial_up   : [B_MAX, inter_dim, P]
// partial_gate : [B_MAX, inter_dim, P]
// fused        : [B_MAX, inter_dim]
// partial_down : [B_MAX, hidden_dim, P]
// -----------------------------------------------------------------------------

__global__ void partial_up_gate_kernel(
    PackedTileMatrix w_up,
    PackedTileMatrix w_gate,
    const half* __restrict__ d_input,   // [num_tokens, hidden_dim]
    float* __restrict__ partial_up,     // [B_MAX, inter_dim, P]
    float* __restrict__ partial_gate,   // [B_MAX, inter_dim, P]
    int num_tokens,
    int hidden_dim,
    int inter_dim) {
    __shared__ float x_sh[K_TILE];

    const int out_tile = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int partition_id = blockIdx.z;
    const int row_in_tile = threadIdx.x;

    const int out_idx = out_tile * N_TILE + row_in_tile;
    const bool valid = (batch_idx < num_tokens && out_idx < inter_dim);

    const int partition_span = ceil_div_int(hidden_dim, P);
    const int k_begin = partition_id * partition_span;
    const int k_end = min(hidden_dim, k_begin + partition_span);

    float up_acc = 0.0f;
    float gate_acc = 0.0f;

    const int num_k_tiles = ceil_div_int(hidden_dim, K_TILE);
    const int first_k_tile = k_begin / K_TILE;
    const int last_k_tile_exclusive = ceil_div_int(k_end, K_TILE);

    for (int k_tile_id = first_k_tile; k_tile_id < last_k_tile_exclusive; ++k_tile_id) {
        const int k0 = k_tile_id * K_TILE;

        if (row_in_tile < K_TILE) {
            const int k = k0 + row_in_tile;
            x_sh[row_in_tile] =
                (batch_idx < num_tokens && k < hidden_dim && k >= k_begin && k < k_end)
                    ? __half2float(d_input[batch_idx * hidden_dim + k])
                    : 0.0f;
        }
        __syncthreads();

        if (valid) {
            const int tile_id = tile_id_of(out_tile, k_tile_id, num_k_tiles);

            const float up_tile_scale = w_up.scales[tile_id];
            const float gate_tile_scale = w_gate.scales[tile_id];

            const uint8_t* up_tile_weights =
                w_up.weights + packed_tile_base(tile_id);
            const uint8_t* gate_tile_weights =
                w_gate.weights + packed_tile_base(tile_id);

            const size_t row_base = static_cast<size_t>(row_in_tile) * K_TILE;

#pragma unroll
            for (int kk = 0; kk < K_TILE; ++kk) {
                const int k = k0 + kk;
                if (k >= k_begin && k < k_end && k < hidden_dim) {
                    const float x = x_sh[kk];
                    const uint8_t up_packed = up_tile_weights[row_base + kk];
                    const uint8_t gate_packed = gate_tile_weights[row_base + kk];

                    up_acc += x * (decode_fp8_byte(up_packed, w_up.fp8_format) * up_tile_scale);
                    gate_acc += x * (decode_fp8_byte(gate_packed, w_gate.fp8_format) * gate_tile_scale);
                }
            }
        }

        __syncthreads();
    }

    if (valid) {
        const size_t base = ((batch_idx * inter_dim + out_idx) * P) + partition_id;
        partial_up[base] = up_acc;
        partial_gate[base] = gate_acc;
    }
}

__global__ void reduce_fuse_kernel(
    const float* __restrict__ partial_up,
    const float* __restrict__ partial_gate,
    float* __restrict__ fused,
    int num_tokens,
    int inter_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_tokens * inter_dim;
    if (idx >= total) return;

    const int batch_idx = idx / inter_dim;
    const int inter_idx = idx % inter_dim;

    float up = 0.0f;
    float gate = 0.0f;

#pragma unroll
    for (int p = 0; p < P; ++p) {
        up += partial_up[((batch_idx * inter_dim + inter_idx) * P) + p];
        gate += partial_gate[((batch_idx * inter_dim + inter_idx) * P) + p];
    }

    fused[batch_idx * inter_dim + inter_idx] = up * silu(gate);
}

__global__ void partial_down_kernel(
    PackedTileMatrix w_down,
    const float* __restrict__ fused,    // [num_tokens, inter_dim]
    float* __restrict__ partial_down,   // [B_MAX, hidden_dim, P]
    int num_tokens,
    int hidden_dim,
    int inter_dim) {
    __shared__ float fused_sh[K_TILE];

    const int out_tile = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int partition_id = blockIdx.z;
    const int row_in_tile = threadIdx.x;

    const int out_idx = out_tile * N_TILE + row_in_tile;
    const bool valid = (batch_idx < num_tokens && out_idx < hidden_dim);

    const int partition_span = ceil_div_int(inter_dim, P);
    const int k_begin = partition_id * partition_span;
    const int k_end = min(inter_dim, k_begin + partition_span);

    float acc = 0.0f;

    const int num_k_tiles = ceil_div_int(inter_dim, K_TILE);
    const int first_k_tile = k_begin / K_TILE;
    const int last_k_tile_exclusive = ceil_div_int(k_end, K_TILE);

    for (int k_tile_id = first_k_tile; k_tile_id < last_k_tile_exclusive; ++k_tile_id) {
        const int k0 = k_tile_id * K_TILE;

        if (row_in_tile < K_TILE) {
            const int k = k0 + row_in_tile;
            fused_sh[row_in_tile] =
                (batch_idx < num_tokens && k < inter_dim && k >= k_begin && k < k_end)
                    ? fused[batch_idx * inter_dim + k]
                    : 0.0f;
        }
        __syncthreads();

        if (valid) {
            const int tile_id = tile_id_of(out_tile, k_tile_id, num_k_tiles);
            const float tile_scale = w_down.scales[tile_id];
            const uint8_t* tile_weights = w_down.weights + packed_tile_base(tile_id);
            const size_t row_base = static_cast<size_t>(row_in_tile) * K_TILE;

#pragma unroll
            for (int kk = 0; kk < K_TILE; ++kk) {
                const int k = k0 + kk;
                if (k >= k_begin && k < k_end && k < inter_dim) {
                    const uint8_t packed = tile_weights[row_base + kk];
                    acc += fused_sh[kk] * (decode_fp8_byte(packed, w_down.fp8_format) * tile_scale);
                }
            }
        }

        __syncthreads();
    }

    if (valid) {
        partial_down[((batch_idx * hidden_dim + out_idx) * P) + partition_id] = acc;
    }
}

__global__ void reduce_down_kernel(
    const float* __restrict__ partial_down,
    half* __restrict__ d_output,
    int num_tokens,
    int hidden_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_tokens * hidden_dim;
    if (idx >= total) return;

    const int batch_idx = idx / hidden_dim;
    const int hidden_idx = idx % hidden_dim;

    float sum = 0.0f;
#pragma unroll
    for (int p = 0; p < P; ++p) {
        sum += partial_down[((batch_idx * hidden_dim + hidden_idx) * P) + p];
    }

    d_output[batch_idx * hidden_dim + hidden_idx] = __float2half(sum);
}

}  // namespace

size_t workspace_bytes_for_tiny(
    int max_tokens,
    int hidden_dim,
    int inter_dim) {
    const size_t elems =
        2ull * B_MAX * static_cast<size_t>(inter_dim) * P +
        1ull * B_MAX * static_cast<size_t>(inter_dim) +
        1ull * B_MAX * static_cast<size_t>(hidden_dim) * P;
    return elems * sizeof(float);
}

bool launch_expert_mlp_tiny(
    const PackedTileExpertWeights& weights,
    const half* d_input,
    half* d_output,
    float* d_workspace,
    int num_tokens,
    cudaStream_t stream) {
    if (num_tokens <= 0 || num_tokens > B_MAX) {
        std::fprintf(stderr,
                     "launch_expert_mlp_tiny: bad num_tokens=%d (B_MAX=%d)\n",
                     num_tokens, B_MAX);
        return false;
    }
    if (weights.hidden_dim <= 0 || weights.inter_dim <= 0) {
        std::fprintf(stderr,
                     "launch_expert_mlp_tiny: invalid dims hidden=%d inter=%d\n",
                     weights.hidden_dim, weights.inter_dim);
        return false;
    }
    if (d_input == nullptr || d_output == nullptr || d_workspace == nullptr) {
        std::fprintf(stderr, "launch_expert_mlp_tiny: null pointer\n");
        return false;
    }

    if (!check_packed_tile_matrix(weights.w_up, weights.inter_dim, weights.hidden_dim, "w_up")) {
        return false;
    }
    if (!check_packed_tile_matrix(weights.w_gate, weights.inter_dim, weights.hidden_dim, "w_gate")) {
        return false;
    }
    if (!check_packed_tile_matrix(weights.w_down, weights.hidden_dim, weights.inter_dim, "w_down")) {
        return false;
    }

    const int hidden_dim = weights.hidden_dim;
    const int inter_dim = weights.inter_dim;

    float* partial_up   = d_workspace;
    float* partial_gate = partial_up   + static_cast<size_t>(B_MAX) * inter_dim * P;
    float* fused        = partial_gate + static_cast<size_t>(B_MAX) * inter_dim * P;
    float* partial_down = fused        + static_cast<size_t>(B_MAX) * inter_dim;

    cudaError_t err = cudaSuccess;

    {
        dim3 block(N_TILE, 1, 1);
        dim3 grid(ceil_div_int(inter_dim, N_TILE), num_tokens, P);

        partial_up_gate_kernel<<<grid, block, 0, stream>>>(
            weights.w_up,
            weights.w_gate,
            d_input,
            partial_up,
            partial_gate,
            num_tokens,
            hidden_dim,
            inter_dim);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                         "partial_up_gate_kernel launch failed: %s\n",
                         cudaGetErrorString(err));
            return false;
        }
    }

    {
        const int total = num_tokens * inter_dim;
        dim3 block(256, 1, 1);
        dim3 grid(ceil_div_int(total, 256), 1, 1);

        reduce_fuse_kernel<<<grid, block, 0, stream>>>(
            partial_up, partial_gate, fused, num_tokens, inter_dim);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                         "reduce_fuse_kernel launch failed: %s\n",
                         cudaGetErrorString(err));
            return false;
        }
    }

    {
        dim3 block(N_TILE, 1, 1);
        dim3 grid(ceil_div_int(hidden_dim, N_TILE), num_tokens, P);

        partial_down_kernel<<<grid, block, 0, stream>>>(
            weights.w_down, fused, partial_down, num_tokens, hidden_dim, inter_dim);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                         "partial_down_kernel launch failed: %s\n",
                         cudaGetErrorString(err));
            return false;
        }
    }

    {
        const int total = num_tokens * hidden_dim;
        dim3 block(256, 1, 1);
        dim3 grid(ceil_div_int(total, 256), 1, 1);

        reduce_down_kernel<<<grid, block, 0, stream>>>(
            partial_down, d_output, num_tokens, hidden_dim);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                         "reduce_down_kernel launch failed: %s\n",
                         cudaGetErrorString(err));
            return false;
        }
    }

    return true;
}

}  // namespace expert
