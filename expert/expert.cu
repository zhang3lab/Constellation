#include "expert.h"

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace expert {
namespace {

constexpr int kTileK = 32;
constexpr int kTileN = 32;
constexpr int kTileM = 4;

static_assert(kTileK == 32, "current kernel assumes kTileK == 32");
static_assert(kTileN == 32, "current kernel assumes kTileN == 32");

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

// Per-row grouped scales:
// group = row * ceil(cols / group_size) + col / group_size
__device__ __forceinline__ float load_dequant(
    const QuantMatrix& w,
    int row,
    int col) {
    const int idx = row * w.cols + col;
    const uint8_t packed = w.data[idx];

    float scale = 1.0f;
    if (w.scales != nullptr && w.group_size > 0) {
        const int groups_per_row = (w.cols + w.group_size - 1) / w.group_size;
        const int group = row * groups_per_row + (col / w.group_size);
        scale = w.scales[group];
    }

    return decode_fp8_byte(packed, w.fp8_format) * scale;
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// Stage 1:
//   up   = X @ W_up^T
//   gate = X @ W_gate^T
//   fused = silu(gate) * up
//
// X      : [num_tokens, hidden_dim]
// W_up   : [inter_dim, hidden_dim]
// W_gate : [inter_dim, hidden_dim]
// fused  : [num_tokens, inter_dim]
__global__ void up_gate_fused_kernel(
    QuantMatrix w_up,
    QuantMatrix w_gate,
    const half* __restrict__ input,
    float* __restrict__ fused,
    int num_tokens,
    int hidden_dim,
    int inter_dim) {
    __shared__ half x_sh[kTileM][kTileK];

    const int inter_idx = blockIdx.x * kTileN + threadIdx.x;
    const int token_idx = blockIdx.y * kTileM + threadIdx.y;

    float up_acc = 0.0f;
    float gate_acc = 0.0f;

    for (int k0 = 0; k0 < hidden_dim; k0 += kTileK) {
        const int k_load = k0 + threadIdx.x;

        if (token_idx < num_tokens) {
            x_sh[threadIdx.y][threadIdx.x] =
                (k_load < hidden_dim) ? input[token_idx * hidden_dim + k_load]
                                      : __float2half(0.0f);
        }
        __syncthreads();

        if (token_idx < num_tokens && inter_idx < inter_dim) {
#pragma unroll
            for (int kk = 0; kk < kTileK; ++kk) {
                const int k = k0 + kk;
                if (k < hidden_dim) {
                    const float x = __half2float(x_sh[threadIdx.y][kk]);
                    const float wup = load_dequant(w_up, inter_idx, k);
                    const float wgate = load_dequant(w_gate, inter_idx, k);
                    up_acc += x * wup;
                    gate_acc += x * wgate;
                }
            }
        }

        __syncthreads();
    }

    if (token_idx < num_tokens && inter_idx < inter_dim) {
        fused[token_idx * inter_dim + inter_idx] = up_acc * silu(gate_acc);
    }
}

// Stage 2:
//   output = fused @ W_down^T
//
// fused   : [num_tokens, inter_dim]
// W_down  : [hidden_dim, inter_dim]
// output  : [num_tokens, hidden_dim]
__global__ void down_proj_kernel(
    QuantMatrix w_down,
    const float* __restrict__ fused,
    half* __restrict__ output,
    int num_tokens,
    int hidden_dim,
    int inter_dim) {
    __shared__ float fused_sh[kTileM][kTileK];

    const int out_idx = blockIdx.x * kTileN + threadIdx.x;
    const int token_idx = blockIdx.y * kTileM + threadIdx.y;

    float acc = 0.0f;

    for (int k0 = 0; k0 < inter_dim; k0 += kTileK) {
        const int k_load = k0 + threadIdx.x;

        if (token_idx < num_tokens) {
            fused_sh[threadIdx.y][threadIdx.x] =
                (k_load < inter_dim) ? fused[token_idx * inter_dim + k_load] : 0.0f;
        }
        __syncthreads();

        if (token_idx < num_tokens && out_idx < hidden_dim) {
#pragma unroll
            for (int kk = 0; kk < kTileK; ++kk) {
                const int k = k0 + kk;
                if (k < inter_dim) {
                    const float w = load_dequant(w_down, out_idx, k);
                    acc += fused_sh[threadIdx.y][kk] * w;
                }
            }
        }

        __syncthreads();
    }

    if (token_idx < num_tokens && out_idx < hidden_dim) {
        output[token_idx * hidden_dim + out_idx] = __float2half(acc);
    }
}

__host__ bool check_matrix_shape(
    const QuantMatrix& w,
    int rows,
    int cols,
    const char* name) {
    if (w.data == nullptr) {
        std::fprintf(stderr, "%s.data is null\n", name);
        return false;
    }
    if (w.rows != rows || w.cols != cols) {
        std::fprintf(stderr,
                     "%s shape mismatch: got [%d, %d], expected [%d, %d]\n",
                     name, w.rows, w.cols, rows, cols);
        return false;
    }
    if (w.group_size <= 0) {
        std::fprintf(stderr, "%s.group_size must be > 0, got %d\n", name, w.group_size);
        return false;
    }
    return true;
}

}  // namespace

bool launch_expert_mlp(
    const ExpertWeights& weights,
    const half* d_input,
    half* d_output,
    float* d_fused_workspace,
    int num_tokens,
    cudaStream_t stream) {
    if (num_tokens <= 0) {
        std::fprintf(stderr, "launch_expert_mlp: invalid num_tokens=%d\n", num_tokens);
        return false;
    }
    if (weights.hidden_dim <= 0 || weights.inter_dim <= 0) {
        std::fprintf(stderr,
                     "launch_expert_mlp: invalid dims hidden=%d inter=%d\n",
                     weights.hidden_dim, weights.inter_dim);
        return false;
    }
    if (d_input == nullptr || d_output == nullptr || d_fused_workspace == nullptr) {
        std::fprintf(stderr, "launch_expert_mlp: null input/output/workspace pointer\n");
        return false;
    }

    if (!check_matrix_shape(weights.w_up,   weights.inter_dim,  weights.hidden_dim, "w_up")) {
        return false;
    }
    if (!check_matrix_shape(weights.w_gate, weights.inter_dim,  weights.hidden_dim, "w_gate")) {
        return false;
    }
    if (!check_matrix_shape(weights.w_down, weights.hidden_dim, weights.inter_dim,  "w_down")) {
        return false;
    }

    dim3 block(kTileN, kTileM, 1);

    dim3 grid_up_gate(
        (weights.inter_dim + kTileN - 1) / kTileN,
        (num_tokens + kTileM - 1) / kTileM,
        1);

    up_gate_fused_kernel<<<grid_up_gate, block, 0, stream>>>(
        weights.w_up,
        weights.w_gate,
        d_input,
        d_fused_workspace,
        num_tokens,
        weights.hidden_dim,
        weights.inter_dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "up_gate_fused_kernel launch failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }

    dim3 grid_down(
        (weights.hidden_dim + kTileN - 1) / kTileN,
        (num_tokens + kTileM - 1) / kTileM,
        1);

    down_proj_kernel<<<grid_down, block, 0, stream>>>(
        weights.w_down,
        d_fused_workspace,
        d_output,
        num_tokens,
        weights.hidden_dim,
        weights.inter_dim);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "down_proj_kernel launch failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }

    return true;
}

}  // namespace expert
