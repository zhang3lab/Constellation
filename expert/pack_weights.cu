#include "expert.h"

#include <cstdio>
#include <cmath>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace expert {
namespace {

constexpr int N_TILE = kPackedNTile;  // 128
constexpr int K_TILE = kPackedKTile;  // 64

__host__ __device__ inline int ceil_div_int(int a, int b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__ float fp8_max_finite(int fp8_format) {
    return (fp8_format == FP8_E5M2) ? 57344.0f : 240.0f;
}

__device__ __forceinline__ uint8_t encode_fp8_generic(
    float x,
    int exp_bits,
    int mant_bits,
    int bias) {
    const int sign_bit = (x < 0.0f) ? 1 : 0;
    float ax = fabsf(x);

    if (isnan(ax)) {
        const int exp_all_ones = (1 << exp_bits) - 1;
        return static_cast<uint8_t>((sign_bit << 7) |
                                    (exp_all_ones << mant_bits) |
                                    1);
    }

    if (isinf(ax)) {
        const int exp_all_ones = (1 << exp_bits) - 1;
        return static_cast<uint8_t>((sign_bit << 7) |
                                    (exp_all_ones << mant_bits));
    }

    if (ax == 0.0f) {
        return static_cast<uint8_t>(sign_bit << 7);
    }

    const int exp_all_ones = (1 << exp_bits) - 1;
    const float min_normal = ldexpf(1.0f, 1 - bias);
    const float sub_step = ldexpf(1.0f, 1 - bias - mant_bits);

    if (ax < min_normal) {
        int mant = __float2int_rn(ax / sub_step);
        const int mant_max = (1 << mant_bits) - 1;
        if (mant < 0) mant = 0;
        if (mant > mant_max) mant = mant_max;
        return static_cast<uint8_t>((sign_bit << 7) | mant);
    }

    int e = static_cast<int>(floorf(log2f(ax)));
    float base = ldexpf(1.0f, e);
    float frac = ax / base - 1.0f;

    int mant = __float2int_rn(frac * static_cast<float>(1 << mant_bits));
    if (mant == (1 << mant_bits)) {
        mant = 0;
        e += 1;
    }

    int exp_field = e + bias;
    if (exp_field >= exp_all_ones) {
        return static_cast<uint8_t>((sign_bit << 7) |
                                    (exp_all_ones << mant_bits));
    }

    if (exp_field <= 0) {
        int sub_mant = __float2int_rn(ax / sub_step);
        const int mant_max = (1 << mant_bits) - 1;
        if (sub_mant < 0) sub_mant = 0;
        if (sub_mant > mant_max) sub_mant = mant_max;
        return static_cast<uint8_t>((sign_bit << 7) | sub_mant);
    }

    return static_cast<uint8_t>((sign_bit << 7) |
                                (exp_field << mant_bits) |
                                mant);
}

__device__ __forceinline__ uint8_t encode_fp8(float x, int fp8_format) {
    if (fp8_format == FP8_E5M2) {
        return encode_fp8_generic(x, 5, 2, 15);
    }
    return encode_fp8_generic(x, 4, 3, 7);
}

// One block packs one tile.
// blockDim.x = N_TILE = 128
__global__ void pack_tile_kernel(
    const half* __restrict__ d_src,      // [rows, cols], row-major
    float* __restrict__ d_dst_scales,    // [num_tiles]
    uint8_t* __restrict__ d_dst_weights, // [num_tiles * N_TILE * K_TILE]
    int rows,
    int cols,
    int fp8_format) {
    const int tile_id = blockIdx.x;
    const int row_in_tile = threadIdx.x;  // 0..127

    const int num_k_tiles = ceil_div_int(cols, K_TILE);
    const int out_tile = tile_id / num_k_tiles;
    const int k_tile_id = tile_id % num_k_tiles;

    const int row0 = out_tile * N_TILE;
    const int col0 = k_tile_id * K_TILE;
    const int row = row0 + row_in_tile;

    __shared__ float sh_row_max[N_TILE];
    __shared__ float sh_scale;

    float local_max = 0.0f;

    if (row < rows) {
#pragma unroll
        for (int kk = 0; kk < K_TILE; ++kk) {
            const int col = col0 + kk;
            if (col < cols) {
                const float v = fabsf(__half2float(d_src[row * cols + col]));
                if (v > local_max) local_max = v;
            }
        }
    }

    sh_row_max[row_in_tile] = local_max;
    __syncthreads();

    for (int stride = N_TILE / 2; stride > 0; stride >>= 1) {
        if (row_in_tile < stride) {
            if (sh_row_max[row_in_tile + stride] > sh_row_max[row_in_tile]) {
                sh_row_max[row_in_tile] = sh_row_max[row_in_tile + stride];
            }
        }
        __syncthreads();
    }

    if (row_in_tile == 0) {
        const float max_abs = sh_row_max[0];
        sh_scale = (max_abs > 0.0f) ? (max_abs / fp8_max_finite(fp8_format)) : 1.0f;
        d_dst_scales[tile_id] = sh_scale;
    }
    __syncthreads();

    const float scale = sh_scale;
    const size_t weights_base = static_cast<size_t>(tile_id) * N_TILE * K_TILE;
    const size_t row_base = weights_base + static_cast<size_t>(row_in_tile) * K_TILE;

    if (row < rows) {
#pragma unroll
        for (int kk = 0; kk < K_TILE; ++kk) {
            const int col = col0 + kk;
            uint8_t packed = 0;
            if (col < cols) {
                const float x = __half2float(d_src[row * cols + col]);
                const float qin = (scale > 0.0f) ? (x / scale) : 0.0f;
                packed = encode_fp8(qin, fp8_format);
            }
            d_dst_weights[row_base + kk] = packed;
        }
    } else {
#pragma unroll
        for (int kk = 0; kk < K_TILE; ++kk) {
            d_dst_weights[row_base + kk] = 0;
        }
    }
}

}  // namespace

bool quantize_and_pack_weight_tile_major_cuda(
    const half* d_src,
    float* d_dst_scales,
    uint8_t* d_dst_weights,
    int rows,
    int cols,
    int fp8_format,
    cudaStream_t stream) {
    if (d_src == nullptr || d_dst_scales == nullptr || d_dst_weights == nullptr) {
        std::fprintf(stderr, "quantize_and_pack_weight_tile_major_cuda: null pointer\n");
        return false;
    }
    if (rows <= 0 || cols <= 0) {
        std::fprintf(stderr,
                     "quantize_and_pack_weight_tile_major_cuda: bad shape rows=%d cols=%d\n",
                     rows, cols);
        return false;
    }
    if (fp8_format != FP8_E4M3 && fp8_format != FP8_E5M2) {
        std::fprintf(stderr,
                     "quantize_and_pack_weight_tile_major_cuda: bad fp8_format=%d\n",
                     fp8_format);
        return false;
    }

    const int num_tiles = static_cast<int>(packed_tile_num_tiles(rows, cols));
    dim3 block(N_TILE, 1, 1);
    dim3 grid(num_tiles, 1, 1);

    pack_tile_kernel<<<grid, block, 0, stream>>>(
        d_src, d_dst_scales, d_dst_weights, rows, cols, fp8_format);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "pack_tile_kernel launch failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }

    return true;
}

}  // namespace expert
