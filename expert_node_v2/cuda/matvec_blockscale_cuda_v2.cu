#include "expert_node_v2/cuda/matvec_blockscale_cuda_v2.h"

#include <cuda_fp16.h>
#if EXPERT_NODE_V2_HAS_CUDA_BF16
#include <cuda_bf16.h>
#endif
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "expert_node_v2/cuda/cuda_common_v2.cuh"
#include "expert_node_v2/cuda/fp8_decode_lut_v2.h"

namespace {

template <class TIn, class TOut, int WARPS_PER_BLOCK>
__global__ void matvec_blockscale_kernel(
    int rows,
    int cols,
    const std::uint8_t* __restrict__ weights,
    int row_block,
    int col_block,
    int num_col_blocks,
    const float* __restrict__ scales,
    const TIn* __restrict__ x,
    TOut* __restrict__ y,
    const float* __restrict__ lut) {
    constexpr int WARP_SIZE = 32;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    if (row >= rows) return;

    const int rb = row / row_block;
    float sum = 0.0f;

    for (int k = lane; k < cols; k += WARP_SIZE) {
        const int cb = k / col_block;

        const std::size_t w_idx =
            static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) +
            static_cast<std::size_t>(k);
        const std::size_t s_idx =
            static_cast<std::size_t>(rb) *
                static_cast<std::size_t>(num_col_blocks) +
            static_cast<std::size_t>(cb);

        const float scale = scales[s_idx];
        const float w = lut[weights[w_idx]] * scale;
        const float xv = act_to_float<TIn>(x[k]);
        sum += w * xv;
    }

    sum = warp_sum(sum);
    if (lane == 0) {
        y[row] = float_to_act<TOut>(sum);
    }
}

}  // namespace

template <class TIn, class TOut>
bool LaunchMatvecBlockScaleCudaV2(
    const MatrixBlockScaleViewV2& W,
    const TIn* d_x,
    TOut* d_y,
    cudaStream_t stream) {
    if (d_x == nullptr || d_y == nullptr) return false;
    if (W.matrix.rows <= 0 || W.matrix.cols <= 0) return false;
    if (W.weight.data.empty() || W.scale.data.empty()) return false;

    int device_id = -1;
    cudaError_t err = cudaGetDevice(&device_id);
    if (err != cudaSuccess) {
        return false;
    }

    const float* lut = GetOrInitFp8DecodeLutCudaV2(device_id, W.matrix.fp8_format, stream);
    if (lut == nullptr) {
        return false;
    }

    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;
    const int blocks = (W.matrix.rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    matvec_blockscale_kernel<TIn, TOut, WARPS_PER_BLOCK>
        <<<blocks, THREADS, 0, stream>>>(
            W.matrix.rows,
            W.matrix.cols,
            W.weight.data.data(),
            W.scale_meta.row_block,
            W.scale_meta.col_block,
            W.scale_meta.num_col_blocks,
            W.scale.data.data(),
            d_x,
            d_y,
            lut);

    err = cudaGetLastError();
    return err == cudaSuccess;
}

template bool LaunchMatvecBlockScaleCudaV2<float, float>(
    const MatrixBlockScaleViewV2&,
    const float*,
    float*,
    cudaStream_t);

template bool LaunchMatvecBlockScaleCudaV2<float, __half>(
    const MatrixBlockScaleViewV2&,
    const float*,
    __half*,
    cudaStream_t);

template bool LaunchMatvecBlockScaleCudaV2<__half, float>(
    const MatrixBlockScaleViewV2&,
    const __half*,
    float*,
    cudaStream_t);

template bool LaunchMatvecBlockScaleCudaV2<__half, __half>(
    const MatrixBlockScaleViewV2&,
    const __half*,
    __half*,
    cudaStream_t);

#if EXPERT_NODE_V2_HAS_CUDA_BF16
template bool LaunchMatvecBlockScaleCudaV2<float, __nv_bfloat16>(
    const MatrixBlockScaleViewV2&,
    const float*,
    __nv_bfloat16*,
    cudaStream_t);

template bool LaunchMatvecBlockScaleCudaV2<__nv_bfloat16, float>(
    const MatrixBlockScaleViewV2&,
    const __nv_bfloat16*,
    float*,
    cudaStream_t);

template bool LaunchMatvecBlockScaleCudaV2<__nv_bfloat16, __nv_bfloat16>(
    const MatrixBlockScaleViewV2&,
    const __nv_bfloat16*,
    __nv_bfloat16*,
    cudaStream_t);
#endif
