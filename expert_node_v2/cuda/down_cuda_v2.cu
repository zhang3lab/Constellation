#include "expert_node_v2/cuda/down_cuda_v2.h"

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

template <class TAct, int WARPS_PER_BLOCK>
__global__ void down_blockscale_kernel(
    int rows,
    int cols,
    const std::uint8_t* __restrict__ weights,
    int row_block,
    int col_block,
    int num_col_blocks,
    const float* __restrict__ scales,
    const float* __restrict__ h,
    TAct* __restrict__ y,
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
        sum += w * h[k];
    }

    sum = warp_sum(sum);
    if (lane == 0) {
        y[row] = float_to_act<TAct>(sum);
    }
}

}  // namespace

template <class TAct>
bool LaunchDownCudaV2Impl(
    const MatrixBlockScaleViewV2& w_down_device_view,
    const float* d_h,
    TAct* d_y,
    cudaStream_t stream) {
    if (d_h == nullptr || d_y == nullptr) return false;

    const int rows = w_down_device_view.matrix.rows;
    const int cols = w_down_device_view.matrix.cols;
    if (rows <= 0 || cols <= 0) return false;

    if (w_down_device_view.weight.data.empty() ||
        w_down_device_view.scale.data.empty()) {
        return false;
    }

    int device_id = -1;
    cudaError_t err = cudaGetDevice(&device_id);
    if (err != cudaSuccess) {
        return false;
    }

    const float* lut =
        GetOrInitFp8DecodeLutCudaV2(device_id, w_down_device_view.matrix.fp8_format, stream);
    if (lut == nullptr) {
        return false;
    }

    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;
    const int blocks = (rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    down_blockscale_kernel<TAct, WARPS_PER_BLOCK>
        <<<blocks, THREADS, 0, stream>>>(
            rows,
            cols,
            w_down_device_view.weight.data.data(),
            w_down_device_view.scale_meta.row_block,
            w_down_device_view.scale_meta.col_block,
            w_down_device_view.scale_meta.num_col_blocks,
            w_down_device_view.scale.data.data(),
            d_h,
            d_y,
            lut);

    err = cudaGetLastError();
    return err == cudaSuccess;
}

template bool LaunchDownCudaV2Impl<__half>(
    const MatrixBlockScaleViewV2&,
    const float*,
    __half*,
    cudaStream_t);

#if EXPERT_NODE_V2_HAS_CUDA_BF16
template bool LaunchDownCudaV2Impl<__nv_bfloat16>(
    const MatrixBlockScaleViewV2&,
    const float*,
    __nv_bfloat16*,
    cudaStream_t);
#endif
