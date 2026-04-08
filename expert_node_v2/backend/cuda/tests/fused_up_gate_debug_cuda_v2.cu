#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if EXPERT_NODE_V2_ENABLE_CUDA_BF16
#include <cuda_bf16.h>
#endif

#include <cstdint>

#include "expert_node_v2/backend/cuda/fp8_decode_lut_v2.h"
#include "expert_node_v2/expert_format_v2.h"

struct UpDebugItemCudaV2 {
    int k;
    int cb;
    std::size_t w_idx;
    std::size_t s_idx;
    std::uint8_t w_byte;
    float scale;
    float decoded;
    float x_val;
    float contrib;
};

namespace {

template <class TAct>
__device__ inline float debug_act_to_float(TAct x);

template <>
__device__ inline float debug_act_to_float<__half>(__half x) {
    return __half2float(x);
}

#if EXPERT_NODE_V2_ENABLE_CUDA_BF16
template <>
__device__ inline float debug_act_to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}
#endif

template <class TAct>
__global__ void debug_up_row0_kernel(
    int cols,
    const std::uint8_t* __restrict__ up_weights,
    int up_row_block,
    int up_col_block,
    int up_num_col_blocks,
    const float* __restrict__ up_scales,
    const TAct* __restrict__ x,
    const float* __restrict__ lut_up,
    UpDebugItemCudaV2* __restrict__ out,
    int k_limit) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= k_limit || k >= cols) return;

    const int row = 0;
    const int rb_up = row / up_row_block;
    const int cb_up = k / up_col_block;

    const std::size_t up_w_idx =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) +
        static_cast<std::size_t>(k);
    const std::size_t up_s_idx =
        static_cast<std::size_t>(rb_up) *
            static_cast<std::size_t>(up_num_col_blocks) +
        static_cast<std::size_t>(cb_up);

    const std::uint8_t w_byte = up_weights[up_w_idx];
    const float scale = up_scales[up_s_idx];
    const float decoded = lut_up[w_byte];
    const float x_val = debug_act_to_float<TAct>(x[k]);
    const float contrib = decoded * scale * x_val;

    out[k] = UpDebugItemCudaV2{
        .k = k,
        .cb = cb_up,
        .w_idx = up_w_idx,
        .s_idx = up_s_idx,
        .w_byte = w_byte,
        .scale = scale,
        .decoded = decoded,
        .x_val = x_val,
        .contrib = contrib,
    };
}

}  // namespace

template <class TAct>
bool LaunchDebugUpRow0CudaV2(
    const MatrixBlockScaleViewV2& w_up,
    const TAct* d_x,
    UpDebugItemCudaV2* d_out,
    int k_limit,
    cudaStream_t stream) {
    if (d_x == nullptr || d_out == nullptr || k_limit <= 0) {
        return false;
    }

    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return false;
    }

    const float* lut_up =
        GetOrInitFp8DecodeLutCudaV2(device, w_up.matrix.fp8_format, stream);
    if (lut_up == nullptr) {
        return false;
    }

    const int threads = 64;
    const int blocks = (k_limit + threads - 1) / threads;

    debug_up_row0_kernel<TAct><<<blocks, threads, 0, stream>>>(
        w_up.matrix.cols,
        w_up.weight.data.data(),
        w_up.scale_meta.row_block,
        w_up.scale_meta.col_block,
        w_up.scale_meta.num_col_blocks,
        w_up.scale.data.data(),
        d_x,
        lut_up,
        d_out,
        k_limit);

    return cudaGetLastError() == cudaSuccess;
}

template bool LaunchDebugUpRow0CudaV2<__half>(
    const MatrixBlockScaleViewV2&,
    const __half*,
    UpDebugItemCudaV2*,
    int,
    cudaStream_t);

#if EXPERT_NODE_V2_ENABLE_CUDA_BF16
template bool LaunchDebugUpRow0CudaV2<__nv_bfloat16>(
    const MatrixBlockScaleViewV2&,
    const __nv_bfloat16*,
    UpDebugItemCudaV2*,
    int,
    cudaStream_t);
#endif
