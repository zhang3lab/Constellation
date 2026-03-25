#include "expert_node_v2/cuda/mlp_blockscale_cuda_v2.h"

#include <cuda_fp16.h>
#if EXPERT_NODE_V2_HAS_CUDA_BF16
#include <cuda_bf16.h>
#endif
#include <cuda_runtime.h>

#include "expert_node_v2/cuda/cuda_common_v2.cuh"
#include "expert_node_v2/cuda/fp8_decode_lut_v2.h"

namespace {

template <class TAct>
__global__ void zero_float_kernel(float* x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = 0.0f;
}

template <class TAct>
bool launch_zero_float(float* x, int n, cudaStream_t stream) {
    if (x == nullptr || n <= 0) return false;
    constexpr int THREADS = 256;
    const int blocks = (n + THREADS - 1) / THREADS;
    zero_float_kernel<TAct><<<blocks, THREADS, 0, stream>>>(x, n);
    return cudaGetLastError() == cudaSuccess;
}

template <class TAct, int WARPS_PER_BLOCK>
__global__ void fused_up_gate_kernel(
    int rows,
    int cols,

    const std::uint8_t* __restrict__ up_weights,
    int up_row_block,
    int up_col_block,
    int up_num_col_blocks,
    const float* __restrict__ up_scales,

    const std::uint8_t* __restrict__ gate_weights,
    int gate_row_block,
    int gate_col_block,
    int gate_num_col_blocks,
    const float* __restrict__ gate_scales,

    const TAct* __restrict__ x,
    float* __restrict__ h,

    const float* __restrict__ lut_up,
    const float* __restrict__ lut_gate) {
    constexpr int WARP_SIZE = 32;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    if (row >= rows) return;

    const int rb_up = row / up_row_block;
    const int rb_gate = row / gate_row_block;

    float up_sum = 0.0f;
    float gate_sum = 0.0f;

    for (int k = lane; k < cols; k += WARP_SIZE) {
        const int cb_up = k / up_col_block;
        const int cb_gate = k / gate_col_block;

        const std::size_t up_w_idx =
            static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) +
            static_cast<std::size_t>(k);
        const std::size_t up_s_idx =
            static_cast<std::size_t>(rb_up) *
                static_cast<std::size_t>(up_num_col_blocks) +
            static_cast<std::size_t>(cb_up);

        const std::size_t gate_w_idx =
            static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) +
            static_cast<std::size_t>(k);
        const std::size_t gate_s_idx =
            static_cast<std::size_t>(rb_gate) *
                static_cast<std::size_t>(gate_num_col_blocks) +
            static_cast<std::size_t>(cb_gate);

        const float x_val = act_to_float<TAct>(x[k]);

        const float up_w = lut_up[up_weights[up_w_idx]] * up_scales[up_s_idx];
        const float gate_w = lut_gate[gate_weights[gate_w_idx]] * gate_scales[gate_s_idx];

        up_sum += up_w * x_val;
        gate_sum += gate_w * x_val;
    }

    up_sum = warp_sum(up_sum);
    gate_sum = warp_sum(gate_sum);

    if (lane == 0) {
        const float silu_gate = gate_sum / (1.0f + expf(-gate_sum));
        h[row] = silu_gate * up_sum;
    }
}

}  // namespace

template <class TAct>
bool LaunchFusedUpGateCudaV2Impl(
    const MatrixBlockScaleViewV2& w_up_device_view,
    const MatrixBlockScaleViewV2& w_gate_device_view,
    const TAct* d_x,
    float* d_h,
    cudaStream_t stream) {
    if (d_x == nullptr || d_h == nullptr) return false;

    const int inter_dim = w_up_device_view.matrix.rows;
    const int hidden_dim = w_up_device_view.matrix.cols;
    if (inter_dim <= 0 || hidden_dim <= 0) return false;

    if (w_gate_device_view.matrix.rows != inter_dim ||
        w_gate_device_view.matrix.cols != hidden_dim) {
        return false;
    }

    int device_id = -1;
    cudaError_t err = cudaGetDevice(&device_id);
    if (err != cudaSuccess) {
        return false;
    }

    const float* lut_up =
        GetOrInitFp8DecodeLutCudaV2(device_id, w_up_device_view.matrix.fp8_format, stream);
    if (lut_up == nullptr) {
        return false;
    }

    const float* lut_gate =
        GetOrInitFp8DecodeLutCudaV2(device_id, w_gate_device_view.matrix.fp8_format, stream);
    if (lut_gate == nullptr) {
        return false;
    }

    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;
    const int blocks = (inter_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    fused_up_gate_kernel<TAct, WARPS_PER_BLOCK>
        <<<blocks, THREADS, 0, stream>>>(
            inter_dim,
            hidden_dim,

            w_up_device_view.weight.data.data(),
            w_up_device_view.scale_meta.row_block,
            w_up_device_view.scale_meta.col_block,
            w_up_device_view.scale_meta.num_col_blocks,
            w_up_device_view.scale.data.data(),

            w_gate_device_view.weight.data.data(),
            w_gate_device_view.scale_meta.row_block,
            w_gate_device_view.scale_meta.col_block,
            w_gate_device_view.scale_meta.num_col_blocks,
            w_gate_device_view.scale.data.data(),

            d_x,
            d_h,
            lut_up,
            lut_gate);

    err = cudaGetLastError();
    return err == cudaSuccess;
}

template bool LaunchFusedUpGateCudaV2Impl<__half>(
    const MatrixBlockScaleViewV2&,
    const MatrixBlockScaleViewV2&,
    const __half*,
    float*,
    cudaStream_t);

#if EXPERT_NODE_V2_HAS_CUDA_BF16
template bool LaunchFusedUpGateCudaV2Impl<__nv_bfloat16>(
    const MatrixBlockScaleViewV2&,
    const MatrixBlockScaleViewV2&,
    const __nv_bfloat16*,
    float*,
    cudaStream_t);
#endif
