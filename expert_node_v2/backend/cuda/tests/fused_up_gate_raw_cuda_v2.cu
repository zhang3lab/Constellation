#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if EXPERT_NODE_V2_ENABLE_CUDA_BF16
#include <cuda_bf16.h>
#endif

#include <cstdint>

#include "expert_node_v2/backend/cuda/fp8_decode_lut_v2.h"
#include "expert_node_v2/expert_format_v2.h"

namespace {

template <class TAct>
__device__ inline float act_to_float_raw(TAct x);

template <>
__device__ inline float act_to_float_raw<__half>(__half x) {
    return __half2float(x);
}

#if EXPERT_NODE_V2_ENABLE_CUDA_BF16
template <>
__device__ inline float act_to_float_raw<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}
#endif

__device__ inline float warp_sum_raw_float(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ inline double warp_sum_raw_double(double v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

template <class TAct, int WARPS_PER_BLOCK>
__global__ void fused_up_gate_raw_kernel_floatacc(
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
    float* __restrict__ up_out,
    float* __restrict__ gate_out,

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

        const float x_val = act_to_float_raw<TAct>(x[k]);
        const float up_w = lut_up[up_weights[up_w_idx]] * up_scales[up_s_idx];
        const float gate_w =
            lut_gate[gate_weights[gate_w_idx]] * gate_scales[gate_s_idx];

        up_sum += up_w * x_val;
        gate_sum += gate_w * x_val;
    }

    up_sum = warp_sum_raw_float(up_sum);
    gate_sum = warp_sum_raw_float(gate_sum);

    if (lane == 0) {
        up_out[row] = up_sum;
        gate_out[row] = gate_sum;
    }
}

template <class TAct, int WARPS_PER_BLOCK>
__global__ void fused_up_gate_raw_kernel_doubleacc(
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
    float* __restrict__ up_out,
    float* __restrict__ gate_out,

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

    double up_sum = 0.0;
    double gate_sum = 0.0;

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

        const double x_val = static_cast<double>(act_to_float_raw<TAct>(x[k]));
        const double up_w =
            static_cast<double>(lut_up[up_weights[up_w_idx]]) *
            static_cast<double>(up_scales[up_s_idx]);
        const double gate_w =
            static_cast<double>(lut_gate[gate_weights[gate_w_idx]]) *
            static_cast<double>(gate_scales[gate_s_idx]);

        up_sum += up_w * x_val;
        gate_sum += gate_w * x_val;
    }

    up_sum = warp_sum_raw_double(up_sum);
    gate_sum = warp_sum_raw_double(gate_sum);

    if (lane == 0) {
        up_out[row] = static_cast<float>(up_sum);
        gate_out[row] = static_cast<float>(gate_sum);
    }
}

}  // namespace

template <class TAct>
bool LaunchFusedUpGateRawCudaV2(
    const MatrixBlockScaleViewV2& w_up,
    const MatrixBlockScaleViewV2& w_gate,
    const TAct* d_x,
    float* d_up_out,
    float* d_gate_out,
    cudaStream_t stream) {
    if (d_x == nullptr || d_up_out == nullptr || d_gate_out == nullptr) {
        return false;
    }

    const int rows = w_up.matrix.rows;
    const int cols = w_up.matrix.cols;
    if (rows <= 0 || cols <= 0) return false;

    if (w_gate.matrix.rows != rows || w_gate.matrix.cols != cols) {
        return false;
    }

    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return false;
    }

    const float* lut_up =
        GetOrInitFp8DecodeLutCudaV2(device, w_up.matrix.fp8_format, stream);
    const float* lut_gate =
        GetOrInitFp8DecodeLutCudaV2(device, w_gate.matrix.fp8_format, stream);
    if (lut_up == nullptr || lut_gate == nullptr) {
        return false;
    }

    constexpr int kWarpsPerBlock = 4;
    constexpr int kThreads = 32 * kWarpsPerBlock;
    const dim3 block(kThreads);
    const dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);

    fused_up_gate_raw_kernel_floatacc<TAct, kWarpsPerBlock><<<grid, block, 0, stream>>>(
        rows,
        cols,
        w_up.weight.data.data(),
        w_up.scale_meta.row_block,
        w_up.scale_meta.col_block,
        w_up.scale_meta.num_col_blocks,
        w_up.scale.data.data(),
        w_gate.weight.data.data(),
        w_gate.scale_meta.row_block,
        w_gate.scale_meta.col_block,
        w_gate.scale_meta.num_col_blocks,
        w_gate.scale.data.data(),
        d_x,
        d_up_out,
        d_gate_out,
        lut_up,
        lut_gate);

    return cudaGetLastError() == cudaSuccess;
}

template <class TAct>
bool LaunchFusedUpGateRawDoubleAccCudaV2(
    const MatrixBlockScaleViewV2& w_up,
    const MatrixBlockScaleViewV2& w_gate,
    const TAct* d_x,
    float* d_up_out,
    float* d_gate_out,
    cudaStream_t stream) {
    if (d_x == nullptr || d_up_out == nullptr || d_gate_out == nullptr) {
        return false;
    }

    const int rows = w_up.matrix.rows;
    const int cols = w_up.matrix.cols;
    if (rows <= 0 || cols <= 0) return false;

    if (w_gate.matrix.rows != rows || w_gate.matrix.cols != cols) {
        return false;
    }

    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return false;
    }

    const float* lut_up =
        GetOrInitFp8DecodeLutCudaV2(device, w_up.matrix.fp8_format, stream);
    const float* lut_gate =
        GetOrInitFp8DecodeLutCudaV2(device, w_gate.matrix.fp8_format, stream);
    if (lut_up == nullptr || lut_gate == nullptr) {
        return false;
    }

    constexpr int kWarpsPerBlock = 4;
    constexpr int kThreads = 32 * kWarpsPerBlock;
    const dim3 block(kThreads);
    const dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);

    fused_up_gate_raw_kernel_doubleacc<TAct, kWarpsPerBlock><<<grid, block, 0, stream>>>(
        rows,
        cols,
        w_up.weight.data.data(),
        w_up.scale_meta.row_block,
        w_up.scale_meta.col_block,
        w_up.scale_meta.num_col_blocks,
        w_up.scale.data.data(),
        w_gate.weight.data.data(),
        w_gate.scale_meta.row_block,
        w_gate.scale_meta.col_block,
        w_gate.scale_meta.num_col_blocks,
        w_gate.scale.data.data(),
        d_x,
        d_up_out,
        d_gate_out,
        lut_up,
        lut_gate);

    return cudaGetLastError() == cudaSuccess;
}

template bool LaunchFusedUpGateRawCudaV2<__half>(
    const MatrixBlockScaleViewV2&,
    const MatrixBlockScaleViewV2&,
    const __half*,
    float*,
    float*,
    cudaStream_t);

template bool LaunchFusedUpGateRawDoubleAccCudaV2<__half>(
    const MatrixBlockScaleViewV2&,
    const MatrixBlockScaleViewV2&,
    const __half*,
    float*,
    float*,
    cudaStream_t);

#if EXPERT_NODE_V2_ENABLE_CUDA_BF16
template bool LaunchFusedUpGateRawCudaV2<__nv_bfloat16>(
    const MatrixBlockScaleViewV2&,
    const MatrixBlockScaleViewV2&,
    const __nv_bfloat16*,
    float*,
    float*,
    cudaStream_t);

template bool LaunchFusedUpGateRawDoubleAccCudaV2<__nv_bfloat16>(
    const MatrixBlockScaleViewV2&,
    const MatrixBlockScaleViewV2&,
    const __nv_bfloat16*,
    float*,
    float*,
    cudaStream_t);
#endif
