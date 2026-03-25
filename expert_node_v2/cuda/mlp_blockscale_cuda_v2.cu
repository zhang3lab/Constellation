#include "expert_node_v2/cuda/mlp_blockscale_cuda_v2.h"

#include <cuda_fp16.h>
#if EXPERT_NODE_V2_HAS_CUDA_BF16
#include <cuda_bf16.h>
#endif
#include <cuda_runtime.h>

#include "expert_node_v2/cuda/matvec_blockscale_cuda_v2.h"

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

    // Temporary stub so down path can be tested first.
    return launch_zero_float<TAct>(d_h, inter_dim, stream);
}

template <class TAct>
bool LaunchDownCudaV2Impl(
    const MatrixBlockScaleViewV2& w_down_device_view,
    const float* d_h,
    TAct* d_y,
    cudaStream_t stream) {
    return LaunchMatvecBlockScaleCudaV2<float, TAct>(
        w_down_device_view,
        d_h,
        d_y,
        stream);
}

template bool LaunchFusedUpGateCudaV2Impl<__half>(
    const MatrixBlockScaleViewV2&,
    const MatrixBlockScaleViewV2&,
    const __half*,
    float*,
    cudaStream_t);

template bool LaunchDownCudaV2Impl<__half>(
    const MatrixBlockScaleViewV2&,
    const float*,
    __half*,
    cudaStream_t);

#if EXPERT_NODE_V2_HAS_CUDA_BF16
template bool LaunchFusedUpGateCudaV2Impl<__nv_bfloat16>(
    const MatrixBlockScaleViewV2&,
    const MatrixBlockScaleViewV2&,
    const __nv_bfloat16*,
    float*,
    cudaStream_t);

template bool LaunchDownCudaV2Impl<__nv_bfloat16>(
    const MatrixBlockScaleViewV2&,
    const float*,
    __nv_bfloat16*,
    cudaStream_t);
#endif
