#include "expert_node_v2/cuda/backend_cuda_v2.h"

#include <cstddef>
#include <cstdint>
#include <span>

#include "expert_node_v2/cuda/mlp_blockscale_cuda_v2.h"
#include "expert_node_v2/expert_loader_v2.h"

namespace {

template <class T>
bool UploadSpanToCuda(
    std::span<const T> src,
    DeviceBufferV2<T>* out) {
    if (out == nullptr) return false;

    out->clear();
    if (src.empty()) return false;

    T* ptr = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr), src.size_bytes());
    if (err != cudaSuccess) {
        return false;
    }

    err = cudaMemcpy(ptr, src.data(), src.size_bytes(), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(ptr);
        return false;
    }

    out->data = ptr;
    out->size = src.size();
    return true;
}

template <class T>
void FreeDeviceBuffer(DeviceBufferV2<T>* buf) {
    if (buf == nullptr) return;
    if (buf->data != nullptr) {
        cudaFree(buf->data);
    }
    buf->clear();
}

bool UploadOneMatrixCuda(
    const MatrixBlockScaleViewV2& host_view,
    MatrixMetaV2* out_meta,
    BlockScaleMetaV2* out_scale_meta,
    DeviceBufferV2<std::uint8_t>* out_weight,
    DeviceBufferV2<float>* out_scale) {
    if (out_meta == nullptr || out_scale_meta == nullptr ||
        out_weight == nullptr || out_scale == nullptr) {
        return false;
    }

    *out_meta = host_view.matrix;
    *out_scale_meta = host_view.scale_meta;

    if (!UploadSpanToCuda(host_view.weight.data, out_weight)) {
        return false;
    }
    if (!UploadSpanToCuda(host_view.scale.data, out_scale)) {
        FreeDeviceBuffer(out_weight);
        return false;
    }

    return true;
}

}  // namespace

bool UploadExpertCudaV2(
    const ExpertTensorBundleV2& host_bundle,
    ExpertDeviceStorageV2* out_storage) {
    if (out_storage == nullptr) return false;
    out_storage->clear();

    ExpertWeightsViewV2 host_view;
    if (!BuildExpertWeightsViewV2(host_bundle, &host_view)) {
        return false;
    }

    if (!UploadOneMatrixCuda(
            host_view.w_up,
            &out_storage->w_up_meta,
            &out_storage->w_up_scale_meta,
            &out_storage->w_up_weight,
            &out_storage->w_up_scale)) {
        FreeExpertWeightsCudaV2(out_storage);
        return false;
    }

    if (!UploadOneMatrixCuda(
            host_view.w_gate,
            &out_storage->w_gate_meta,
            &out_storage->w_gate_scale_meta,
            &out_storage->w_gate_weight,
            &out_storage->w_gate_scale)) {
        FreeExpertWeightsCudaV2(out_storage);
        return false;
    }

    if (!UploadOneMatrixCuda(
            host_view.w_down,
            &out_storage->w_down_meta,
            &out_storage->w_down_scale_meta,
            &out_storage->w_down_weight,
            &out_storage->w_down_scale)) {
        FreeExpertWeightsCudaV2(out_storage);
        return false;
    }

    return true;
}

void FreeExpertWeightsCudaV2(ExpertDeviceStorageV2* storage) {
    if (storage == nullptr) return;

    FreeDeviceBuffer(&storage->w_up_weight);
    FreeDeviceBuffer(&storage->w_up_scale);

    FreeDeviceBuffer(&storage->w_gate_weight);
    FreeDeviceBuffer(&storage->w_gate_scale);

    FreeDeviceBuffer(&storage->w_down_weight);
    FreeDeviceBuffer(&storage->w_down_scale);

    storage->clear();
}

bool InitExpertWorkspaceCudaV2(
    const ExpertWorkspaceConfigV2& config,
    ExpertWorkspaceCudaV2* out_ws) {
    if (out_ws == nullptr) return false;
    out_ws->clear();

    if (config.hidden_dim <= 0 || config.inter_dim <= 0) {
        return false;
    }

    float* h_ptr = nullptr;
    cudaError_t err = cudaMalloc(
        reinterpret_cast<void**>(&h_ptr),
        static_cast<std::size_t>(config.inter_dim) * sizeof(float));
    if (err != cudaSuccess) {
        return false;
    }

    out_ws->h_tmp.data = h_ptr;
    out_ws->h_tmp.size = static_cast<std::size_t>(config.inter_dim);
    return true;
}

void FreeExpertWorkspaceCudaV2(ExpertWorkspaceCudaV2* ws) {
    if (ws == nullptr) return;
    FreeDeviceBuffer(&ws->h_tmp);
    ws->clear();
}

template <class TAct>
bool RunExpertCudaV2(
    const ExpertWeightsViewV2& expert_device_view,
    ExpertWorkspaceCudaV2* ws,
    const TAct* d_x,
    TAct* d_y,
    cudaStream_t stream) {
    if (ws == nullptr || d_x == nullptr || d_y == nullptr) {
        return false;
    }

    const int hidden_dim = expert_device_view.w_up.matrix.cols;
    const int inter_dim = expert_device_view.w_up.matrix.rows;

    if (hidden_dim <= 0 || inter_dim <= 0) {
        return false;
    }

    if (expert_device_view.w_gate.matrix.cols != hidden_dim ||
        expert_device_view.w_gate.matrix.rows != inter_dim) {
        return false;
    }

    if (expert_device_view.w_down.matrix.rows != hidden_dim ||
        expert_device_view.w_down.matrix.cols != inter_dim) {
        return false;
    }

    if (ws->h_tmp.data == nullptr ||
        ws->h_tmp.size < static_cast<std::size_t>(inter_dim)) {
        return false;
    }

    if (!LaunchFusedUpGateCudaV2Impl(
            expert_device_view.w_up,
            expert_device_view.w_gate,
            d_x,
            ws->h_tmp.data,
            stream)) {
        return false;
    }

    if (!LaunchDownCudaV2Impl(
            expert_device_view.w_down,
            ws->h_tmp.data,
            d_y,
            stream)) {
        return false;
    }

    return true;
}

template bool RunExpertCudaV2<__half>(
    const ExpertWeightsViewV2&,
    ExpertWorkspaceCudaV2*,
    const __half*,
    __half*,
    cudaStream_t);

#if EXPERT_NODE_V2_HAS_CUDA_BF16
template bool RunExpertCudaV2<__nv_bfloat16>(
    const ExpertWeightsViewV2&,
    ExpertWorkspaceCudaV2*,
    const __nv_bfloat16*,
    __nv_bfloat16*,
    cudaStream_t);
#endif
