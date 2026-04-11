#include "expert_node_v2/backend/cpu/backend_cpu_v2.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <span>

#include "expert_node_v2/backend/cpu/down_cpu_v2.h"
#include "expert_node_v2/backend/cpu/fused_up_gate_cpu_v2.h"
#include "expert_node_v2/expert_format_v2.h"

namespace {

template <class T>
bool CopySpanToHost(
    std::span<const T> src,
    DeviceBufferV2<T>* out) {
    if (out == nullptr) return false;

    if (out->data != nullptr || out->size != 0) {
        std::fprintf(stderr,
                     "[CopySpanToHost] output buffer must be empty before copy: "
                     "data=%p size=%zu\n",
                     static_cast<void*>(out->data),
                     out->size);
        std::abort();
    }

    if (src.empty()) {
        return false;
    }

    T* ptr = new (std::nothrow) T[src.size()];
    if (ptr == nullptr) {
        out->data = nullptr;
        out->size = 0;
        return false;
    }

    std::memcpy(ptr, src.data(), src.size_bytes());

    out->data = ptr;
    out->size = src.size();
    return true;
}

template <class T>
void FreeHostBuffer(DeviceBufferV2<T>* buf) {
    if (buf == nullptr) return;

    if (buf->data != nullptr) {
        delete[] buf->data;
        buf->data = nullptr;
    }

    buf->size = 0;
}

bool UploadOneMatrixCpu(
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

    if (!CopySpanToHost(host_view.weight.data, out_weight)) {
        return false;
    }

    if (!CopySpanToHost(host_view.scale.data, out_scale)) {
        FreeHostBuffer(out_weight);
        return false;
    }

    return true;
}

}  // namespace

bool UploadExpertCpuV2(
    int local_gpu_id,
    const ExpertTensorBundleV2& host_bundle,
    ExpertDeviceStorageV2* out_storage) {
    if (local_gpu_id < 0 || out_storage == nullptr) return false;

    *out_storage = ExpertDeviceStorageV2{};

    ExpertWeightsViewV2 host_view;
    if (!BuildExpertWeightsViewV2(host_bundle, &host_view)) {
        std::fprintf(stderr,
                     "[UploadExpertCpuV2] BuildExpertWeightsViewV2 failed local_gpu_id=%d\n",
                     local_gpu_id);
        host_bundle.debug_print("  ");
        return false;
    }

    if (!UploadOneMatrixCpu(
            host_view.w_up,
            &out_storage->w_up_meta,
            &out_storage->w_up_scale_meta,
            &out_storage->w_up_weight,
            &out_storage->w_up_scale)) {
        FreeExpertWeightsCpuV2(out_storage);
        return false;
    }

    if (!UploadOneMatrixCpu(
            host_view.w_gate,
            &out_storage->w_gate_meta,
            &out_storage->w_gate_scale_meta,
            &out_storage->w_gate_weight,
            &out_storage->w_gate_scale)) {
        FreeExpertWeightsCpuV2(out_storage);
        return false;
    }

    if (!UploadOneMatrixCpu(
            host_view.w_down,
            &out_storage->w_down_meta,
            &out_storage->w_down_scale_meta,
            &out_storage->w_down_weight,
            &out_storage->w_down_scale)) {
        FreeExpertWeightsCpuV2(out_storage);
        return false;
    }

    return true;
}

void FreeExpertWeightsCpuV2(ExpertDeviceStorageV2* storage) {
    if (storage == nullptr) return;

    FreeHostBuffer(&storage->w_up_weight);
    FreeHostBuffer(&storage->w_up_scale);

    FreeHostBuffer(&storage->w_gate_weight);
    FreeHostBuffer(&storage->w_gate_scale);

    FreeHostBuffer(&storage->w_down_weight);
    FreeHostBuffer(&storage->w_down_scale);

    storage->w_up_meta = MatrixMetaV2{};
    storage->w_gate_meta = MatrixMetaV2{};
    storage->w_down_meta = MatrixMetaV2{};

    storage->w_up_scale_meta = BlockScaleMetaV2{};
    storage->w_gate_scale_meta = BlockScaleMetaV2{};
    storage->w_down_scale_meta = BlockScaleMetaV2{};
}

bool InitExpertWorkspaceCpuV2(
    const ExpertWorkspaceConfigV2& config,
    ExpertWorkspaceCpuV2* out_ws) {
    if (out_ws == nullptr) return false;

    if (out_ws->tmp.data != nullptr || out_ws->tmp.size != 0) {
        std::fprintf(stderr,
                     "[InitExpertWorkspaceCpuV2] workspace must be empty before init: "
                     "tmp.data=%p tmp.size=%zu\n",
                     static_cast<void*>(out_ws->tmp.data),
                     out_ws->tmp.size);
        std::abort();
    }

    if (config.hidden_dim <= 0 || config.inter_dim <= 0) {
        return false;
    }

    float* ptr = new (std::nothrow) float[static_cast<std::size_t>(config.inter_dim)];
    if (ptr == nullptr) {
        out_ws->tmp.data = nullptr;
        out_ws->tmp.size = 0;
        return false;
    }

    out_ws->tmp.data = ptr;
    out_ws->tmp.size = static_cast<std::size_t>(config.inter_dim);
    return true;
}

void FreeExpertWorkspaceCpuV2(ExpertWorkspaceCpuV2* ws) {
    if (ws == nullptr) return;
    FreeHostBuffer(&ws->tmp);
}

bool RunExpertCpuV2(
    const ExpertWeightsViewV2& expert_device_view,
    ExpertWorkspaceCpuV2* ws,
    const void* x,
    common::ActivationDType input_dtype,
    void* y,
    common::ActivationDType output_dtype) {
    if (ws == nullptr || x == nullptr || y == nullptr) {
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

    if (ws->tmp.data == nullptr ||
        ws->tmp.size < static_cast<std::size_t>(inter_dim)) {
        return false;
    }

    switch (input_dtype) {
        case common::ActivationDType::FP16:
        case common::ActivationDType::BF16:
            break;
        default:
            return false;
    }

    switch (output_dtype) {
        case common::ActivationDType::FP16:
        case common::ActivationDType::BF16:
            break;
        default:
            return false;
    }

    constexpr int kCpuUpGateOmpThreads = 16;
    constexpr int kCpuDownOmpThreads = 16;
    if (!RunFusedUpGateCpuV2(
            expert_device_view.w_up,
            expert_device_view.w_gate,
            x,
            input_dtype,
            ws->tmp.data,
            kCpuUpGateOmpThreads)) {
        return false;
    }

    if (!RunDownCpuV2(
            expert_device_view.w_down,
            ws->tmp.data,
            y,
            output_dtype,
            kCpuDownOmpThreads)) {
        return false;
    }

    return true;
}
