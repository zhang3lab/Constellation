#include "expert_node_v2/backend/cpu_fp16_resident/backend_cpu_fp16_resident_v2.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <span>

#include "expert_node_v2/backend/activation_codec_v2.h"
#include "expert_node_v2/backend/cpu_fp16_resident/down_cpu_fp16_resident_v2.h"
#include "expert_node_v2/backend/cpu_fp16_resident/fused_up_gate_cpu_fp16_resident_v2.h"
#include "expert_node_v2/backend/fp8_lut_v2.h"
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

bool ExpandMatrixToFp16ResidentBytesV2(
    const MatrixBlockScaleViewV2& host_view,
    DeviceBufferV2<std::uint8_t>* out_weight) {
    if (out_weight == nullptr) return false;

    if (out_weight->data != nullptr || out_weight->size != 0) {
        std::fprintf(stderr,
                     "[ExpandMatrixToFp16ResidentBytesV2] output weight buffer must be empty before expand: "
                     "data=%p size=%zu\n",
                     static_cast<void*>(out_weight->data),
                     out_weight->size);
        std::abort();
    }

    const int rows = host_view.matrix.rows;
    const int cols = host_view.matrix.cols;
    if (rows <= 0 || cols <= 0) return false;

    const float* lut = GetHostFp8LutV2(host_view.matrix.fp8_format);
    if (lut == nullptr) return false;

    const auto weights = host_view.weight.data;
    const auto scales = host_view.scale.data;

    const int row_block = host_view.scale_meta.row_block;
    const int col_block = host_view.scale_meta.col_block;
    const int num_col_blocks = host_view.scale_meta.num_col_blocks;

    if (row_block <= 0 || col_block <= 0 || num_col_blocks <= 0) {
        return false;
    }

    const std::size_t elems =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    const std::size_t u16_count = elems;
    const std::size_t bytes = u16_count * sizeof(std::uint16_t);

    auto* ptr = new (std::nothrow) std::uint8_t[bytes];
    if (ptr == nullptr) {
        out_weight->data = nullptr;
        out_weight->size = 0;
        return false;
    }

    auto* dst_u16 = reinterpret_cast<std::uint16_t*>(ptr);

    for (int row = 0; row < rows; ++row) {
        const int rb = row / row_block;
        const std::size_t row_base =
            static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);

        for (int k0 = 0; k0 < cols; k0 += col_block) {
            const int k1 = std::min(k0 + col_block, cols);
            const int cb = k0 / col_block;

            const std::size_t s_idx =
                static_cast<std::size_t>(rb) *
                    static_cast<std::size_t>(num_col_blocks) +
                static_cast<std::size_t>(cb);

            const float scale = scales[s_idx];

            for (int k = k0; k < k1; ++k) {
                const std::size_t w_idx =
                    row_base + static_cast<std::size_t>(k);
                const float w = lut[weights[w_idx]] * scale;
                dst_u16[w_idx] =
                    EncodeActivationFromFloatV2(
                        common::ActivationDType::FP16, w);
            }
        }
    }

    out_weight->data = ptr;
    out_weight->size = bytes;
    return true;
}

bool UploadOneMatrixCpuFp16Resident(
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

    if (!ExpandMatrixToFp16ResidentBytesV2(host_view, out_weight)) {
        return false;
    }

    // cpu_fp16_resident contract:
    // weight buffer stores row-major fp16 resident payload directly.
    // scale buffer is unused.
    out_scale->data = nullptr;
    out_scale->size = 0;
    return true;
}

}  // namespace

bool UploadExpertCpuFp16ResidentV2(
    int local_gpu_id,
    const ExpertTensorBundleV2& host_bundle,
    ExpertDeviceStorageV2* out_storage) {
    if (local_gpu_id < 0 || out_storage == nullptr) return false;

    *out_storage = ExpertDeviceStorageV2{};

    ExpertWeightsViewV2 host_view;
    if (!BuildExpertWeightsViewV2(host_bundle, &host_view)) {
        std::fprintf(stderr,
                     "[UploadExpertCpuFp16ResidentV2] BuildExpertWeightsViewV2 failed local_gpu_id=%d\n",
                     local_gpu_id);
        host_bundle.debug_print("  ");
        return false;
    }

    if (!UploadOneMatrixCpuFp16Resident(
            host_view.w_up,
            &out_storage->w_up_meta,
            &out_storage->w_up_scale_meta,
            &out_storage->w_up_weight,
            &out_storage->w_up_scale)) {
        FreeExpertWeightsCpuFp16ResidentV2(out_storage);
        return false;
    }

    if (!UploadOneMatrixCpuFp16Resident(
            host_view.w_gate,
            &out_storage->w_gate_meta,
            &out_storage->w_gate_scale_meta,
            &out_storage->w_gate_weight,
            &out_storage->w_gate_scale)) {
        FreeExpertWeightsCpuFp16ResidentV2(out_storage);
        return false;
    }

    if (!UploadOneMatrixCpuFp16Resident(
            host_view.w_down,
            &out_storage->w_down_meta,
            &out_storage->w_down_scale_meta,
            &out_storage->w_down_weight,
            &out_storage->w_down_scale)) {
        FreeExpertWeightsCpuFp16ResidentV2(out_storage);
        return false;
    }

    return true;
}

void FreeExpertWeightsCpuFp16ResidentV2(ExpertDeviceStorageV2* storage) {
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

bool InitExpertWorkspaceCpuFp16ResidentV2(
    const ExpertWorkspaceConfigV2& config,
    ExpertWorkspaceCpuV2* out_ws) {
    return InitExpertWorkspaceCpuV2(config, out_ws);
}

void FreeExpertWorkspaceCpuFp16ResidentV2(ExpertWorkspaceCpuV2* ws) {
    FreeExpertWorkspaceCpuV2(ws);
}

bool RunExpertCpuFp16ResidentV2(
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

    constexpr int kCpuFp16ResidentUpGateOmpThreads = 4;
    constexpr int kCpuFp16ResidentDownOmpThreads = 4;

    if (!RunFusedUpGateCpuFp16ResidentV2(
            expert_device_view.w_up,
            expert_device_view.w_gate,
            x,
            input_dtype,
            ws->tmp.data,
            kCpuFp16ResidentUpGateOmpThreads)) {
        return false;
    }

    if (!RunDownCpuFp16ResidentV2(
            expert_device_view.w_down,
            ws->tmp.data,
            y,
            output_dtype,
            kCpuFp16ResidentDownOmpThreads)) {
        return false;
    }

    return true;
}
