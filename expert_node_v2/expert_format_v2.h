#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <string>
#include <vector>

#include "common/protocol.h"

enum class Fp8Format : std::int32_t {
    IEEE_E4M3 = 0,
    IEEE_E5M2 = 1,
    TORCH_E4M3FN = 2,
};

inline const char* fp8_format_name(Fp8Format fmt) {
    switch (fmt) {
        case Fp8Format::IEEE_E4M3: return "IEEE_E4M3";
        case Fp8Format::IEEE_E5M2: return "IEEE_E5M2";
        case Fp8Format::TORCH_E4M3FN: return "TORCH_E4M3FN";
        default: return "UNKNOWN_FP8_FORMAT";
    }
}

inline int ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

struct ExpertWorkspaceConfigV2 {
    int hidden_dim = 0;
    int inter_dim = 0;
};

// -----------------------------------------------------------------------------
// Raw host tensor received from protocol.
// -----------------------------------------------------------------------------
struct HostTensorV2 {
    std::vector<std::uint8_t> bytes;
    common::TensorMeta meta;
    bool ready = false;

    void clear() {
        std::vector<std::uint8_t> empty;
        bytes.swap(empty);
        meta = common::TensorMeta{};
        ready = false;
    }

    std::size_t num_bytes() const {
        return bytes.size();
    }
};

// -----------------------------------------------------------------------------
// One expert receives six tensors from server.
// -----------------------------------------------------------------------------
struct ExpertTensorBundleV2 {
    HostTensorV2 w_up;
    HostTensorV2 w_up_scale;
    HostTensorV2 w_gate;
    HostTensorV2 w_gate_scale;
    HostTensorV2 w_down;
    HostTensorV2 w_down_scale;

    bool all_ready() const {
        return w_up.ready &&
               w_up_scale.ready &&
               w_gate.ready &&
               w_gate_scale.ready &&
               w_down.ready &&
               w_down_scale.ready;
    }

    void clear() {
        w_up.clear();
        w_up_scale.clear();
        w_gate.clear();
        w_gate_scale.clear();
        w_down.clear();
        w_down_scale.clear();
    }

    void debug_print(const char* prefix = "") const;
};

// -----------------------------------------------------------------------------
// Matrix metadata.
// -----------------------------------------------------------------------------
struct MatrixMetaV2 {
    int rows = 0;
    int cols = 0;
    Fp8Format fp8_format = Fp8Format::TORCH_E4M3FN;
};

// -----------------------------------------------------------------------------
// DeepSeek block-scale metadata.
// scale layout is row-major [num_row_blocks, num_col_blocks].
// -----------------------------------------------------------------------------
struct BlockScaleMetaV2 {
    int row_block = 128;
    int col_block = 128;
    int num_row_blocks = 0;
    int num_col_blocks = 0;
};

// -----------------------------------------------------------------------------
// Non-owning host/device-neutral views.
// -----------------------------------------------------------------------------
struct WeightBufferViewV2 {
    std::span<const std::uint8_t> data;
};

struct BlockScaleBufferViewV2 {
    std::span<const float> data;
};

struct MatrixBlockScaleViewV2 {
    MatrixMetaV2 matrix;
    WeightBufferViewV2 weight;
    BlockScaleMetaV2 scale_meta;
    BlockScaleBufferViewV2 scale;
};

struct ExpertWeightsViewV2 {
    MatrixBlockScaleViewV2 w_up;
    MatrixBlockScaleViewV2 w_gate;
    MatrixBlockScaleViewV2 w_down;
};

// -----------------------------------------------------------------------------
// Backend-managed device buffer handle.
// `data` points to backend-allocated device-resident memory and `size` counts
// T elements.
// This type does NOT own deallocation logic and must NOT provide a `clear()`
// helper that merely nulls the pointer.
// Allocation/copy/free are handled explicitly by backend-specific code.
// The destructor asserts that backend-specific free/reset has already happened.
// -----------------------------------------------------------------------------
template <class T>
struct DeviceBufferV2 {
    T* data = nullptr;
    std::size_t size = 0;  // number of T elements

    ~DeviceBufferV2() {
        if (data != nullptr || size != 0) {
            std::fprintf(
                stderr,
                "[DeviceBufferV2] leaked device buffer detected at destruction: "
                "data=%p size=%zu\n",
                static_cast<void*>(data),
                size);
            std::abort();
        }
    }
};

// -----------------------------------------------------------------------------
// Backend-managed device-resident storage for one expert.
// Holds FP8 weight bytes + float block scales for w_up / w_gate / w_down, and
// can materialize uniform matrix/expert views for runtime execution.
//
// This type is only a container of backend-managed device handles.
// It must NOT provide a `clear()` helper that merely resets fields without
// freeing backend resources.
// Callers must release storage through the corresponding backend-specific free
// function, and only then reset the struct state.
// -----------------------------------------------------------------------------
struct ExpertDeviceStorageV2 {
    MatrixMetaV2 w_up_meta;
    MatrixMetaV2 w_gate_meta;
    MatrixMetaV2 w_down_meta;

    BlockScaleMetaV2 w_up_scale_meta;
    BlockScaleMetaV2 w_gate_scale_meta;
    BlockScaleMetaV2 w_down_scale_meta;

    DeviceBufferV2<std::uint8_t> w_up_weight;
    DeviceBufferV2<float> w_up_scale;

    DeviceBufferV2<std::uint8_t> w_gate_weight;
    DeviceBufferV2<float> w_gate_scale;

    DeviceBufferV2<std::uint8_t> w_down_weight;
    DeviceBufferV2<float> w_down_scale;

    MatrixBlockScaleViewV2 w_up_view() const {
        return make_matrix_view(
            w_up_meta,
            w_up_weight,
            w_up_scale_meta,
            w_up_scale);
    }

    MatrixBlockScaleViewV2 w_gate_view() const {
        return make_matrix_view(
            w_gate_meta,
            w_gate_weight,
            w_gate_scale_meta,
            w_gate_scale);
    }

    MatrixBlockScaleViewV2 w_down_view() const {
        return make_matrix_view(
            w_down_meta,
            w_down_weight,
            w_down_scale_meta,
            w_down_scale);
    }

    ExpertWeightsViewV2 view() const {
        return ExpertWeightsViewV2{
            .w_up = w_up_view(),
            .w_gate = w_gate_view(),
            .w_down = w_down_view(),
        };
    }

private:
    static MatrixBlockScaleViewV2 make_matrix_view(
        const MatrixMetaV2& matrix,
        const DeviceBufferV2<std::uint8_t>& weight,
        const BlockScaleMetaV2& scale_meta,
        const DeviceBufferV2<float>& scale) {
        return MatrixBlockScaleViewV2{
            .matrix = matrix,
            .weight = WeightBufferViewV2{
                std::span<const std::uint8_t>(weight.data, weight.size),
            },
            .scale_meta = scale_meta,
            .scale = BlockScaleBufferViewV2{
                std::span<const float>(scale.data, scale.size),
            },
        };
    }
};

bool BuildMatrixBlockScaleViewV2(
    const HostTensorV2& weight_ht,
    const HostTensorV2& scale_ht,
    int rows,
    int cols,
    MatrixBlockScaleViewV2* out);

bool BuildExpertWeightsViewV2(
    const ExpertTensorBundleV2& bundle,
    ExpertWeightsViewV2* out);
