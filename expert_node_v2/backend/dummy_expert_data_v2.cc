#include "expert_node_v2/backend/dummy_expert_data_v2.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "expert_node_v2/backend/activation_codec_v2.h"

void FillDummyExpertBundleV2(
    ExpertTensorBundleV2* bundle,
    const DummyExpertShapeV2& shape) {
    if (bundle == nullptr) return;
    bundle->clear();

    const int hidden_dim = shape.hidden_dim;
    const int inter_dim = shape.inter_dim;
    const int row_block = shape.row_block;
    const int col_block = shape.col_block;

    const int up_num_row_blocks = (inter_dim + row_block - 1) / row_block;
    const int up_num_col_blocks = (hidden_dim + col_block - 1) / col_block;

    bundle->w_up.meta.shape = {
        static_cast<std::uint64_t>(inter_dim),
        static_cast<std::uint64_t>(hidden_dim),
    };
    bundle->w_up.meta.dtype = "torch.float8_e4m3fn";
    bundle->w_up.meta.row_block = static_cast<std::uint32_t>(row_block);
    bundle->w_up.meta.col_block = static_cast<std::uint32_t>(col_block);
    bundle->w_up.bytes.resize(static_cast<std::size_t>(inter_dim) * hidden_dim);
    bundle->w_up.ready = true;
    for (std::size_t i = 0; i < bundle->w_up.bytes.size(); ++i) {
        bundle->w_up.bytes[i] = static_cast<std::uint8_t>((i * 7 + 3) & 0xff);
    }

    bundle->w_up_scale.meta.shape = {
        static_cast<std::uint64_t>(up_num_row_blocks),
        static_cast<std::uint64_t>(up_num_col_blocks),
    };
    bundle->w_up_scale.meta.dtype = "torch.float32";
    bundle->w_up_scale.bytes.resize(
        static_cast<std::size_t>(up_num_row_blocks) *
        static_cast<std::size_t>(up_num_col_blocks) * sizeof(float));
    bundle->w_up_scale.ready = true;
    {
        float* scales = reinterpret_cast<float*>(bundle->w_up_scale.bytes.data());
        for (int rb = 0; rb < up_num_row_blocks; ++rb) {
            for (int cb = 0; cb < up_num_col_blocks; ++cb) {
                scales[rb * up_num_col_blocks + cb] =
                    0.010f + 0.0005f * static_cast<float>(rb) +
                    0.00005f * static_cast<float>(cb);
            }
        }
    }

    bundle->w_gate.meta.shape = {
        static_cast<std::uint64_t>(inter_dim),
        static_cast<std::uint64_t>(hidden_dim),
    };
    bundle->w_gate.meta.dtype = "torch.float8_e4m3fn";
    bundle->w_gate.meta.row_block = static_cast<std::uint32_t>(row_block);
    bundle->w_gate.meta.col_block = static_cast<std::uint32_t>(col_block);
    bundle->w_gate.bytes.resize(static_cast<std::size_t>(inter_dim) * hidden_dim);
    bundle->w_gate.ready = true;
    for (std::size_t i = 0; i < bundle->w_gate.bytes.size(); ++i) {
        bundle->w_gate.bytes[i] = static_cast<std::uint8_t>((i * 11 + 5) & 0xff);
    }

    bundle->w_gate_scale.meta.shape = {
        static_cast<std::uint64_t>(up_num_row_blocks),
        static_cast<std::uint64_t>(up_num_col_blocks),
    };
    bundle->w_gate_scale.meta.dtype = "torch.float32";
    bundle->w_gate_scale.bytes.resize(
        static_cast<std::size_t>(up_num_row_blocks) *
        static_cast<std::size_t>(up_num_col_blocks) * sizeof(float));
    bundle->w_gate_scale.ready = true;
    {
        float* scales = reinterpret_cast<float*>(bundle->w_gate_scale.bytes.data());
        for (int rb = 0; rb < up_num_row_blocks; ++rb) {
            for (int cb = 0; cb < up_num_col_blocks; ++cb) {
                scales[rb * up_num_col_blocks + cb] =
                    0.020f + 0.0003f * static_cast<float>(rb) +
                    0.00007f * static_cast<float>(cb);
            }
        }
    }

    const int down_rows = hidden_dim;
    const int down_cols = inter_dim;
    const int down_num_row_blocks = (down_rows + row_block - 1) / row_block;
    const int down_num_col_blocks = (down_cols + col_block - 1) / col_block;

    bundle->w_down.meta.shape = {
        static_cast<std::uint64_t>(down_rows),
        static_cast<std::uint64_t>(down_cols),
    };
    bundle->w_down.meta.dtype = "torch.float8_e4m3fn";
    bundle->w_down.meta.row_block = static_cast<std::uint32_t>(row_block);
    bundle->w_down.meta.col_block = static_cast<std::uint32_t>(col_block);
    bundle->w_down.bytes.resize(static_cast<std::size_t>(down_rows) * down_cols);
    bundle->w_down.ready = true;
    for (std::size_t i = 0; i < bundle->w_down.bytes.size(); ++i) {
        bundle->w_down.bytes[i] = static_cast<std::uint8_t>((i * 13 + 17) & 0xff);
    }

    bundle->w_down_scale.meta.shape = {
        static_cast<std::uint64_t>(down_num_row_blocks),
        static_cast<std::uint64_t>(down_num_col_blocks),
    };
    bundle->w_down_scale.meta.dtype = "torch.float32";
    bundle->w_down_scale.bytes.resize(
        static_cast<std::size_t>(down_num_row_blocks) *
        static_cast<std::size_t>(down_num_col_blocks) * sizeof(float));
    bundle->w_down_scale.ready = true;
    {
        float* scales = reinterpret_cast<float*>(bundle->w_down_scale.bytes.data());
        for (int rb = 0; rb < down_num_row_blocks; ++rb) {
            for (int cb = 0; cb < down_num_col_blocks; ++cb) {
                scales[rb * down_num_col_blocks + cb] =
                    0.005f + 0.001f * static_cast<float>(rb) +
                    0.0001f * static_cast<float>(cb);
            }
        }
    }
}

void FillDummyInputActivationV2(
    int hidden_dim,
    common::ActivationDType dtype,
    std::vector<float>* out_float,
    std::vector<std::uint16_t>* out_encoded) {
    if (out_float == nullptr || out_encoded == nullptr) return;

    out_float->resize(static_cast<std::size_t>(hidden_dim));
    out_encoded->resize(static_cast<std::size_t>(hidden_dim));

    for (int i = 0; i < hidden_dim; ++i) {
        const float x = std::sin(0.0005f * static_cast<float>(i));
        (*out_float)[static_cast<std::size_t>(i)] = x;
        (*out_encoded)[static_cast<std::size_t>(i)] =
            EncodeActivationFromFloatV2(dtype, x);
    }
}
