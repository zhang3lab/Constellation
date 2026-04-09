#include "expert_node_v2/backend/cpu/fused_up_gate_cpu_v2.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "expert_node_v2/backend/activation_codec_v2.h"
#include "expert_node_v2/backend/fp8_lut_v2.h"

bool RunFusedUpGateCpuV2(
    const MatrixBlockScaleViewV2& w_up,
    const MatrixBlockScaleViewV2& w_gate,
    const void* x,
    common::ActivationDType input_dtype,
    float* h) {
    if (x == nullptr || h == nullptr) return false;

    const int rows = w_up.matrix.rows;
    const int cols = w_up.matrix.cols;
    if (rows <= 0 || cols <= 0) return false;

    if (w_gate.matrix.rows != rows || w_gate.matrix.cols != cols) {
        return false;
    }

    switch (input_dtype) {
        case common::ActivationDType::FP16:
        case common::ActivationDType::BF16:
            break;
        default:
            return false;
    }

    const float* lut_up = GetHostFp8LutV2(w_up.matrix.fp8_format);
    const float* lut_gate = GetHostFp8LutV2(w_gate.matrix.fp8_format);
    if (lut_up == nullptr || lut_gate == nullptr) return false;

    const auto up_weights = w_up.weight.data;
    const auto up_scales = w_up.scale.data;
    const auto gate_weights = w_gate.weight.data;
    const auto gate_scales = w_gate.scale.data;
    const auto* x_u16 = static_cast<const std::uint16_t*>(x);

    const int up_row_block = w_up.scale_meta.row_block;
    const int up_col_block = w_up.scale_meta.col_block;
    const int up_num_col_blocks = w_up.scale_meta.num_col_blocks;

    const int gate_row_block = w_gate.scale_meta.row_block;
    const int gate_col_block = w_gate.scale_meta.col_block;
    const int gate_num_col_blocks = w_gate.scale_meta.num_col_blocks;

    std::vector<float> x_f32(static_cast<std::size_t>(cols));
    for (int k = 0; k < cols; ++k) {
        x_f32[static_cast<std::size_t>(k)] =
            DecodeActivationToFloatV2(input_dtype, x_u16[k]);
    }

    for (int row = 0; row < rows; ++row) {
        const int rb_up = row / up_row_block;
        const int rb_gate = row / gate_row_block;

        const std::size_t row_base =
            static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);

        float up_sum = 0.0f;
        float gate_sum = 0.0f;

        for (int k0 = 0; k0 < cols; k0 += up_col_block) {
            const int k1 = std::min(k0 + up_col_block, cols);

            const int cb_up = k0 / up_col_block;
            const int cb_gate = k0 / gate_col_block;

            const std::size_t up_s_idx =
                static_cast<std::size_t>(rb_up) *
                    static_cast<std::size_t>(up_num_col_blocks) +
                static_cast<std::size_t>(cb_up);

            const std::size_t gate_s_idx =
                static_cast<std::size_t>(rb_gate) *
                    static_cast<std::size_t>(gate_num_col_blocks) +
                static_cast<std::size_t>(cb_gate);

            const float up_scale = up_scales[up_s_idx];
            const float gate_scale = gate_scales[gate_s_idx];

            for (int k = k0; k < k1; ++k) {
                const std::size_t w_idx =
                    row_base + static_cast<std::size_t>(k);
                const float x_val = x_f32[static_cast<std::size_t>(k)];

                up_sum += (lut_up[up_weights[w_idx]] * up_scale) * x_val;
                gate_sum += (lut_gate[gate_weights[w_idx]] * gate_scale) * x_val;
            }
        }

        const float silu_gate = gate_sum / (1.0f + std::exp(-gate_sum));
        h[row] = silu_gate * up_sum;
    }

    return true;
}
