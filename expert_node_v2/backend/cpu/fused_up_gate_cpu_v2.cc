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

    for (int row = 0; row < rows; ++row) {
        const int rb_up = row / w_up.scale_meta.row_block;
        const int rb_gate = row / w_gate.scale_meta.row_block;

        float up_sum = 0.0f;
        float gate_sum = 0.0f;

        for (int k = 0; k < cols; ++k) {
            const int cb_up = k / w_up.scale_meta.col_block;
            const int cb_gate = k / w_gate.scale_meta.col_block;

            const std::size_t up_w_idx =
                static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) +
                static_cast<std::size_t>(k);
            const std::size_t up_s_idx =
                static_cast<std::size_t>(rb_up) *
                    static_cast<std::size_t>(w_up.scale_meta.num_col_blocks) +
                static_cast<std::size_t>(cb_up);

            const std::size_t gate_w_idx =
                static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) +
                static_cast<std::size_t>(k);
            const std::size_t gate_s_idx =
                static_cast<std::size_t>(rb_gate) *
                    static_cast<std::size_t>(w_gate.scale_meta.num_col_blocks) +
                static_cast<std::size_t>(cb_gate);

            const float x_val = DecodeActivationToFloatV2(input_dtype, x_u16[k]);

            const float up_wv =
                lut_up[up_weights[up_w_idx]] * up_scales[up_s_idx];
            const float gate_wv =
                lut_gate[gate_weights[gate_w_idx]] * gate_scales[gate_s_idx];

            up_sum += up_wv * x_val;
            gate_sum += gate_wv * x_val;
        }

        const float silu_gate = gate_sum / (1.0f + std::exp(-gate_sum));
        h[row] = silu_gate * up_sum;
    }

    return true;
}
