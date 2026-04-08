#include "expert_node_v2/backend/expert_reference_v2.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "expert_node_v2/backend/activation_codec_v2.h"
#include "expert_node_v2/backend/fp8_lut_v2.h"

bool RunFusedUpGateReferenceV2(
    const MatrixBlockScaleViewV2& w_up,
    const MatrixBlockScaleViewV2& w_gate,
    const void* x,
    common::ActivationDType input_dtype,
    std::vector<float>* out_h) {
    if (x == nullptr || out_h == nullptr) return false;

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

    out_h->assign(static_cast<std::size_t>(rows), 0.0f);

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
        (*out_h)[static_cast<std::size_t>(row)] = silu_gate * up_sum;
    }

    return true;
}

bool RunDownReferenceV2(
    const MatrixBlockScaleViewV2& w_down,
    const float* h,
    common::ActivationDType output_dtype,
    std::vector<std::uint8_t>* out_y_bytes) {
    if (h == nullptr || out_y_bytes == nullptr) return false;

    const int rows = w_down.matrix.rows;
    const int cols = w_down.matrix.cols;
    if (rows <= 0 || cols <= 0) return false;

    switch (output_dtype) {
        case common::ActivationDType::FP16:
        case common::ActivationDType::BF16:
            break;
        default:
            return false;
    }

    const float* lut = GetHostFp8LutV2(w_down.matrix.fp8_format);
    if (lut == nullptr) return false;

    const auto weights = w_down.weight.data;
    const auto scales = w_down.scale.data;

    out_y_bytes->assign(
        static_cast<std::size_t>(rows) * sizeof(std::uint16_t), 0);
    auto* y_u16 = reinterpret_cast<std::uint16_t*>(out_y_bytes->data());

    for (int row = 0; row < rows; ++row) {
        const int rb = row / w_down.scale_meta.row_block;
        float sum = 0.0f;

        for (int k = 0; k < cols; ++k) {
            const int cb = k / w_down.scale_meta.col_block;

            const std::size_t w_idx =
                static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) +
                static_cast<std::size_t>(k);
            const std::size_t s_idx =
                static_cast<std::size_t>(rb) *
                    static_cast<std::size_t>(w_down.scale_meta.num_col_blocks) +
                static_cast<std::size_t>(cb);

            const float w = lut[weights[w_idx]] * scales[s_idx];
            sum += w * h[k];
        }

        y_u16[row] = EncodeActivationFromFloatV2(output_dtype, sum);
    }

    return true;
}

bool RunExpertReferenceV2(
    const ExpertWeightsViewV2& weights,
    const void* x,
    common::ActivationDType input_dtype,
    common::ActivationDType output_dtype,
    std::vector<std::uint8_t>* out_y_bytes,
    std::vector<float>* out_h_debug) {
    if (out_y_bytes == nullptr) return false;

    std::vector<float> h;
    if (!RunFusedUpGateReferenceV2(
            weights.w_up,
            weights.w_gate,
            x,
            input_dtype,
            &h)) {
        return false;
    }

    if (!RunDownReferenceV2(
            weights.w_down,
            h.data(),
            output_dtype,
            out_y_bytes)) {
        return false;
    }

    if (out_h_debug != nullptr) {
        *out_h_debug = std::move(h);
    }

    return true;
}
