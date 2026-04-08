#include "expert_node_v2/backend/cpu/down_cpu_v2.h"

#include <cstddef>
#include <cstdint>

#include "expert_node_v2/backend/activation_codec_v2.h"
#include "expert_node_v2/backend/fp8_lut_v2.h"

bool RunDownCpuV2(
    const MatrixBlockScaleViewV2& w_down,
    const float* h,
    void* y,
    common::ActivationDType output_dtype) {
    if (h == nullptr || y == nullptr) return false;

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
    auto* y_u16 = static_cast<std::uint16_t*>(y);

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

            const float scale = scales[s_idx];
            const float w = lut[weights[w_idx]] * scale;
            sum += w * h[k];
        }

        y_u16[row] = EncodeActivationFromFloatV2(output_dtype, sum);
    }

    return true;
}
