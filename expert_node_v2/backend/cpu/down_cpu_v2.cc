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

    const int row_block = w_down.scale_meta.row_block;
    const int col_block = w_down.scale_meta.col_block;
    const int num_col_blocks = w_down.scale_meta.num_col_blocks;

    for (int row = 0; row < rows; ++row) {
        const int rb = row / row_block;
        const std::size_t row_base =
            static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);

        float sum = 0.0f;

        for (int k0 = 0; k0 < cols; k0 += col_block) {
            const int k1 = std::min(k0 + col_block, cols);
            const int cb = k0 / col_block;

            const std::size_t s_idx =
                static_cast<std::size_t>(rb) *
                    static_cast<std::size_t>(num_col_blocks) +
                static_cast<std::size_t>(cb);

            const float scale = scales[s_idx];

            int k = k0;
            for (; k + 3 < k1; k += 4) {
                const std::size_t w_idx0 = row_base + static_cast<std::size_t>(k + 0);
                const std::size_t w_idx1 = row_base + static_cast<std::size_t>(k + 1);
                const std::size_t w_idx2 = row_base + static_cast<std::size_t>(k + 2);
                const std::size_t w_idx3 = row_base + static_cast<std::size_t>(k + 3);

                const float h0 = h[k + 0];
                const float h1 = h[k + 1];
                const float h2 = h[k + 2];
                const float h3 = h[k + 3];

                const float w0 = lut[weights[w_idx0]] * scale;
                const float w1 = lut[weights[w_idx1]] * scale;
                const float w2 = lut[weights[w_idx2]] * scale;
                const float w3 = lut[weights[w_idx3]] * scale;

                sum += w0 * h0;
                sum += w1 * h1;
                sum += w2 * h2;
                sum += w3 * h3;
            }

            for (; k < k1; ++k) {
                const std::size_t w_idx =
                    row_base + static_cast<std::size_t>(k);
                sum += (lut[weights[w_idx]] * scale) * h[k];
            }
        }

        y_u16[row] = EncodeActivationFromFloatV2(output_dtype, sum);
    }

    return true;
}
