#include "expert_node_v2/expert_format_v2.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace {

bool IsExpectedWeightDType(const std::string& dtype) {
    return dtype == "torch.float8_e4m3fn";
}

bool IsExpectedScaleDType(const std::string& dtype) {
    return dtype == "torch.float32";
}

bool HasShape2(const HostTensorV2& ht, std::int64_t d0, std::int64_t d1) {
    return ht.shape.size() == 2 && ht.shape[0] == d0 && ht.shape[1] == d1;
}

}  // namespace

bool BuildMatrixBlockScaleViewV2(
    const HostTensorV2& weight_ht,
    const HostTensorV2& scale_ht,
    int rows,
    int cols,
    MatrixBlockScaleViewV2* out) {
    if (out == nullptr) return false;
    if (rows <= 0 || cols <= 0) return false;
    if (!weight_ht.ready || !scale_ht.ready) return false;

    if (!IsExpectedWeightDType(weight_ht.dtype)) {
        return false;
    }
    if (!IsExpectedScaleDType(scale_ht.dtype)) {
        return false;
    }

    const int row_block = 128;
    const int col_block = 128;
    const int num_row_blocks = ceil_div_int(rows, row_block);
    const int num_col_blocks = ceil_div_int(cols, col_block);

    if (!HasShape2(weight_ht, rows, cols)) {
        return false;
    }
    if (!HasShape2(scale_ht, num_row_blocks, num_col_blocks)) {
        return false;
    }

    const std::size_t weight_bytes_expected =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    if (weight_ht.bytes.size() != weight_bytes_expected) {
        return false;
    }

    const std::size_t scale_elems_expected =
        static_cast<std::size_t>(num_row_blocks) * static_cast<std::size_t>(num_col_blocks);
    const std::size_t scale_bytes_expected = scale_elems_expected * sizeof(float);
    if (scale_ht.bytes.size() != scale_bytes_expected) {
        return false;
    }

    out->matrix.rows = rows;
    out->matrix.cols = cols;
    out->matrix.fp8_format = Fp8Format::TORCH_E4M3FN;

    out->weight.data = std::span<const std::uint8_t>(weight_ht.bytes);

    out->scale_meta.row_block = row_block;
    out->scale_meta.col_block = col_block;
    out->scale_meta.num_row_blocks = num_row_blocks;
    out->scale_meta.num_col_blocks = num_col_blocks;

    out->scale.data = std::span<const float>(
        reinterpret_cast<const float*>(scale_ht.bytes.data()),
        scale_elems_expected);

    return true;
}

bool BuildExpertWeightsViewV2(
    const ExpertTensorBundleV2& bundle,
    ExpertWeightsViewV2* out) {
    if (out == nullptr) return false;
    if (!bundle.all_ready()) return false;

    if (!BuildMatrixBlockScaleViewV2(bundle.w_up, bundle.w_up_scale, 2048, 7168, &out->w_up)) {
        return false;
    }
    if (!BuildMatrixBlockScaleViewV2(bundle.w_gate, bundle.w_gate_scale, 2048, 7168, &out->w_gate)) {
        return false;
    }
    if (!BuildMatrixBlockScaleViewV2(bundle.w_down, bundle.w_down_scale, 7168, 2048, &out->w_down)) {
        return false;
    }

    return true;
}
