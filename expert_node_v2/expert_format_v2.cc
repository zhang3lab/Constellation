#include "expert_node_v2/expert_format_v2.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>

namespace {

bool IsExpectedWeightDType(const std::string& dtype) {
    return dtype == "float8_e4m3fn";
}

bool IsExpectedScaleDType(const std::string& dtype) {
    return dtype == "float32";
}

bool HasShape2(const HostTensorV2& ht, std::uint64_t d0, std::uint64_t d1) {
    return ht.meta.shape.size() == 2 &&
           ht.meta.shape[0] == d0 &&
           ht.meta.shape[1] == d1;
}

bool GetShape2(const HostTensorV2& ht, int* d0, int* d1) {
    if (d0 == nullptr || d1 == nullptr) return false;
    if (ht.meta.shape.size() != 2) return false;
    if (ht.meta.shape[0] > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    if (ht.meta.shape[1] > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        return false;
    }

    *d0 = static_cast<int>(ht.meta.shape[0]);
    *d1 = static_cast<int>(ht.meta.shape[1]);
    return true;
}

unsigned long long dim0(const HostTensorV2& t) {
    return t.meta.shape.size() > 0
        ? static_cast<unsigned long long>(t.meta.shape[0])
        : 0ULL;
}

unsigned long long dim1(const HostTensorV2& t) {
    return t.meta.shape.size() > 1
        ? static_cast<unsigned long long>(t.meta.shape[1])
        : 0ULL;
}

void print_one_tensor(const char* prefix, const char* name, const HostTensorV2& t) {
    std::fprintf(stderr,
                 "%s%s: ready=%d bytes=%zu shape=(%llu,%llu) dtype=%s block=(%u,%u)\n",
                 prefix,
                 name,
                 static_cast<int>(t.ready),
                 t.bytes.size(),
                 dim0(t),
                 dim1(t),
                 t.meta.dtype.c_str(),
                 static_cast<unsigned>(t.meta.row_block),
                 static_cast<unsigned>(t.meta.col_block));
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

    if (!IsExpectedWeightDType(weight_ht.meta.dtype)) {
        return false;
    }
    if (!IsExpectedScaleDType(scale_ht.meta.dtype)) {
        return false;
    }

    const int row_block = static_cast<int>(weight_ht.meta.row_block);
    const int col_block = static_cast<int>(weight_ht.meta.col_block);
    if (row_block <= 0 || col_block <= 0) {
        return false;
    }

    if (scale_ht.meta.row_block != weight_ht.meta.row_block ||
        scale_ht.meta.col_block != weight_ht.meta.col_block) {
        return false;
    }

    const int num_row_blocks = ceil_div_int(rows, row_block);
    const int num_col_blocks = ceil_div_int(cols, col_block);

    if (!HasShape2(weight_ht,
                   static_cast<std::uint64_t>(rows),
                   static_cast<std::uint64_t>(cols))) {
        return false;
    }
    if (!HasShape2(scale_ht,
                   static_cast<std::uint64_t>(num_row_blocks),
                   static_cast<std::uint64_t>(num_col_blocks))) {
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

    int w_up_rows = 0, w_up_cols = 0;
    int w_gate_rows = 0, w_gate_cols = 0;
    int w_down_rows = 0, w_down_cols = 0;

    if (!GetShape2(bundle.w_up, &w_up_rows, &w_up_cols)) return false;
    if (!GetShape2(bundle.w_gate, &w_gate_rows, &w_gate_cols)) return false;
    if (!GetShape2(bundle.w_down, &w_down_rows, &w_down_cols)) return false;

    if (w_up_rows != w_gate_rows || w_up_cols != w_gate_cols) {
        return false;
    }
    if (w_down_rows != w_up_cols || w_down_cols != w_up_rows) {
        return false;
    }

    if (!BuildMatrixBlockScaleViewV2(
            bundle.w_up, bundle.w_up_scale, w_up_rows, w_up_cols, &out->w_up)) {
	std::fprintf(stderr, "[BuildExpertWeightsViewV2] w_up failed\n");
        return false;
    }
    if (!BuildMatrixBlockScaleViewV2(
            bundle.w_gate, bundle.w_gate_scale, w_gate_rows, w_gate_cols, &out->w_gate)) {
	std::fprintf(stderr, "[BuildExpertWeightsViewV2] w_gate failed\n");
        return false;
    }
    if (!BuildMatrixBlockScaleViewV2(
            bundle.w_down, bundle.w_down_scale, w_down_rows, w_down_cols, &out->w_down)) {
	std::fprintf(stderr, "[BuildExpertWeightsViewV2] w_down failed\n");
        return false;
    }

    return true;
}

void ExpertTensorBundleV2::debug_print(const char* prefix) const {
    if (prefix == nullptr) {
        prefix = "";
    }

    std::fprintf(stderr,
                 "%sExpertTensorBundleV2 all_ready=%d\n",
                 prefix,
                 static_cast<int>(all_ready()));

    print_one_tensor(prefix, "w_up", w_up);
    print_one_tensor(prefix, "w_up_scale", w_up_scale);
    print_one_tensor(prefix, "w_gate", w_gate);
    print_one_tensor(prefix, "w_gate_scale", w_gate_scale);
    print_one_tensor(prefix, "w_down", w_down);
    print_one_tensor(prefix, "w_down_scale", w_down_scale);
}
