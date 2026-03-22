// expert_pack_clean.cu
//
// Row-major FP8 packer matched to expert_matched.h.
//
// This file provides host-side packing helpers for the current clean pipeline:
//   - row-major packed FP8 weights
//   - per-row per-k-chunk scales
//   - simple nearest-neighbor FP8 host pack path
//
// It intentionally stays simple and correctness-first.

#include "expert.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>

// -----------------------------------------------------------------------------
// Host FP8 decode helpers
// -----------------------------------------------------------------------------

static float decode_fp8_e4m3_host(uint8_t x) {
    const int sign = (x >> 7) & 0x1;
    const int exp  = (x >> 3) & 0xF;
    const int mant = x & 0x7;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        const float frac = static_cast<float>(mant) / 8.0f;
        const float val = std::ldexp(frac, -6);
        return sign ? -val : val;
    }

    if (exp == 0xF) {
        const float inf = std::numeric_limits<float>::infinity();
        const float nan = std::numeric_limits<float>::quiet_NaN();
        if (mant == 0) return sign ? -inf : inf;
        return nan;
    }

    const float frac = 1.0f + static_cast<float>(mant) / 8.0f;
    const float val = std::ldexp(frac, exp - 7);
    return sign ? -val : val;
}

static float decode_fp8_e5m2_host(uint8_t x) {
    const int sign = (x >> 7) & 0x1;
    const int exp  = (x >> 2) & 0x1F;
    const int mant = x & 0x3;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        const float frac = static_cast<float>(mant) / 4.0f;
        const float val = std::ldexp(frac, -14);
        return sign ? -val : val;
    }

    if (exp == 0x1F) {
        const float inf = std::numeric_limits<float>::infinity();
        const float nan = std::numeric_limits<float>::quiet_NaN();
        if (mant == 0) return sign ? -inf : inf;
        return nan;
    }

    const float frac = 1.0f + static_cast<float>(mant) / 4.0f;
    const float val = std::ldexp(frac, exp - 15);
    return sign ? -val : val;
}

static float decode_fp8_host(uint8_t x, Fp8Format fmt) {
    return (fmt == Fp8Format::E5M2) ? decode_fp8_e5m2_host(x)
                                    : decode_fp8_e4m3_host(x);
}

static void build_decode_lut(Fp8Format fmt, float lut[256]) {
    for (int i = 0; i < 256; ++i) {
        lut[i] = decode_fp8_host(static_cast<uint8_t>(i), fmt);
    }
}

// Very simple nearest-neighbor quantizer against decoded LUT values.
// This is correctness-first, not speed-first.
static uint8_t quantize_fp8_nearest(float x, Fp8Format fmt, const float lut[256]) {
    const float maxv = fp8_max_finite(fmt);
    if (x > maxv) x = maxv;
    if (x < -maxv) x = -maxv;

    int best_i = 0;
    float best_err = std::numeric_limits<float>::infinity();

    for (int i = 0; i < 256; ++i) {
        const float qi = lut[i];
        if (!std::isfinite(qi)) continue;
        const float err = std::fabs(qi - x);
        if (err < best_err) {
            best_err = err;
            best_i = i;
        }
    }
    return static_cast<uint8_t>(best_i);
}

// -----------------------------------------------------------------------------
// Public pack helpers
// -----------------------------------------------------------------------------

size_t packed_weights_bytes(int rows, int cols) {
    return (rows > 0 && cols > 0) ? (size_t)rows * cols * sizeof(uint8_t) : 0;
}

size_t packed_scales_bytes(int rows, int cols, int k_chunk) {
    if (rows <= 0 || cols <= 0 || k_chunk <= 0) return 0;
    return (size_t)rows * ceil_div_int(cols, k_chunk) * sizeof(float);
}

static bool pack_impl_from_float(
    const float* src,
    int rows,
    int cols,
    int k_chunk,
    Fp8Format fp8_format,
    PackedRowMajorMatrixHost* out) {
    if (!src || !out) return false;
    if (rows <= 0 || cols <= 0 || k_chunk <= 0) return false;

    out->rows = rows;
    out->cols = cols;
    out->fp8_format = fp8_format;
    out->k_chunk = k_chunk;
    out->num_k_chunks = ceil_div_int(cols, k_chunk);
    out->weights = nullptr;
    out->scales = nullptr;

    const size_t w_bytes = packed_weights_bytes(rows, cols);
    const size_t s_bytes = packed_scales_bytes(rows, cols, k_chunk);

    out->weights = static_cast<uint8_t*>(std::malloc(w_bytes));
    out->scales = static_cast<float*>(std::malloc(s_bytes));
    if (!out->weights || !out->scales) {
        if (out->weights) std::free(out->weights);
        if (out->scales) std::free(out->scales);
        out->weights = nullptr;
        out->scales = nullptr;
        return false;
    }

    float lut[256];
    build_decode_lut(fp8_format, lut);

    const float max_fp = fp8_max_finite(fp8_format);

    for (int r = 0; r < rows; ++r) {
        for (int kc = 0; kc < out->num_k_chunks; ++kc) {
            const int k0 = kc * k_chunk;
            const int k1 = std::min(cols, k0 + k_chunk);

            float amax = 0.0f;
            for (int k = k0; k < k1; ++k) {
                amax = std::max(amax, std::fabs(src[(size_t)r * cols + k]));
            }

            float scale = (amax > 0.0f) ? (amax / max_fp) : 1.0f;
            if (!(scale > 0.0f) || !std::isfinite(scale)) scale = 1.0f;
            out->scales[(size_t)r * out->num_k_chunks + kc] = scale;

            for (int k = k0; k < k1; ++k) {
                const float v_scaled = src[(size_t)r * cols + k] / scale;
                out->weights[(size_t)r * cols + k] =
                    quantize_fp8_nearest(v_scaled, fp8_format, lut);
            }
        }
    }

    return true;
}

bool pack_row_major_fp8_from_float(
    const float* src,
    int rows,
    int cols,
    int k_chunk,
    Fp8Format fp8_format,
    PackedRowMajorMatrixHost* out) {
    return pack_impl_from_float(src, rows, cols, k_chunk, fp8_format, out);
}

bool pack_row_major_fp8_from_half(
    const __half* src,
    int rows,
    int cols,
    int k_chunk,
    Fp8Format fp8_format,
    PackedRowMajorMatrixHost* out) {
    if (!src || !out) return false;
    if (rows <= 0 || cols <= 0 || k_chunk <= 0) return false;

    std::vector<float> tmp((size_t)rows * cols);
    for (size_t i = 0; i < tmp.size(); ++i) {
        tmp[i] = __half2float(src[i]);
    }

    return pack_impl_from_float(tmp.data(), rows, cols, k_chunk, fp8_format, out);
}

bool pack_row_major_fp8_from_fp8_bytes(
    const uint8_t* src,
    int rows,
    int cols,
    int k_chunk,
    Fp8Format src_format,
    Fp8Format packed_format,
    PackedRowMajorMatrixHost* out) {
    if (!src || !out) return false;
    if (rows <= 0 || cols <= 0 || k_chunk <= 0) return false;

    std::vector<float> tmp((size_t)rows * cols);
    for (size_t i = 0; i < tmp.size(); ++i) {
        tmp[i] = decode_fp8_host(src[i], src_format);
    }

    return pack_impl_from_float(
        tmp.data(),
        rows,
        cols,
        k_chunk,
        packed_format,
        out);
}

void free_packed_row_major_matrix_host(PackedRowMajorMatrixHost* p) {
    if (!p) return;

    if (p->weights) std::free(p->weights);
    if (p->scales) std::free(p->scales);

    p->rows = 0;
    p->cols = 0;
    p->fp8_format = Fp8Format::E4M3;
    p->k_chunk = 0;
    p->num_k_chunks = 0;
    p->weights = nullptr;
    p->scales = nullptr;
}
