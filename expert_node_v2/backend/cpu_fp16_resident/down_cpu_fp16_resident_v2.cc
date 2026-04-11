#include "expert_node_v2/backend/cpu_fp16_resident/down_cpu_fp16_resident_v2.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#if !defined(__AVX2__) || !defined(__F16C__)
#error "cpu_fp16_resident backend requires AVX2 and F16C"
#endif

#include <immintrin.h>

#include "expert_node_v2/backend/activation_codec_v2.h"

// For cpu_fp16_resident backend, w_down.weight.data stores row-major FP16
// resident weights directly, not FP8 blockscale payload.
namespace {

bool RunDownCpuFp16ResidentKernelV2(
    const std::uint16_t* weights,
    int rows,
    int cols,
    const float* h,
    float* y,
    int omp_threads) {
    if (weights == nullptr || h == nullptr || y == nullptr) return false;
    if (rows <= 0 || cols <= 0) return false;
    if (omp_threads <= 0) omp_threads = 1;

#pragma omp parallel for schedule(static) num_threads(omp_threads)
    for (int row = 0; row < rows; ++row) {
        const std::size_t row_base =
            static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        int k = 0;

        for (; k + 31 < cols; k += 32) {
            const std::uint16_t* src0 =
                weights + row_base + static_cast<std::size_t>(k + 0);
            const std::uint16_t* src1 =
                weights + row_base + static_cast<std::size_t>(k + 8);
            const std::uint16_t* src2 =
                weights + row_base + static_cast<std::size_t>(k + 16);
            const std::uint16_t* src3 =
                weights + row_base + static_cast<std::size_t>(k + 24);

            const __m128i h8_0 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(src0));
            const __m128i h8_1 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(src1));
            const __m128i h8_2 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(src2));
            const __m128i h8_3 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(src3));

            const __m256 wv0 = _mm256_cvtph_ps(h8_0);
            const __m256 wv1 = _mm256_cvtph_ps(h8_1);
            const __m256 wv2 = _mm256_cvtph_ps(h8_2);
            const __m256 wv3 = _mm256_cvtph_ps(h8_3);

            const __m256 hv0 = _mm256_loadu_ps(h + k + 0);
            const __m256 hv1 = _mm256_loadu_ps(h + k + 8);
            const __m256 hv2 = _mm256_loadu_ps(h + k + 16);
            const __m256 hv3 = _mm256_loadu_ps(h + k + 24);

#if defined(__FMA__)
            acc0 = _mm256_fmadd_ps(wv0, hv0, acc0);
            acc1 = _mm256_fmadd_ps(wv1, hv1, acc1);
            acc2 = _mm256_fmadd_ps(wv2, hv2, acc2);
            acc3 = _mm256_fmadd_ps(wv3, hv3, acc3);
#else
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(wv0, hv0));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(wv1, hv1));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(wv2, hv2));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(wv3, hv3));
#endif
        }

        __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
                                   _mm256_add_ps(acc2, acc3));

        for (; k + 15 < cols; k += 16) {
            const std::uint16_t* src0 =
                weights + row_base + static_cast<std::size_t>(k + 0);
            const std::uint16_t* src1 =
                weights + row_base + static_cast<std::size_t>(k + 8);

            const __m128i h8_0 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(src0));
            const __m128i h8_1 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(src1));

            const __m256 wv0 = _mm256_cvtph_ps(h8_0);
            const __m256 wv1 = _mm256_cvtph_ps(h8_1);

            const __m256 hv0 = _mm256_loadu_ps(h + k + 0);
            const __m256 hv1 = _mm256_loadu_ps(h + k + 8);

#if defined(__FMA__)
            acc = _mm256_fmadd_ps(wv0, hv0, acc);
            acc = _mm256_fmadd_ps(wv1, hv1, acc);
#else
            acc = _mm256_add_ps(acc, _mm256_mul_ps(wv0, hv0));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(wv1, hv1));
#endif
        }

        for (; k + 7 < cols; k += 8) {
            const std::uint16_t* src =
                weights + row_base + static_cast<std::size_t>(k);

            const __m128i h8 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
            const __m256 wv = _mm256_cvtph_ps(h8);
            const __m256 hv = _mm256_loadu_ps(h + k);

#if defined(__FMA__)
            acc = _mm256_fmadd_ps(wv, hv, acc);
#else
            acc = _mm256_add_ps(acc, _mm256_mul_ps(wv, hv));
#endif
        }

        alignas(32) float acc_buf[8];
        _mm256_store_ps(acc_buf, acc);
        float sum =
            acc_buf[0] + acc_buf[1] + acc_buf[2] + acc_buf[3] +
            acc_buf[4] + acc_buf[5] + acc_buf[6] + acc_buf[7];

        for (; k < cols; ++k) {
            const std::uint16_t bits =
                weights[row_base + static_cast<std::size_t>(k)];

            const __m128i h1 = _mm_cvtsi32_si128(static_cast<int>(bits));
            const __m128 f4 = _mm_cvtph_ps(h1);
            const float w = _mm_cvtss_f32(f4);

            sum += w * h[k];
        }

        y[row] = sum;
    }

    return true;
#endif
}

}  // namespace

bool RunDownCpuFp16ResidentV2(
    const MatrixBlockScaleViewV2& w_down,
    const float* h,
    void* y,
    common::ActivationDType output_dtype,
    int omp_threads) {
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

    const std::size_t elems =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    const std::size_t bytes_needed = elems * sizeof(std::uint16_t);

    if (w_down.weight.data.size() != bytes_needed) {
        return false;
    }

    const auto* weights_fp16 =
        reinterpret_cast<const std::uint16_t*>(w_down.weight.data.data());

    std::vector<float> y_f32(static_cast<std::size_t>(rows), 0.0f);
    if (!RunDownCpuFp16ResidentKernelV2(
            weights_fp16,
            rows,
            cols,
            h,
            y_f32.data(),
            omp_threads)) {
        return false;
    }

    auto* y_u16 = static_cast<std::uint16_t*>(y);
    for (int row = 0; row < rows; ++row) {
        y_u16[row] = EncodeActivationFromFloatV2(output_dtype, y_f32[row]);
    }

    return true;
}
