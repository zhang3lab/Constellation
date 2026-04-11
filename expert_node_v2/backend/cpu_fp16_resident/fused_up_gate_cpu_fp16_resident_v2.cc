#include "expert_node_v2/backend/cpu_fp16_resident/fused_up_gate_cpu_fp16_resident_v2.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#if !defined(__AVX2__) || !defined(__F16C__)
#error "cpu_fp16_resident backend requires AVX2 and F16C"
#endif

#include <immintrin.h>

#include "expert_node_v2/backend/activation_codec_v2.h"

namespace {

inline float HSum256ToScalar(__m256 v) {
    alignas(32) float buf[8];
    _mm256_store_ps(buf, v);
    return buf[0] + buf[1] + buf[2] + buf[3] +
           buf[4] + buf[5] + buf[6] + buf[7];
}

inline float SiLU(float x) {
    return x / (1.0f + std::exp(-x));
}

}  // namespace

bool RunFusedUpGateCpuFp16ResidentV2(
    const MatrixBlockScaleViewV2& w_up,
    const MatrixBlockScaleViewV2& w_gate,
    const void* x,
    common::ActivationDType input_dtype,
    float* h,
    int omp_threads) {
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

    const std::size_t elems =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    const std::size_t bytes_needed = elems * sizeof(std::uint16_t);

    // cpu_fp16_resident backend contract:
    // w_up.weight.data / w_gate.weight.data store row-major FP16 resident payload directly.
    if (w_up.weight.data.size() != bytes_needed) return false;
    if (w_gate.weight.data.size() != bytes_needed) return false;

    const auto* up_weights =
        reinterpret_cast<const std::uint16_t*>(w_up.weight.data.data());
    const auto* gate_weights =
        reinterpret_cast<const std::uint16_t*>(w_gate.weight.data.data());

    if (omp_threads <= 0) omp_threads = 1;

    const auto* x_u16 = static_cast<const std::uint16_t*>(x);
    std::vector<float> x_f32(static_cast<std::size_t>(cols));
    for (int k = 0; k < cols; ++k) {
        x_f32[static_cast<std::size_t>(k)] =
            DecodeActivationToFloatV2(input_dtype, x_u16[k]);
    }

#pragma omp parallel for schedule(static) num_threads(omp_threads)
    for (int row = 0; row < rows; ++row) {
        const std::size_t row_base =
            static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);

        __m256 up_acc0 = _mm256_setzero_ps();
        __m256 up_acc1 = _mm256_setzero_ps();
        __m256 gate_acc0 = _mm256_setzero_ps();
        __m256 gate_acc1 = _mm256_setzero_ps();

        int k = 0;

        // unroll2 + acc2
        for (; k + 15 < cols; k += 16) {
            const std::uint16_t* up_src0 =
                up_weights + row_base + static_cast<std::size_t>(k + 0);
            const std::uint16_t* up_src1 =
                up_weights + row_base + static_cast<std::size_t>(k + 8);
            const std::uint16_t* gate_src0 =
                gate_weights + row_base + static_cast<std::size_t>(k + 0);
            const std::uint16_t* gate_src1 =
                gate_weights + row_base + static_cast<std::size_t>(k + 8);

            const __m128i up_h8_0 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(up_src0));
            const __m128i up_h8_1 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(up_src1));
            const __m128i gate_h8_0 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(gate_src0));
            const __m128i gate_h8_1 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(gate_src1));

            const __m256 up_wv0 = _mm256_cvtph_ps(up_h8_0);
            const __m256 up_wv1 = _mm256_cvtph_ps(up_h8_1);
            const __m256 gate_wv0 = _mm256_cvtph_ps(gate_h8_0);
            const __m256 gate_wv1 = _mm256_cvtph_ps(gate_h8_1);

            const __m256 xv0 = _mm256_loadu_ps(x_f32.data() + k + 0);
            const __m256 xv1 = _mm256_loadu_ps(x_f32.data() + k + 8);

#if defined(__FMA__)
            up_acc0 = _mm256_fmadd_ps(up_wv0, xv0, up_acc0);
            up_acc1 = _mm256_fmadd_ps(up_wv1, xv1, up_acc1);
            gate_acc0 = _mm256_fmadd_ps(gate_wv0, xv0, gate_acc0);
            gate_acc1 = _mm256_fmadd_ps(gate_wv1, xv1, gate_acc1);
#else
            up_acc0 = _mm256_add_ps(up_acc0, _mm256_mul_ps(up_wv0, xv0));
            up_acc1 = _mm256_add_ps(up_acc1, _mm256_mul_ps(up_wv1, xv1));
            gate_acc0 = _mm256_add_ps(gate_acc0, _mm256_mul_ps(gate_wv0, xv0));
            gate_acc1 = _mm256_add_ps(gate_acc1, _mm256_mul_ps(gate_wv1, xv1));
#endif
        }

        __m256 up_acc = _mm256_add_ps(up_acc0, up_acc1);
        __m256 gate_acc = _mm256_add_ps(gate_acc0, gate_acc1);

        for (; k + 7 < cols; k += 8) {
            const std::uint16_t* up_src =
                up_weights + row_base + static_cast<std::size_t>(k);
            const std::uint16_t* gate_src =
                gate_weights + row_base + static_cast<std::size_t>(k);

            const __m128i up_h8 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(up_src));
            const __m128i gate_h8 =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(gate_src));

            const __m256 up_wv = _mm256_cvtph_ps(up_h8);
            const __m256 gate_wv = _mm256_cvtph_ps(gate_h8);
            const __m256 xv = _mm256_loadu_ps(x_f32.data() + k);

#if defined(__FMA__)
            up_acc = _mm256_fmadd_ps(up_wv, xv, up_acc);
            gate_acc = _mm256_fmadd_ps(gate_wv, xv, gate_acc);
#else
            up_acc = _mm256_add_ps(up_acc, _mm256_mul_ps(up_wv, xv));
            gate_acc = _mm256_add_ps(gate_acc, _mm256_mul_ps(gate_wv, xv));
#endif
        }

        float up_sum = HSum256ToScalar(up_acc);
        float gate_sum = HSum256ToScalar(gate_acc);

        for (; k < cols; ++k) {
            const std::uint16_t up_bits =
                up_weights[row_base + static_cast<std::size_t>(k)];
            const std::uint16_t gate_bits =
                gate_weights[row_base + static_cast<std::size_t>(k)];

            const __m128i up_h1 = _mm_cvtsi32_si128(static_cast<int>(up_bits));
            const __m128i gate_h1 = _mm_cvtsi32_si128(static_cast<int>(gate_bits));

            const float up_w = _mm_cvtss_f32(_mm_cvtph_ps(up_h1));
            const float gate_w = _mm_cvtss_f32(_mm_cvtph_ps(gate_h1));
            const float xv = x_f32[static_cast<std::size_t>(k)];

            up_sum += up_w * xv;
            gate_sum += gate_w * xv;
        }

        h[row] = SiLU(gate_sum) * up_sum;
    }

    return true;
}
