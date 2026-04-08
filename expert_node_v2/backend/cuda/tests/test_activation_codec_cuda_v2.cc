#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <cuda_fp16.h>
#if EXPERT_NODE_V2_ENABLE_CUDA_BF16
#include <cuda_bf16.h>
#endif

#include "expert_node_v2/backend/activation_codec_v2.h"

namespace {

void test_fp16_matches_cuda_bits() {
    const int n = 4096;
    for (int i = 0; i < n; ++i) {
        const float x = std::sin(0.0005f * static_cast<float>(i));

        const std::uint16_t cpu_bits = EncodeFloatToFp16V2(x);

        const __half h = __float2half(x);
        std::uint16_t cuda_bits = 0;
        static_assert(sizeof(cuda_bits) == sizeof(h));
        std::memcpy(&cuda_bits, &h, sizeof(cuda_bits));

        if (cpu_bits != cuda_bits) {
            std::printf(
                "FAIL fp16 bits mismatch i=%d x=%g cpu_bits=0x%04x cuda_bits=0x%04x "
                "cpu_decode=%g cuda_decode=%g\n",
                i,
                x,
                static_cast<unsigned>(cpu_bits),
                static_cast<unsigned>(cuda_bits),
                DecodeFp16ToFloatV2(cpu_bits),
                __half2float(h));
            std::exit(1);
        }
    }
}

void test_bf16_matches_cuda_bits() {
#if EXPERT_NODE_V2_ENABLE_CUDA_BF16
    const int n = 4096;
    for (int i = 0; i < n; ++i) {
        const float x = std::sin(0.0005f * static_cast<float>(i));

        const std::uint16_t cpu_bits = EncodeFloatToBf16V2(x);

        const __nv_bfloat16 b = __float2bfloat16(x);
        std::uint16_t cuda_bits = 0;
        static_assert(sizeof(cuda_bits) == sizeof(b));
        std::memcpy(&cuda_bits, &b, sizeof(cuda_bits));

        if (cpu_bits != cuda_bits) {
            std::printf(
                "FAIL bf16 bits mismatch i=%d x=%g cpu_bits=0x%04x cuda_bits=0x%04x "
                "cpu_decode=%g cuda_decode=%g\n",
                i,
                x,
                static_cast<unsigned>(cpu_bits),
                static_cast<unsigned>(cuda_bits),
                DecodeBf16ToFloatV2(cpu_bits),
                __bfloat162float(b));
            std::exit(1);
        }
    }
#endif
}

}  // namespace

int main() {
    test_fp16_matches_cuda_bits();
    test_bf16_matches_cuda_bits();

    std::printf("test_activation_codec_cuda_v2: PASS\n");
    return 0;
}
