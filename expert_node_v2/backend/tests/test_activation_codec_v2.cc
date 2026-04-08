#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "common/protocol.h"
#include "expert_node_v2/backend/activation_codec_v2.h"

namespace {

bool is_nan_f(float x) {
    return std::isnan(x);
}

bool is_inf_f(float x) {
    return std::isinf(x);
}

bool almost_equal(float a, float b, float atol) {
    if (is_nan_f(a) && is_nan_f(b)) return true;
    if (is_inf_f(a) || is_inf_f(b)) return a == b;
    return std::fabs(a - b) <= atol;
}

void test_fp16_basic_decode() {
    struct Case {
        std::uint16_t bits;
        float expect;
        const char* name;
    };

    const Case cases[] = {
        {0x0000u, 0.0f, "fp16 +0"},
        {0x8000u, -0.0f, "fp16 -0"},
        {0x3c00u, 1.0f, "fp16 1"},
        {0xbc00u, -1.0f, "fp16 -1"},
        {0x3800u, 0.5f, "fp16 0.5"},
        {0x3400u, 0.25f, "fp16 0.25"},
        {0x4000u, 2.0f, "fp16 2"},
    };

    for (const auto& c : cases) {
        const float got = DecodeFp16ToFloatV2(c.bits);
        if (!almost_equal(got, c.expect, 0.0f)) {
            std::printf("FAIL %s bits=0x%04x got=%g expect=%g\n",
                        c.name,
                        static_cast<unsigned>(c.bits),
                        got,
                        c.expect);
            std::exit(1);
        }
    }
}

void test_bf16_basic_decode() {
    struct Case {
        std::uint16_t bits;
        float expect;
        const char* name;
    };

    const Case cases[] = {
        {0x0000u, 0.0f, "bf16 +0"},
        {0x8000u, -0.0f, "bf16 -0"},
        {0x3f80u, 1.0f, "bf16 1"},
        {0xbf80u, -1.0f, "bf16 -1"},
        {0x3f00u, 0.5f, "bf16 0.5"},
        {0x3e80u, 0.25f, "bf16 0.25"},
        {0x4000u, 2.0f, "bf16 2"},
    };

    for (const auto& c : cases) {
        const float got = DecodeBf16ToFloatV2(c.bits);
        if (!almost_equal(got, c.expect, 0.0f)) {
            std::printf("FAIL %s bits=0x%04x got=%g expect=%g\n",
                        c.name,
                        static_cast<unsigned>(c.bits),
                        got,
                        c.expect);
            std::exit(1);
        }
    }
}

void test_fp16_roundtrip_sanity() {
    const std::vector<float> vals = {
        0.0f,
        -0.0f,
        1.0f,
        -1.0f,
        0.5f,
        0.25f,
        0.499914f,
        65504.0f,
        1.0e-3f,
        -1.0e-3f,
        1.0e-4f,
        -1.0e-4f,
    };

    for (float x : vals) {
        const std::uint16_t bits = EncodeFloatToFp16V2(x);
        const float y = DecodeFp16ToFloatV2(bits);

        const std::uint16_t expect_bits = EncodeFloatToFp16V2(x);
        const float expect = DecodeFp16ToFloatV2(expect_bits);

        if (!almost_equal(y, expect, 0.0f)) {
            std::printf("FAIL fp16 roundtrip x=%g bits=0x%04x got=%g expect=%g\n",
                        x,
                        static_cast<unsigned>(bits),
                        y,
                        expect);
            std::exit(1);
        }
    }
}

void test_bf16_roundtrip_sanity() {
    const std::vector<float> vals = {
        0.0f,
        -0.0f,
        1.0f,
        -1.0f,
        0.5f,
        0.25f,
        0.499914f,
        1.0e-3f,
        -1.0e-3f,
        1.0e10f,
        -1.0e10f,
    };

    for (float x : vals) {
        const std::uint16_t bits = EncodeFloatToBf16V2(x);
        const float y = DecodeBf16ToFloatV2(bits);
        const float expect = DecodeBf16ToFloatV2(EncodeFloatToBf16V2(x));

        if (!almost_equal(y, expect, 0.0f)) {
            std::printf("FAIL bf16 roundtrip x=%g bits=0x%04x got=%g expect=%g\n",
                        x,
                        static_cast<unsigned>(bits),
                        y,
                        expect);
            std::exit(1);
        }
    }
}

void test_activation_dispatch() {
    const float x = 0.499914f;

    {
        const std::uint16_t bits =
            EncodeActivationFromFloatV2(common::ActivationDType::FP16, x);
        const float y =
            DecodeActivationToFloatV2(common::ActivationDType::FP16, bits);
        const float expect = DecodeFp16ToFloatV2(EncodeFloatToFp16V2(x));
        if (!almost_equal(y, expect, 0.0f)) {
            std::printf("FAIL activation dispatch fp16 x=%g got=%g expect=%g\n",
                        x,
                        y,
                        expect);
            std::exit(1);
        }
    }

    {
        const std::uint16_t bits =
            EncodeActivationFromFloatV2(common::ActivationDType::BF16, x);
        const float y =
            DecodeActivationToFloatV2(common::ActivationDType::BF16, bits);
        const float expect = DecodeBf16ToFloatV2(EncodeFloatToBf16V2(x));
        if (!almost_equal(y, expect, 0.0f)) {
            std::printf("FAIL activation dispatch bf16 x=%g got=%g expect=%g\n",
                        x,
                        y,
                        expect);
            std::exit(1);
        }
    }
}

}  // namespace

int main() {
    test_fp16_basic_decode();
    test_bf16_basic_decode();
    test_fp16_roundtrip_sanity();
    test_bf16_roundtrip_sanity();
    test_activation_dispatch();

    std::printf("test_activation_codec_v2: PASS\n");
    return 0;
}
