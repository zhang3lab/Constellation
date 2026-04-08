#pragma once

#include <cstdint>
#include <cstring>

#include "common/protocol.h"

inline float DecodeFp16ToFloatV2(std::uint16_t bits) {
    const std::uint32_t sign =
        (static_cast<std::uint32_t>(bits & 0x8000u)) << 16;
    const std::uint32_t exp = (bits >> 10) & 0x1Fu;
    const std::uint32_t frac = bits & 0x03FFu;

    std::uint32_t out = 0;
    if (exp == 0) {
        if (frac == 0) {
            out = sign;
        } else {
            std::uint32_t mant = frac;
            int e = -14;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --e;
            }
            mant &= 0x03FFu;
            out = sign |
                  static_cast<std::uint32_t>((e + 127) << 23) |
                  (mant << 13);
        }
    } else if (exp == 0x1F) {
        out = sign | 0x7F800000u | (frac << 13);
    } else {
        const std::uint32_t e = exp + (127 - 15);
        out = sign | (e << 23) | (frac << 13);
    }

    float f = 0.0f;
    std::memcpy(&f, &out, sizeof(float));
    return f;
}

inline float DecodeBf16ToFloatV2(std::uint16_t bits) {
    const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

inline std::uint16_t EncodeFloatToFp16V2(float x) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));

    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    int exp = static_cast<int>((bits >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t mant = bits & 0x7FFFFFu;

    if (exp <= 0) {
        if (exp < -10) return static_cast<std::uint16_t>(sign);
        mant = (mant | 0x800000u) >> (1 - exp);
        return static_cast<std::uint16_t>(
            sign | ((mant + 0x1000u) >> 13));
    }

    if (exp >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7C00u);
    }

    std::uint32_t half_mant = (mant + 0x1000u) >> 13;
    if (half_mant == 0x0400u) {
        half_mant = 0;
        ++exp;
        if (exp >= 31) {
            return static_cast<std::uint16_t>(sign | 0x7C00u);
        }
    }

    return static_cast<std::uint16_t>(
        sign |
        (static_cast<std::uint32_t>(exp) << 10) |
        half_mant);
}

inline std::uint16_t EncodeFloatToBf16V2(float x) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    bits += 0x00008000u;
    return static_cast<std::uint16_t>(bits >> 16);
}

inline float DecodeActivationToFloatV2(
    common::ActivationDType dtype,
    std::uint16_t bits) {
    switch (dtype) {
        case common::ActivationDType::FP16:
            return DecodeFp16ToFloatV2(bits);
        case common::ActivationDType::BF16:
            return DecodeBf16ToFloatV2(bits);
        default:
            return 0.0f;
    }
}

inline std::uint16_t EncodeActivationFromFloatV2(
    common::ActivationDType dtype,
    float value) {
    switch (dtype) {
        case common::ActivationDType::FP16:
            return EncodeFloatToFp16V2(value);
        case common::ActivationDType::BF16:
            return EncodeFloatToBf16V2(value);
        default:
            return 0;
    }
}
