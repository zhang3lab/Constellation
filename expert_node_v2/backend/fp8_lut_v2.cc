#include "expert_node_v2/backend/fp8_lut_v2.h"

#include <cmath>
#include <cstdint>

namespace {

bool g_host_luts_built = false;
float g_lut_ieee_e4m3_host[256];
float g_lut_ieee_e5m2_host[256];
float g_lut_torch_e4m3fn_host[256];

float decode_ieee_e4m3_byte(std::uint8_t v) {
    const int sign = (v >> 7) & 0x1;
    const int exp = (v >> 3) & 0xF;
    const int mant = v & 0x7;
    const float s = sign ? -1.0f : 1.0f;
    const int bias = 7;

    if (exp == 0) {
        if (mant == 0) return s * 0.0f;
        return s * std::ldexp(static_cast<float>(mant) / 8.0f, 1 - bias);
    }
    if (exp == 0xF) {
        return mant == 0 ? s * INFINITY : NAN;
    }
    return s * std::ldexp(1.0f + static_cast<float>(mant) / 8.0f, exp - bias);
}

float decode_ieee_e5m2_byte(std::uint8_t v) {
    const int sign = (v >> 7) & 0x1;
    const int exp = (v >> 2) & 0x1F;
    const int mant = v & 0x3;
    const float s = sign ? -1.0f : 1.0f;
    const int bias = 15;

    if (exp == 0) {
        if (mant == 0) return s * 0.0f;
        return s * std::ldexp(static_cast<float>(mant) / 4.0f, 1 - bias);
    }
    if (exp == 0x1F) {
        return mant == 0 ? s * INFINITY : NAN;
    }
    return s * std::ldexp(1.0f + static_cast<float>(mant) / 4.0f, exp - bias);
}

float decode_torch_e4m3fn_byte(std::uint8_t v) {
    const int sign = (v >> 7) & 0x1;
    const int exp = (v >> 3) & 0xF;
    const int mant = v & 0x7;
    const float s = sign ? -1.0f : 1.0f;
    const int bias = 7;

    if (exp == 0) {
        if (mant == 0) return s * 0.0f;
        return s * std::ldexp(static_cast<float>(mant) / 8.0f, 1 - bias);
    }

    // finite-only torch.float8_e4m3fn
    if (exp == 0xF) {
        if (mant == 0x7) return s * 448.0f;
        return s * std::ldexp(1.0f + static_cast<float>(mant) / 8.0f, exp - bias);
    }

    return s * std::ldexp(1.0f + static_cast<float>(mant) / 8.0f, exp - bias);
}

void build_host_luts() {
    if (g_host_luts_built) return;

    for (int i = 0; i < 256; ++i) {
        const std::uint8_t v = static_cast<std::uint8_t>(i);
        g_lut_ieee_e4m3_host[i] = decode_ieee_e4m3_byte(v);
        g_lut_ieee_e5m2_host[i] = decode_ieee_e5m2_byte(v);
        g_lut_torch_e4m3fn_host[i] = decode_torch_e4m3fn_byte(v);
    }

    g_host_luts_built = true;
}

}  // namespace

float DecodeFp8ByteV2(Fp8Format fmt, std::uint8_t v) {
    switch (fmt) {
        case Fp8Format::IEEE_E4M3:
            return decode_ieee_e4m3_byte(v);
        case Fp8Format::IEEE_E5M2:
            return decode_ieee_e5m2_byte(v);
        case Fp8Format::TORCH_E4M3FN:
            return decode_torch_e4m3fn_byte(v);
        default:
            return NAN;
    }
}

const float* GetHostFp8LutV2(Fp8Format fmt) {
    build_host_luts();

    switch (fmt) {
        case Fp8Format::IEEE_E4M3:
            return g_lut_ieee_e4m3_host;
        case Fp8Format::IEEE_E5M2:
            return g_lut_ieee_e5m2_host;
        case Fp8Format::TORCH_E4M3FN:
            return g_lut_torch_e4m3fn_host;
        default:
            return nullptr;
    }
}
