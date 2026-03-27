#include "expert_node_v2/backend/cuda/fp8_decode_lut_v2.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

struct DeviceLutState {
    bool uploaded = false;
    float* lut_ieee_e4m3 = nullptr;
    float* lut_ieee_e5m2 = nullptr;
    float* lut_torch_e4m3fn = nullptr;
};

static bool g_host_luts_built = false;
static float g_lut_ieee_e4m3_host[256];
static float g_lut_ieee_e5m2_host[256];
static float g_lut_torch_e4m3fn_host[256];
static std::vector<DeviceLutState> g_lut_states;

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

DeviceLutState* get_state_for_device(int device_id) {
    if (device_id < 0) return nullptr;
    if (static_cast<int>(g_lut_states.size()) <= device_id) {
        g_lut_states.resize(static_cast<std::size_t>(device_id + 1));
    }
    return &g_lut_states[static_cast<std::size_t>(device_id)];
}

bool alloc_and_copy_lut(const float host_lut[256], float** dev_ptr, cudaStream_t stream) {
    if (dev_ptr == nullptr) return false;
    *dev_ptr = nullptr;

    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(dev_ptr), 256 * sizeof(float));
    if (err != cudaSuccess) {
        return false;
    }

    err = cudaMemcpyAsync(
        *dev_ptr,
        host_lut,
        256 * sizeof(float),
        cudaMemcpyHostToDevice,
        stream);
    if (err != cudaSuccess) {
        cudaFree(*dev_ptr);
        *dev_ptr = nullptr;
        return false;
    }

    return true;
}

bool init_luts_for_device(int device_id, cudaStream_t stream) {
    build_host_luts();

    DeviceLutState* state = get_state_for_device(device_id);
    if (state == nullptr) return false;
    if (state->uploaded) return true;

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        return false;
    }

    float* lut_ieee_e4m3 = nullptr;
    float* lut_ieee_e5m2 = nullptr;
    float* lut_torch_e4m3fn = nullptr;

    if (!alloc_and_copy_lut(g_lut_ieee_e4m3_host, &lut_ieee_e4m3, stream)) {
        goto cleanup;
    }
    if (!alloc_and_copy_lut(g_lut_ieee_e5m2_host, &lut_ieee_e5m2, stream)) {
        goto cleanup;
    }
    if (!alloc_and_copy_lut(g_lut_torch_e4m3fn_host, &lut_torch_e4m3fn, stream)) {
        goto cleanup;
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        goto cleanup;
    }

    state->lut_ieee_e4m3 = lut_ieee_e4m3;
    state->lut_ieee_e5m2 = lut_ieee_e5m2;
    state->lut_torch_e4m3fn = lut_torch_e4m3fn;
    state->uploaded = true;
    return true;

cleanup:
    if (lut_ieee_e4m3 != nullptr) cudaFree(lut_ieee_e4m3);
    if (lut_ieee_e5m2 != nullptr) cudaFree(lut_ieee_e5m2);
    if (lut_torch_e4m3fn != nullptr) cudaFree(lut_torch_e4m3fn);
    return false;
}

}  // namespace

const float* GetOrInitFp8DecodeLutCudaV2(
    int device_id,
    Fp8Format fmt,
    cudaStream_t stream) {
    DeviceLutState* state = get_state_for_device(device_id);
    if (state == nullptr) return nullptr;

    if (!state->uploaded) {
        if (!init_luts_for_device(device_id, stream)) {
            return nullptr;
        }
    }

    switch (fmt) {
        case Fp8Format::IEEE_E4M3:
            return state->lut_ieee_e4m3;
        case Fp8Format::IEEE_E5M2:
            return state->lut_ieee_e5m2;
        case Fp8Format::TORCH_E4M3FN:
            return state->lut_torch_e4m3fn;
        default:
            return nullptr;
    }
}

void ResetFp8DecodeLutsCudaV2() {
    for (auto& s : g_lut_states) {
        if (s.lut_ieee_e4m3 != nullptr) cudaFree(s.lut_ieee_e4m3);
        if (s.lut_ieee_e5m2 != nullptr) cudaFree(s.lut_ieee_e5m2);
        if (s.lut_torch_e4m3fn != nullptr) cudaFree(s.lut_torch_e4m3fn);
        s = DeviceLutState{};
    }
    g_lut_states.clear();
}
