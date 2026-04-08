#include "expert_node_v2/backend/cuda/fp8_decode_lut_v2.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

#include "expert_node_v2/backend/fp8_lut_v2.h"

namespace {

struct DeviceLutState {
    bool uploaded = false;
    float* lut_ieee_e4m3 = nullptr;
    float* lut_ieee_e5m2 = nullptr;
    float* lut_torch_e4m3fn = nullptr;
};

static std::vector<DeviceLutState> g_lut_states;

DeviceLutState* get_state_for_device(int device_id) {
    if (device_id < 0) return nullptr;
    if (static_cast<int>(g_lut_states.size()) <= device_id) {
        g_lut_states.resize(static_cast<std::size_t>(device_id + 1));
    }
    return &g_lut_states[static_cast<std::size_t>(device_id)];
}

bool alloc_and_copy_lut(
    const float* host_lut,
    float** dev_ptr,
    cudaStream_t stream) {
    if (host_lut == nullptr || dev_ptr == nullptr) return false;
    *dev_ptr = nullptr;

    cudaError_t err =
        cudaMalloc(reinterpret_cast<void**>(dev_ptr), 256 * sizeof(float));
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
    DeviceLutState* state = get_state_for_device(device_id);
    if (state == nullptr) return false;
    if (state->uploaded) return true;

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        return false;
    }

    const float* host_ieee_e4m3 = GetHostFp8LutV2(Fp8Format::IEEE_E4M3);
    const float* host_ieee_e5m2 = GetHostFp8LutV2(Fp8Format::IEEE_E5M2);
    const float* host_torch_e4m3fn = GetHostFp8LutV2(Fp8Format::TORCH_E4M3FN);
    if (host_ieee_e4m3 == nullptr ||
        host_ieee_e5m2 == nullptr ||
        host_torch_e4m3fn == nullptr) {
        return false;
    }

    float* lut_ieee_e4m3 = nullptr;
    float* lut_ieee_e5m2 = nullptr;
    float* lut_torch_e4m3fn = nullptr;

    if (!alloc_and_copy_lut(host_ieee_e4m3, &lut_ieee_e4m3, stream)) {
        goto cleanup;
    }
    if (!alloc_and_copy_lut(host_ieee_e5m2, &lut_ieee_e5m2, stream)) {
        goto cleanup;
    }
    if (!alloc_and_copy_lut(host_torch_e4m3fn, &lut_torch_e4m3fn, stream)) {
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
