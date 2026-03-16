#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace expert {

constexpr uint32_t kRequestMagic  = 0x45585052;  // "EXPR"
constexpr uint32_t kResponseMagic = 0x45585053;  // "EXPS"

enum DType : uint32_t {
    DTYPE_FP16 = 0,
};

#pragma pack(push, 1)
struct RequestHeader {
    uint32_t magic = kRequestMagic;
    uint32_t model_expert_id = 0;
    uint32_t num_tokens = 0;
    uint32_t hidden_dim = 0;
    uint32_t dtype = DTYPE_FP16;
    uint32_t reserved = 0;
    uint64_t request_id = 0;
};

struct ResponseHeader {
    uint32_t magic = kResponseMagic;
    uint32_t model_expert_id = 0;
    uint32_t num_tokens = 0;
    uint32_t hidden_dim = 0;
    uint32_t dtype = DTYPE_FP16;
    uint32_t status = 0;
    uint64_t request_id = 0;
};
#pragma pack(pop)

static_assert(sizeof(RequestHeader) == 32, "Unexpected RequestHeader size");
static_assert(sizeof(ResponseHeader) == 32, "Unexpected ResponseHeader size");

enum FP8Format : int {
    FP8_E4M3 = 0,
    FP8_E5M2 = 1,
};

struct QuantMatrix {
    uint8_t* data = nullptr;
    float* scales = nullptr;

    int rows = 0;
    int cols = 0;
    int group_size = 64;
    int fp8_format = FP8_E4M3;
};

struct ExpertWeights {
    // Logical shapes:
    //   w_up   : [inter_dim, hidden_dim]
    //   w_gate : [inter_dim, hidden_dim]
    //   w_down : [hidden_dim, inter_dim]
    QuantMatrix w_up;
    QuantMatrix w_gate;
    QuantMatrix w_down;

    int hidden_dim = 0;
    int inter_dim = 0;
};

struct HostQuantMatrix {
    const uint8_t* data = nullptr;
    const float* scales = nullptr;

    int rows = 0;
    int cols = 0;
    int group_size = 64;
    int fp8_format = FP8_E4M3;
};

struct HostExpertWeights {
    HostQuantMatrix w_up;
    HostQuantMatrix w_gate;
    HostQuantMatrix w_down;

    int hidden_dim = 0;
    int inter_dim = 0;
};

struct DeviceBuffers {
    half* d_input = nullptr;
    half* d_output = nullptr;
    float* d_fused = nullptr;

    int max_tokens = 0;
    int hidden_dim = 0;
    int inter_dim = 0;
};

bool init_expert_runtime(int device_id);

bool allocate_device_buffers(
    DeviceBuffers* buffers,
    int max_tokens,
    int hidden_dim,
    int inter_dim);

void free_device_buffers(DeviceBuffers* buffers);

bool load_expert_weights(
    int model_expert_id,
    const HostExpertWeights& host_weights);

const ExpertWeights* get_expert_weights(int model_expert_id);

bool launch_expert_mlp(
    const ExpertWeights& weights,
    const half* d_input,
    half* d_output,
    float* d_fused_workspace,
    int num_tokens,
    cudaStream_t stream);

bool serve_forever(uint32_t server_id, const char* host, int port);

void shutdown_expert_runtime();

struct HostBuffers {
    half* h_input = nullptr;
    half* h_output = nullptr;

    int max_tokens = 0;
    int hidden_dim = 0;
};

bool allocate_host_buffers(
    HostBuffers* buffers,
    int max_tokens,
    int hidden_dim);

void free_host_buffers(HostBuffers* buffers);

}  // namespace expert
