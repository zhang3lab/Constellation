#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace expert {

// -----------------------------
// Public constants
// -----------------------------

constexpr int kPackedNTile = 128;
constexpr int kPackedKTile = 64;

// v1 only targets tiny batches.
// This is a design target / public limit for the first implementation.
constexpr int kTinyBatchMaxTokens = 8;

// -----------------------------
// Public protocol / enums
// -----------------------------

enum DType : uint32_t {
    DTYPE_FP16 = 0,
};

enum FP8Format : int {
    FP8_E4M3 = 0,
    FP8_E5M2 = 1,
};

constexpr uint32_t kRequestMagic  = 0x45585054;  // "EXPT"
constexpr uint32_t kResponseMagic = 0x45585052;  // "EXPR"

// -----------------------------
// Socket protocol headers
// -----------------------------

struct RequestHeader {
    uint32_t magic = kRequestMagic;
    uint32_t model_expert_id = 0;
    uint32_t num_tokens = 0;
    uint32_t hidden_dim = 0;
    uint32_t dtype = DTYPE_FP16;
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

// -----------------------------
// Host-side source weights
// dense/control side provides FP16 row-major weights.
// -----------------------------

struct HostDenseMatrix {
    const half* data = nullptr;   // host pointer
    int rows = 0;
    int cols = 0;
};

struct HostFP16ExpertWeights {
    HostDenseMatrix w_up;    // [inter_dim, hidden_dim]
    HostDenseMatrix w_gate;  // [inter_dim, hidden_dim]
    HostDenseMatrix w_down;  // [hidden_dim, inter_dim]

    int hidden_dim = 0;
    int inter_dim = 0;
};

// -----------------------------
// Resident packed weights
// Internal kernel logic remains in expert.cu.
// These fields are intentionally minimal but not fully opaque, so that
// loader / runtime / benchmarks can allocate and pass device storage.
// -----------------------------

struct PackedTileMatrix {
    float* scales = nullptr;      // device pointer, [num_tiles]
    uint8_t* weights = nullptr;   // device pointer, [num_tiles * kPackedNTile * kPackedKTile]

    int rows = 0;
    int cols = 0;
    int fp8_format = FP8_E4M3;
};

struct PackedTileExpertWeights {
    PackedTileMatrix w_up;
    PackedTileMatrix w_gate;
    PackedTileMatrix w_down;

    int hidden_dim = 0;
    int inter_dim = 0;
};

// -----------------------------
// Runtime IO buffers
// -----------------------------

struct DeviceBuffers {
    half* d_input = nullptr;      // [max_tokens, hidden_dim]
    half* d_output = nullptr;     // [max_tokens, hidden_dim]

    // Opaque workspace used by launch_expert_mlp_tiny().
    // Exact logical layout is private to expert.cu.
    float* d_workspace = nullptr;

    int max_tokens = 0;
    int hidden_dim = 0;
    int inter_dim = 0;
};

struct HostBuffers {
    half* h_input = nullptr;      // pinned host memory
    half* h_output = nullptr;     // pinned host memory

    int max_tokens = 0;
    int hidden_dim = 0;
};

// -----------------------------
// Packed-weight sizing helpers
// -----------------------------

inline size_t packed_tile_num_out_tiles(int rows) {
    return static_cast<size_t>((rows + kPackedNTile - 1) / kPackedNTile);
}

inline size_t packed_tile_num_k_tiles(int cols) {
    return static_cast<size_t>((cols + kPackedKTile - 1) / kPackedKTile);
}

inline size_t packed_tile_num_tiles(int rows, int cols) {
    return packed_tile_num_out_tiles(rows) * packed_tile_num_k_tiles(cols);
}

inline size_t packed_tile_weight_elems(int rows, int cols) {
    return packed_tile_num_tiles(rows, cols) *
           static_cast<size_t>(kPackedNTile) *
           static_cast<size_t>(kPackedKTile);
}

inline size_t packed_tile_scale_elems(int rows, int cols) {
    return packed_tile_num_tiles(rows, cols);
}

// -----------------------------
// Tiny-batch workspace sizing
// Returned size is in bytes.
// The internal layout is private to expert.cu.
// -----------------------------

size_t workspace_bytes_for_tiny(
    int max_tokens,
    int hidden_dim,
    int inter_dim);

// -----------------------------
// Runtime lifecycle
// -----------------------------

bool init_expert_runtime(int device_id);
void shutdown_expert_runtime();

// -----------------------------
// Buffer management
// max_tokens is expected to be <= kTinyBatchMaxTokens for v1.
// -----------------------------

bool allocate_device_buffers(
    DeviceBuffers* buffers,
    int max_tokens,
    int hidden_dim,
    int inter_dim);

void free_device_buffers(DeviceBuffers* buffers);

bool allocate_host_buffers(
    HostBuffers* buffers,
    int max_tokens,
    int hidden_dim);

void free_host_buffers(HostBuffers* buffers);

// -----------------------------
// Weight loading / lookup
// host FP16 -> resident packed FP8 tile-major
//
// Host pointers are consumed during the call; caller may release the
// source buffers after load_expert_weights_fp16() returns.
// -----------------------------

bool load_expert_weights_fp16(
    int model_expert_id,
    const HostFP16ExpertWeights& host_weights,
    int fp8_format);

// Returned pointer is owned by the runtime and remains valid until the
// expert is replaced or runtime is shut down.
const PackedTileExpertWeights* get_packed_expert_weights(
    int model_expert_id);

// -----------------------------
// GPU-side pack / compute entrypoints
// Implementations live in expert.cu.
// -----------------------------

bool quantize_and_pack_weight_tile_major_cuda(
    const half* d_src,      // [rows, cols], row-major on device
    float* d_dst_scales,    // [num_tiles] on device
    uint8_t* d_dst_weights, // [num_tiles * kPackedNTile * kPackedKTile] on device
    int rows,
    int cols,
    int fp8_format,
    cudaStream_t stream);

// Tiny-batch expert execution path.
//
// Intended for very small batches (e.g. 1..8), where parallelism must
// come primarily from output/K dimensions rather than the batch dimension.
bool launch_expert_mlp_tiny(
    const PackedTileExpertWeights& weights,
    const half* d_input,      // [num_tokens, hidden_dim]
    half* d_output,           // [num_tokens, hidden_dim]
    float* d_workspace,       // opaque workspace
    int num_tokens,
    cudaStream_t stream);

// -----------------------------
// Server
// -----------------------------

bool serve_forever(uint32_t server_id, const char* host, int port);

}  // namespace expert
