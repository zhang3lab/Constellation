// expert.h
//
// Matched interface for the current clean B=4 decode / MLP path.
//
// Design point:
//   - row-major packed FP8 weights
//   - per-row per-k-chunk scales
//   - batched decode matvec as the core primitive
//   - MLP = up + gate + silu_mul + down + cast
//
// Current tuned path:
//   - batch tile sweet spot: 4
//   - rows_per_cta sweet spot: 8
//   - k_chunk sweet spot: 1024
//
// Notes:
//   - Storage format and execution shape are separated.
//   - Current kernels may require shape.k_chunk == W.k_chunk.
//   - All interfaces here are intentionally in the global namespace to match
//     the current .cu files.

#pragma once

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// FP8 format
// -----------------------------------------------------------------------------

enum class Fp8Format : int {
    IEEE_E4M3 = 0,
    IEEE_E5M2 = 1,
    TORCH_E4M3FN = 2,
};

inline int fp8_format_to_int(Fp8Format fmt) {
    return static_cast<int>(fmt);
}

inline float fp8_max_finite(Fp8Format fmt) {
    switch (fmt) {
        case Fp8Format::IEEE_E4M3:
            return 448.0f;
        case Fp8Format::IEEE_E5M2:
            return 57344.0f;
        case Fp8Format::TORCH_E4M3FN:
            return 448.0f;
        default:
            return 448.0f;
    }
}

inline const char* fp8_format_name(Fp8Format fmt) {
    switch (fmt) {
        case Fp8Format::IEEE_E4M3:    return "IEEE_E4M3";
        case Fp8Format::IEEE_E5M2:    return "IEEE_E5M2";
        case Fp8Format::TORCH_E4M3FN: return "TORCH_E4M3FN";
        default:                      return "UNKNOWN";
    }
}

// -----------------------------------------------------------------------------
// Utility
// -----------------------------------------------------------------------------

__host__ __device__ inline int ceil_div_int(int a, int b) {
    return (a + b - 1) / b;
}

// -----------------------------------------------------------------------------
// Execution shape
// -----------------------------------------------------------------------------

struct MlpShape {
    int num_tokens = 0;    // logical batch / decode tokens
    int hidden_dim = 0;    // input / output hidden size
    int inter_dim = 0;     // expert intermediate size
    int k_chunk = 0;       // reduction chunk processed per iteration
    int rows_per_cta = 0;  // logical output rows handled by one CTA
    Fp8Format fp8_format = Fp8Format::IEEE_E4M3;
};

// -----------------------------------------------------------------------------
// Centralized default sizes
// -----------------------------------------------------------------------------

struct DefaultConfig {
    static constexpr int hidden_dim = 7168;
    static constexpr int inter_dim = 2048;
    static constexpr int k_chunk = 1024;
    static constexpr int rows_per_cta = 8;
    static constexpr Fp8Format fp8_format = Fp8Format::IEEE_E4M3;
};

struct SmallTestConfig {
    static constexpr int hidden_dim = 269;
    static constexpr int inter_dim = 131;
    static constexpr int k_chunk = 256;
    static constexpr int rows_per_cta = 4;
    static constexpr Fp8Format fp8_format = Fp8Format::IEEE_E4M3;
};

// -----------------------------------------------------------------------------
// Packed row-major matrix
//
// Semantics:
//   y[rows] = W[rows, cols] * x[cols]
//
// Storage:
//   weights: row-major uint8 bytes, shape [rows, cols]
//   scales : row-major float scales, shape [rows, ceil_div(cols, k_chunk)]
//
// Decode rule:
//   decoded_weight = LUT[weights[r, c]] * scales[r, c / k_chunk]
// -----------------------------------------------------------------------------

struct PackedRowMajorMatrix {
    int rows = 0;
    int cols = 0;
    int k_chunk = 0;
    int num_k_chunks = 0;
    Fp8Format fp8_format = Fp8Format::IEEE_E4M3;

    const uint8_t* weights = nullptr; // [rows * cols]
    const float* scales = nullptr;    // [rows * num_k_chunks]
};

struct PackedRowMajorMatrixHost {
    int rows = 0;
    int cols = 0;
    Fp8Format fp8_format = Fp8Format::IEEE_E4M3;
    int k_chunk = 0;
    int num_k_chunks = 0;

    uint8_t* weights = nullptr;
    float* scales = nullptr;
};

size_t packed_weights_bytes(int rows, int cols);
size_t packed_scales_bytes(int rows, int cols, int k_chunk);

bool pack_row_major_fp8_from_float(
    const float* src,
    int rows,
    int cols,
    int k_chunk,
    Fp8Format fp8_format,
    PackedRowMajorMatrixHost* out);

bool pack_row_major_fp8_from_fp8_bytes(
    const uint8_t* src,
    int rows,
    int cols,
    int k_chunk,
    Fp8Format src_format,
    Fp8Format packed_format,
    PackedRowMajorMatrixHost* out);

void free_packed_row_major_matrix_host(PackedRowMajorMatrixHost* p);

// -----------------------------------------------------------------------------
// Device-side full MLP view
// -----------------------------------------------------------------------------

struct DeviceMlpView {
    PackedRowMajorMatrix w_up;
    PackedRowMajorMatrix w_gate;
    PackedRowMajorMatrix w_down;
};

// -----------------------------------------------------------------------------
// Workspace layout helpers
//
// Workspace buffers are float and laid out as:
//
//   up   : [num_tokens, inter_dim]
//   gate : [num_tokens, inter_dim]
//   fused: [num_tokens, inter_dim]
//   outf : [num_tokens, hidden_dim]
//
// All offsets are returned in bytes.
// -----------------------------------------------------------------------------

inline size_t workspace_up_offset_bytes(const MlpShape& /*shape*/) {
    return 0;
}

inline size_t workspace_gate_offset_bytes(const MlpShape& shape) {
    return (size_t)shape.num_tokens * shape.inter_dim * sizeof(float);
}

inline size_t workspace_fused_offset_bytes(const MlpShape& shape) {
    return workspace_gate_offset_bytes(shape) +
           (size_t)shape.num_tokens * shape.inter_dim * sizeof(float);
}

inline size_t workspace_outf_offset_bytes(const MlpShape& shape) {
    return workspace_fused_offset_bytes(shape) +
           (size_t)shape.num_tokens * shape.inter_dim * sizeof(float);
}

inline size_t workspace_num_bytes(const MlpShape& shape) {
    return workspace_outf_offset_bytes(shape) +
           (size_t)shape.num_tokens * shape.hidden_dim * sizeof(float);
}

// Packed uint8 weights are one byte each.
inline size_t packed_num_bytes(int rows, int cols) {
    return (size_t)rows * cols * sizeof(uint8_t);
}

// -----------------------------------------------------------------------------
// Public CUDA entry points implemented in .cu files
// -----------------------------------------------------------------------------

bool initialize(cudaStream_t stream);

bool launch_matvec_decode(const PackedRowMajorMatrix& W,
                          const __half* d_x,
                          __half* d_y,
                          const MlpShape& shape,
                          cudaStream_t stream);

bool launch_matvec_decode_from_float(const PackedRowMajorMatrix& W,
                                     const float* d_x,
                                     float* d_y,
                                     const MlpShape& shape,
                                     cudaStream_t stream);

bool launch_mlp(const DeviceMlpView& mlp,
                const __half* d_input,
                __half* d_output,
                float* d_workspace,
                const MlpShape& shape,
                cudaStream_t stream);

// -----------------------------------------------------------------------------
// Vector ops
// -----------------------------------------------------------------------------

namespace expert {

bool launch_silu_mul(const float* d_up,
                     const float* d_gate,
                     float* d_out,
                     int num_tokens,
                     int inter_dim,
                     cudaStream_t stream);

bool launch_cast_float_to_half(const float* d_in,
                               __half* d_out,
                               int num_tokens,
                               int dim,
                               cudaStream_t stream);

}; // namespace expert
