// test_expert_tiny.cpp
#include "expert.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        std::abort();
    }
}

float decode_fp8_e4m3_host(uint8_t x) {
    const int sign = (x >> 7) & 0x1;
    const int exp  = (x >> 3) & 0xF;
    const int mant = x & 0x7;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        const float frac = static_cast<float>(mant) / 8.0f;
        const float val = std::ldexp(frac, -6);
        return sign ? -val : val;
    }

    if (exp == 0xF) {
        if (mant == 0) return sign ? -INFINITY : INFINITY;
        return NAN;
    }

    const float frac = 1.0f + static_cast<float>(mant) / 8.0f;
    const float val = std::ldexp(frac, exp - 7);
    return sign ? -val : val;
}

float decode_fp8_e5m2_host(uint8_t x) {
    const int sign = (x >> 7) & 0x1;
    const int exp  = (x >> 2) & 0x1F;
    const int mant = x & 0x3;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        const float frac = static_cast<float>(mant) / 4.0f;
        const float val = std::ldexp(frac, -14);
        return sign ? -val : val;
    }

    if (exp == 0x1F) {
        if (mant == 0) return sign ? -INFINITY : INFINITY;
        return NAN;
    }

    const float frac = 1.0f + static_cast<float>(mant) / 4.0f;
    const float val = std::ldexp(frac, exp - 15);
    return sign ? -val : val;
}

float decode_fp8_host(uint8_t x, int fp8_format) {
    return (fp8_format == expert::FP8_E5M2)
        ? decode_fp8_e5m2_host(x)
        : decode_fp8_e4m3_host(x);
}

float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

bool nearly_equal(float a, float b, float atol = 5e-2f, float rtol = 5e-2f) {
    const float diff = std::fabs(a - b);
    const float tol = atol + rtol * std::fabs(b);
    return diff <= tol;
}

void fill_host_weights(
    std::vector<half>& w,
    int rows,
    int cols,
    float scale = 1.0f) {
    w.resize(static_cast<size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const float v = scale * (0.02f + 0.001f * static_cast<float>((r * 11 + c * 7) % 31));
            w[static_cast<size_t>(r) * cols + c] = __float2half(v);
        }
    }
}

void fill_host_input(
    std::vector<half>& x,
    int num_tokens,
    int hidden_dim) {
    x.resize(static_cast<size_t>(num_tokens) * hidden_dim);
    for (int b = 0; b < num_tokens; ++b) {
        for (int i = 0; i < hidden_dim; ++i) {
            const float v = 0.05f + 0.002f * static_cast<float>((b * 13 + i * 5) % 19);
            x[static_cast<size_t>(b) * hidden_dim + i] = __float2half(v);
        }
    }
}

void unpack_packed_matrix_to_host(
    const expert::PackedTileMatrix& w,
    std::vector<float>& out) {
    const int rows = w.rows;
    const int cols = w.cols;
    out.assign(static_cast<size_t>(rows) * cols, 0.0f);

    const int num_k_tiles = (cols + expert::kPackedKTile - 1) / expert::kPackedKTile;

    std::vector<float> h_scales(expert::packed_tile_scale_elems(rows, cols));
    std::vector<uint8_t> h_weights(expert::packed_tile_weight_elems(rows, cols));

    check_cuda(cudaMemcpy(
        h_scales.data(),
        w.scales,
        h_scales.size() * sizeof(float),
        cudaMemcpyDeviceToHost),
        "cudaMemcpy scales");

    check_cuda(cudaMemcpy(
        h_weights.data(),
        w.weights,
        h_weights.size() * sizeof(uint8_t),
        cudaMemcpyDeviceToHost),
        "cudaMemcpy weights");

    for (int r = 0; r < rows; ++r) {
        const int out_tile = r / expert::kPackedNTile;
        const int n_in_tile = r % expert::kPackedNTile;
        for (int c = 0; c < cols; ++c) {
            const int k_tile_id = c / expert::kPackedKTile;
            const int k_in_tile = c % expert::kPackedKTile;
            const int tile_id = out_tile * num_k_tiles + k_tile_id;

            const float scale = h_scales[tile_id];
            const size_t base = static_cast<size_t>(tile_id) *
                                expert::kPackedNTile * expert::kPackedKTile;
            const uint8_t packed =
                h_weights[base + static_cast<size_t>(n_in_tile) * expert::kPackedKTile + k_in_tile];

            out[static_cast<size_t>(r) * cols + c] =
                decode_fp8_host(packed, w.fp8_format) * scale;
        }
    }
}

void matmul_rowmajor(
    const std::vector<float>& x,   // [B, K]
    const std::vector<float>& w,   // [N, K]
    std::vector<float>& y,         // [B, N]
    int B,
    int K,
    int N) {
    y.assign(static_cast<size_t>(B) * N, 0.0f);
    for (int b = 0; b < B; ++b) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += x[static_cast<size_t>(b) * K + k] *
                       w[static_cast<size_t>(n) * K + k];
            }
            y[static_cast<size_t>(b) * N + n] = acc;
        }
    }
}

void reference_expert(
    const std::vector<half>& h_input,
    const expert::PackedTileExpertWeights& packed,
    int num_tokens,
    std::vector<float>& out_ref) {
    std::vector<float> w_up, w_gate, w_down;
    unpack_packed_matrix_to_host(packed.w_up, w_up);
    unpack_packed_matrix_to_host(packed.w_gate, w_gate);
    unpack_packed_matrix_to_host(packed.w_down, w_down);

    std::vector<float> x(static_cast<size_t>(num_tokens) * packed.hidden_dim);
    for (size_t i = 0; i < x.size(); ++i) x[i] = __half2float(h_input[i]);

    std::vector<float> up, gate;
    matmul_rowmajor(x, w_up, up, num_tokens, packed.hidden_dim, packed.inter_dim);
    matmul_rowmajor(x, w_gate, gate, num_tokens, packed.hidden_dim, packed.inter_dim);

    std::vector<float> fused(static_cast<size_t>(num_tokens) * packed.inter_dim);
    for (int b = 0; b < num_tokens; ++b) {
        for (int i = 0; i < packed.inter_dim; ++i) {
            const size_t idx = static_cast<size_t>(b) * packed.inter_dim + i;
            fused[idx] = up[idx] * silu(gate[idx]);
        }
    }

    matmul_rowmajor(fused, w_down, out_ref, num_tokens, packed.inter_dim, packed.hidden_dim);
}

void run_case(int num_tokens, int fp8_format) {
    const int hidden_dim = 256;
    const int inter_dim = 256;

    std::vector<half> h_w_up, h_w_gate, h_w_down, h_input;
    fill_host_weights(h_w_up, inter_dim, hidden_dim, 1.0f);
    fill_host_weights(h_w_gate, inter_dim, hidden_dim, 0.8f);
    fill_host_weights(h_w_down, hidden_dim, inter_dim, 0.7f);
    fill_host_input(h_input, num_tokens, hidden_dim);

    half *d_w_up = nullptr, *d_w_gate = nullptr, *d_w_down = nullptr;
    half *d_input = nullptr, *d_output = nullptr;
    float* d_workspace = nullptr;

    check_cuda(cudaMalloc(&d_w_up, h_w_up.size() * sizeof(half)), "cudaMalloc d_w_up");
    check_cuda(cudaMalloc(&d_w_gate, h_w_gate.size() * sizeof(half)), "cudaMalloc d_w_gate");
    check_cuda(cudaMalloc(&d_w_down, h_w_down.size() * sizeof(half)), "cudaMalloc d_w_down");
    check_cuda(cudaMalloc(&d_input, h_input.size() * sizeof(half)), "cudaMalloc d_input");
    check_cuda(cudaMalloc(&d_output, static_cast<size_t>(num_tokens) * hidden_dim * sizeof(half)),
               "cudaMalloc d_output");

    const size_t workspace_bytes =
        expert::workspace_bytes_for_tiny(expert::kTinyBatchMaxTokens, hidden_dim, inter_dim);
    check_cuda(cudaMalloc(&d_workspace, workspace_bytes), "cudaMalloc d_workspace");

    check_cuda(cudaMemcpy(d_w_up, h_w_up.data(), h_w_up.size() * sizeof(half), cudaMemcpyHostToDevice),
               "cudaMemcpy d_w_up");
    check_cuda(cudaMemcpy(d_w_gate, h_w_gate.data(), h_w_gate.size() * sizeof(half), cudaMemcpyHostToDevice),
               "cudaMemcpy d_w_gate");
    check_cuda(cudaMemcpy(d_w_down, h_w_down.data(), h_w_down.size() * sizeof(half), cudaMemcpyHostToDevice),
               "cudaMemcpy d_w_down");
    check_cuda(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half), cudaMemcpyHostToDevice),
               "cudaMemcpy d_input");

    expert::PackedTileExpertWeights packed{};
    packed.hidden_dim = hidden_dim;
    packed.inter_dim = inter_dim;

    auto alloc_pack = [&](expert::PackedTileMatrix* m, int rows, int cols) {
        check_cuda(cudaMalloc(&m->scales, expert::packed_tile_scale_elems(rows, cols) * sizeof(float)),
                   "cudaMalloc scales");
        check_cuda(cudaMalloc(&m->weights, expert::packed_tile_weight_elems(rows, cols) * sizeof(uint8_t)),
                   "cudaMalloc weights");
        m->rows = rows;
        m->cols = cols;
        m->fp8_format = fp8_format;
    };

    alloc_pack(&packed.w_up, inter_dim, hidden_dim);
    alloc_pack(&packed.w_gate, inter_dim, hidden_dim);
    alloc_pack(&packed.w_down, hidden_dim, inter_dim);

    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    assert(expert::quantize_and_pack_weight_tile_major_cuda(
        d_w_up, packed.w_up.scales, packed.w_up.weights,
        inter_dim, hidden_dim, fp8_format, stream));
    assert(expert::quantize_and_pack_weight_tile_major_cuda(
        d_w_gate, packed.w_gate.scales, packed.w_gate.weights,
        inter_dim, hidden_dim, fp8_format, stream));
    assert(expert::quantize_and_pack_weight_tile_major_cuda(
        d_w_down, packed.w_down.scales, packed.w_down.weights,
        hidden_dim, inter_dim, fp8_format, stream));

    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize pack");

    assert(expert::launch_expert_mlp_tiny(
        packed, d_input, d_output, d_workspace, num_tokens, stream));

    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize compute");

    std::vector<half> h_output(static_cast<size_t>(num_tokens) * hidden_dim);
    check_cuda(cudaMemcpy(
        h_output.data(), d_output, h_output.size() * sizeof(half), cudaMemcpyDeviceToHost),
        "cudaMemcpy d_output");

    std::vector<float> out_ref;
    reference_expert(h_input, packed, num_tokens, out_ref);

    const char* fmt = (fp8_format == expert::FP8_E5M2) ? "E5M2" : "E4M3";
    std::printf("case: B=%d fmt=%s\n", num_tokens, fmt);

    for (int b = 0; b < num_tokens; ++b) {
        for (int i = 0; i < hidden_dim; ++i) {
            const size_t idx = static_cast<size_t>(b) * hidden_dim + i;
            const float got = __half2float(h_output[idx]);
            const float exp = out_ref[idx];
            if (!nearly_equal(got, exp)) {
                std::fprintf(stderr,
                             "mismatch at b=%d i=%d: got=%f exp=%f\n",
                             b, i, got, exp);
                std::abort();
            }
        }
    }

    std::printf("  passed\n");

    cudaFree(packed.w_up.scales);
    cudaFree(packed.w_up.weights);
    cudaFree(packed.w_gate.scales);
    cudaFree(packed.w_gate.weights);
    cudaFree(packed.w_down.scales);
    cudaFree(packed.w_down.weights);

    cudaFree(d_w_up);
    cudaFree(d_w_gate);
    cudaFree(d_w_down);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_workspace);
    cudaStreamDestroy(stream);
}

}  // namespace

int main() {
    run_case(1, expert::FP8_E4M3);
    run_case(2, expert::FP8_E4M3);
    run_case(4, expert::FP8_E4M3);

    run_case(1, expert::FP8_E5M2);
    run_case(2, expert::FP8_E5M2);
    run_case(4, expert::FP8_E5M2);

    std::printf("All tiny expert tests passed.\n");
    return 0;
}
