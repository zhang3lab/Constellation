// bench_expert_tiny.cpp
#include "expert.h"

#include <cassert>
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

void fill_src_weights(
    half* h_src,
    int rows,
    int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const float v =
                0.05f + 0.001f * static_cast<float>((r * 17 + c * 13) % 97);
            h_src[static_cast<size_t>(r) * cols + c] = __float2half(v);
        }
    }
}

void fill_input(
    half* h_input,
    int num_tokens,
    int hidden_dim) {
    for (int b = 0; b < num_tokens; ++b) {
        for (int i = 0; i < hidden_dim; ++i) {
            const float v =
                0.1f + 0.001f * static_cast<float>((b * 19 + i * 7) % 31);
            h_input[static_cast<size_t>(b) * hidden_dim + i] = __float2half(v);
        }
    }
}

void build_packed_matrix(
    expert::PackedTileMatrix* out,
    const half* d_src,
    int rows,
    int cols,
    int fp8_format,
    cudaStream_t stream) {
    const size_t scale_elems =
        expert::packed_tile_scale_elems(rows, cols);
    const size_t weight_elems =
        expert::packed_tile_weight_elems(rows, cols);

    check_cuda(cudaMalloc(&out->scales, scale_elems * sizeof(float)),
               "cudaMalloc packed scales");
    check_cuda(cudaMalloc(&out->weights, weight_elems * sizeof(uint8_t)),
               "cudaMalloc packed weights");

    out->rows = rows;
    out->cols = cols;
    out->fp8_format = fp8_format;

    const bool ok = expert::quantize_and_pack_weight_tile_major_cuda(
        d_src,
        out->scales,
        out->weights,
        rows,
        cols,
        fp8_format,
        stream);
    assert(ok);
}

void free_packed_matrix(expert::PackedTileMatrix* w) {
    if (w->scales) cudaFree(w->scales);
    if (w->weights) cudaFree(w->weights);
    w->scales = nullptr;
    w->weights = nullptr;
}

float benchmark_case(
    int hidden_dim,
    int inter_dim,
    int num_tokens,
    int fp8_format,
    int warmup,
    int iters) {
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    half *d_w_up_src = nullptr, *d_w_gate_src = nullptr, *d_w_down_src = nullptr;
    check_cuda(cudaMalloc(&d_w_up_src,
                          static_cast<size_t>(inter_dim) * hidden_dim * sizeof(half)),
               "cudaMalloc d_w_up_src");
    check_cuda(cudaMalloc(&d_w_gate_src,
                          static_cast<size_t>(inter_dim) * hidden_dim * sizeof(half)),
               "cudaMalloc d_w_gate_src");
    check_cuda(cudaMalloc(&d_w_down_src,
                          static_cast<size_t>(hidden_dim) * inter_dim * sizeof(half)),
               "cudaMalloc d_w_down_src");

    std::vector<half> h_w_up(static_cast<size_t>(inter_dim) * hidden_dim);
    std::vector<half> h_w_gate(static_cast<size_t>(inter_dim) * hidden_dim);
    std::vector<half> h_w_down(static_cast<size_t>(hidden_dim) * inter_dim);

    fill_src_weights(h_w_up.data(), inter_dim, hidden_dim);
    fill_src_weights(h_w_gate.data(), inter_dim, hidden_dim);
    fill_src_weights(h_w_down.data(), hidden_dim, inter_dim);

    check_cuda(cudaMemcpyAsync(
        d_w_up_src, h_w_up.data(), h_w_up.size() * sizeof(half),
        cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync d_w_up_src");
    check_cuda(cudaMemcpyAsync(
        d_w_gate_src, h_w_gate.data(), h_w_gate.size() * sizeof(half),
        cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync d_w_gate_src");
    check_cuda(cudaMemcpyAsync(
        d_w_down_src, h_w_down.data(), h_w_down.size() * sizeof(half),
        cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync d_w_down_src");

    expert::PackedTileExpertWeights w{};
    w.hidden_dim = hidden_dim;
    w.inter_dim = inter_dim;

    build_packed_matrix(&w.w_up, d_w_up_src, inter_dim, hidden_dim, fp8_format, stream);
    build_packed_matrix(&w.w_gate, d_w_gate_src, inter_dim, hidden_dim, fp8_format, stream);
    build_packed_matrix(&w.w_down, d_w_down_src, hidden_dim, inter_dim, fp8_format, stream);

    half* d_input = nullptr;
    half* d_output = nullptr;
    float* d_workspace = nullptr;

    check_cuda(cudaMalloc(&d_input, static_cast<size_t>(num_tokens) * hidden_dim * sizeof(half)),
               "cudaMalloc d_input");
    check_cuda(cudaMalloc(&d_output, static_cast<size_t>(num_tokens) * hidden_dim * sizeof(half)),
               "cudaMalloc d_output");
    check_cuda(cudaMalloc(&d_workspace,
                          expert::workspace_bytes_for_tiny(expert::kTinyBatchMaxTokens,
                                                           hidden_dim,
                                                           inter_dim)),
               "cudaMalloc d_workspace");

    std::vector<half> h_input(static_cast<size_t>(num_tokens) * hidden_dim);
    fill_input(h_input.data(), num_tokens, hidden_dim);

    check_cuda(cudaMemcpyAsync(
        d_input, h_input.data(), h_input.size() * sizeof(half),
        cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync d_input");

    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize setup");

    cudaEvent_t start = nullptr, stop = nullptr;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    for (int i = 0; i < warmup; ++i) {
        assert(expert::launch_expert_mlp_tiny(
            w, d_input, d_output, d_workspace, num_tokens, stream));
    }
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize warmup");

    check_cuda(cudaEventRecord(start, stream), "cudaEventRecord start");
    for (int i = 0; i < iters; ++i) {
        assert(expert::launch_expert_mlp_tiny(
            w, d_input, d_output, d_workspace, num_tokens, stream));
    }
    check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    float total_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&total_ms, start, stop), "cudaEventElapsedTime");

    check_cuda(cudaEventDestroy(start), "cudaEventDestroy start");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy stop");

    free_packed_matrix(&w.w_up);
    free_packed_matrix(&w.w_gate);
    free_packed_matrix(&w.w_down);

    check_cuda(cudaFree(d_w_up_src), "cudaFree d_w_up_src");
    check_cuda(cudaFree(d_w_gate_src), "cudaFree d_w_gate_src");
    check_cuda(cudaFree(d_w_down_src), "cudaFree d_w_down_src");
    check_cuda(cudaFree(d_input), "cudaFree d_input");
    check_cuda(cudaFree(d_output), "cudaFree d_output");
    check_cuda(cudaFree(d_workspace), "cudaFree d_workspace");
    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");

    return total_ms / static_cast<float>(iters);
}

void report_case(
    int hidden_dim,
    int inter_dim,
    int num_tokens,
    int fp8_format,
    float avg_ms) {
    const double scale_overhead =
        1.0 + 4.0 / (expert::kPackedNTile * expert::kPackedKTile);

    const double bytes_per_token =
        3.0 * static_cast<double>(hidden_dim) *
        static_cast<double>(inter_dim) * scale_overhead;
    const double total_bw_gbps =
        (bytes_per_token * static_cast<double>(num_tokens)) / (avg_ms * 1e-3) / 1e9;
    const double tok_per_s =
        static_cast<double>(num_tokens) / (avg_ms * 1e-3);
    const double pct_of_peak = 100.0 * total_bw_gbps / 936.0;  // RTX 3090 nominal BW
    const char* fmt = (fp8_format == expert::FP8_E5M2) ? "E5M2" : "E4M3";

    std::printf(
        "fmt=%s hidden=%d inter=%d tokens=%d avg_ms=%.6f tok/s=%.2f implied_BW=%.2f GB/s (%.1f%% of 936 GB/s)\n",
        fmt, hidden_dim, inter_dim, num_tokens, avg_ms, tok_per_s, total_bw_gbps, pct_of_peak);
}

}  // namespace

int main() {
    constexpr int hidden_dim = 7168;
    constexpr int inter_dim = 2048;
    constexpr int warmup = 20;
    constexpr int iters = 100;
    constexpr int fp8_format = expert::FP8_E4M3;

    const int token_cases[] = {1, 2, 4, 8};

    for (int num_tokens : token_cases) {
        const float avg_ms = benchmark_case(
            hidden_dim,
            inter_dim,
            num_tokens,
            fp8_format,
            warmup,
            iters);
        report_case(hidden_dim, inter_dim, num_tokens, fp8_format, avg_ms);
    }

    return 0;
}
