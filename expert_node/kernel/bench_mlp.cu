#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "expert.h"

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", msg, cudaGetErrorString(err));
        std::exit(1);
    }
}

static float percentile(std::vector<float> v, double q) {
    if (v.empty()) return 0.0f;
    std::sort(v.begin(), v.end());
    double pos = q * (v.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(pos));
    size_t hi = static_cast<size_t>(std::ceil(pos));
    double t = pos - lo;
    return static_cast<float>((1.0 - t) * v[lo] + t * v[hi]);
}

static std::vector<float> random_vec(size_t n, float scale = 0.5f) {
    static std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> x(n);
    for (size_t i = 0; i < n; ++i) x[i] = dist(rng);
    return x;
}

static std::vector<__half> float_to_half_host(const std::vector<float>& x) {
    std::vector<__half> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = __float2half(x[i]);
    return y;
}

static const char* fp8_format_name(Fp8Format fmt) {
    switch (fmt) {
        case Fp8Format::E4M3: return "E4M3";
        case Fp8Format::E5M2: return "E5M2";
        default: return "UNKNOWN";
    }
}

static PackedRowMajorMatrix make_packed_matrix_device(
    int rows, int cols, int k_chunk, Fp8Format fmt)
{
    std::vector<float> h_w = random_vec(static_cast<size_t>(rows) * cols);

    PackedRowMajorMatrixHost hpack{};
    if (!pack_row_major_fp8_from_float(h_w.data(), rows, cols, k_chunk, fmt, &hpack)) {
        std::fprintf(stderr, "pack_row_major_fp8_from_float failed\n");
        std::exit(1);
    }

    PackedRowMajorMatrix dW{};
    dW.rows = rows;
    dW.cols = cols;
    dW.k_chunk = hpack.k_chunk;
    dW.num_k_chunks = hpack.num_k_chunks;
    dW.fp8_format = hpack.fp8_format;

    uint8_t* d_weights = nullptr;
    float* d_scales = nullptr;

    const size_t w_bytes = packed_weights_bytes(rows, cols);
    const size_t s_bytes = packed_scales_bytes(rows, cols, k_chunk);

    check_cuda(cudaMalloc(&d_weights, w_bytes), "cudaMalloc d_weights");
    check_cuda(cudaMalloc(&d_scales, s_bytes), "cudaMalloc d_scales");

    check_cuda(
        cudaMemcpy(d_weights, hpack.weights, w_bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy d_weights");
    check_cuda(
        cudaMemcpy(d_scales, hpack.scales, s_bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy d_scales");

    dW.weights = d_weights;
    dW.scales = d_scales;

    free_packed_row_major_matrix_host(&hpack);
    return dW;
}

static DeviceMlpView make_device_mlp(
    int hidden_dim, int inter_dim, int k_chunk, Fp8Format fmt)
{
    DeviceMlpView mlp{};
    mlp.w_up   = make_packed_matrix_device(inter_dim, hidden_dim, k_chunk, fmt);
    mlp.w_gate = make_packed_matrix_device(inter_dim, hidden_dim, k_chunk, fmt);
    mlp.w_down = make_packed_matrix_device(hidden_dim, inter_dim, k_chunk, fmt);
    return mlp;
}

static void free_matrix(PackedRowMajorMatrix& W) {
    if (W.weights) {
        cudaFree(const_cast<uint8_t*>(W.weights));
        W.weights = nullptr;
    }
    if (W.scales) {
        cudaFree(const_cast<float*>(W.scales));
        W.scales = nullptr;
    }
}

static void free_mlp(DeviceMlpView& mlp) {
    free_matrix(mlp.w_up);
    free_matrix(mlp.w_gate);
    free_matrix(mlp.w_down);
}

static void run_one_case(
    int batch,
    int hidden_dim,
    int inter_dim,
    int k_chunk,
    Fp8Format fmt,
    int warmup,
    int iters)
{
    MlpShape shape{};
    shape.num_tokens = batch;
    shape.hidden_dim = hidden_dim;
    shape.inter_dim = inter_dim;
    shape.k_chunk = k_chunk;
    shape.rows_per_cta = (hidden_dim <= 512) ? 4 : 8;
    shape.fp8_format = fmt;

    DeviceMlpView mlp = make_device_mlp(hidden_dim, inter_dim, k_chunk, fmt);

    const size_t x_elems = static_cast<size_t>(batch) * hidden_dim;
    const size_t y_elems = static_cast<size_t>(batch) * hidden_dim;
    const size_t workspace_bytes = workspace_num_bytes(shape);

    std::vector<float> hx_f = random_vec(x_elems);
    std::vector<__half> hx = float_to_half_host(hx_f);

    __half* d_x = nullptr;
    __half* d_y = nullptr;
    void* d_workspace_void = nullptr;

    check_cuda(cudaMalloc(&d_x, x_elems * sizeof(__half)), "cudaMalloc d_x");
    check_cuda(cudaMalloc(&d_y, y_elems * sizeof(__half)), "cudaMalloc d_y");
    check_cuda(cudaMalloc(&d_workspace_void, workspace_bytes), "cudaMalloc d_workspace");

    float* d_workspace = reinterpret_cast<float*>(d_workspace_void);

    check_cuda(
        cudaMemcpy(d_x, hx.data(), x_elems * sizeof(__half), cudaMemcpyHostToDevice),
        "cudaMemcpy d_x");

    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    for (int i = 0; i < warmup; ++i) {
        bool ok = launch_mlp(mlp, d_x, d_y, d_workspace, shape, stream);
        if (!ok) {
            std::fprintf(stderr, "launch_mlp failed during warmup\n");
            std::exit(1);
        }
    }
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize warmup");

    std::vector<float> times_ms;
    times_ms.reserve(iters);

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    for (int i = 0; i < iters; ++i) {
        check_cuda(cudaEventRecord(start, stream), "cudaEventRecord start");

        bool ok = launch_mlp(mlp, d_x, d_y, d_workspace, shape, stream);
        if (!ok) {
            std::fprintf(stderr, "launch_mlp failed during benchmark\n");
            std::exit(1);
        }

        check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord stop");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

        float ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
        times_ms.push_back(ms);
    }

    float sum = 0.0f;
    for (float t : times_ms) sum += t;
    const float mean = sum / std::max(1, static_cast<int>(times_ms.size()));
    const float p50 = percentile(times_ms, 0.50);
    const float p90 = percentile(times_ms, 0.90);
    const float p99 = percentile(times_ms, 0.99);
    const double tok_per_s = (mean > 0.0f) ? (1000.0 * batch / mean) : 0.0;

    std::printf(
        "[MLP] batch=%d hidden=%d inter=%d k_chunk=%d fmt=%s "
        "mean=%.4f ms p50=%.4f ms p90=%.4f ms p99=%.4f ms throughput_tok_s=%.2f\n",
        batch, hidden_dim, inter_dim, k_chunk, fp8_format_name(fmt),
        mean, p50, p90, p99, tok_per_s);

    std::printf(
        "CSV,MLP,%d,%d,%d,%d,%s,%.4f,%.4f,%.4f,%.4f,%.2f\n",
        batch, hidden_dim, inter_dim, k_chunk, fp8_format_name(fmt),
        mean, p50, p90, p99, tok_per_s);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_workspace_void);
    free_mlp(mlp);
}

int main() {
    check_cuda(cudaSetDevice(0), "cudaSetDevice");

    const int warmup = 50;
    const int iters = 200;

    std::printf("CSV_HEADER,kind,batch,hidden,inter,k_chunk,fmt,mean_ms,p50_ms,p90_ms,p99_ms,throughput_tok_s\n");

    // 1) Scan k_chunk on the larger shape
    const std::vector<int> scan_batches = {1, 4, 16};
    const std::vector<int> k_chunks = {256, 512, 1024};

    for (int k_chunk : k_chunks) {
        for (int b : scan_batches) {
            run_one_case(b, 7168, 2048, k_chunk, Fp8Format::E4M3, warmup, iters);
        }
    }

    // 2) Small-shape comparison
    const std::vector<int> small_batches = {1, 2, 4, 8, 16, 32};

    for (int b : small_batches) {
        run_one_case(b, 269, 131, 256, Fp8Format::E4M3, warmup, iters);
    }
    for (int b : small_batches) {
        run_one_case(b, 269, 131, 256, Fp8Format::E5M2, warmup, iters);
    }

    // 3) Large-shape baseline for comparison
    const std::vector<int> large_batches = {1, 2, 4, 8, 16, 32};

    for (int b : large_batches) {
        run_one_case(b, 7168, 2048, 1024, Fp8Format::E4M3, warmup, iters);
    }
    for (int b : large_batches) {
        run_one_case(b, 7168, 2048, 1024, Fp8Format::E5M2, warmup, iters);
    }

    return 0;
}
