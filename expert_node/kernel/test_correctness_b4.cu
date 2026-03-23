
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <limits>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "expert_node/kernel/expert.h"

// External entry points from your CUDA files.
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

size_t workspace_up_offset_bytes(const MlpShape& shape);
size_t workspace_gate_offset_bytes(const MlpShape& shape);
size_t workspace_fused_offset_bytes(const MlpShape& shape);
size_t workspace_outf_offset_bytes(const MlpShape& shape);
size_t workspace_num_bytes(const MlpShape& shape);
size_t packed_num_bytes(int rows, int cols);

// ---- small helpers ----

#define CUDA_CHECK(expr) do { \
    cudaError_t _err = (expr); \
    if (_err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        std::exit(1); \
    } \
} while (0)

static float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

static __half f32_to_half(float x) {
    return __float2half_rn(x);
}
static float half_to_f32(__half x) {
    return __half2float(x);
}

struct HostPackedMatrix {
    int rows = 0;
    int cols = 0;
    int k_chunk = 0;
    int num_k_chunks = 0;
    Fp8Format fmt = Fp8Format::E4M3;
    std::vector<uint8_t> weights;
    std::vector<float> scales;
};

static HostPackedMatrix pack_fp8_rowmajor_ref(const std::vector<float>& w,
                                              int rows, int cols, int k_chunk,
                                              Fp8Format fmt) {
    HostPackedMatrix out;
    out.rows = rows;
    out.cols = cols;
    out.k_chunk = k_chunk;
    out.num_k_chunks = ceil_div_int(cols, k_chunk);
    out.fmt = fmt;
    out.weights.resize((size_t)rows * cols);
    out.scales.resize((size_t)rows * out.num_k_chunks);

    for (int r = 0; r < rows; ++r) {
        for (int chunk = 0; chunk < out.num_k_chunks; ++chunk) {
            const int k0 = chunk * k_chunk;
            const int k1 = std::min(cols, k0 + k_chunk);
            float amax = 0.f;
            for (int c = k0; c < k1; ++c) {
                amax = std::max(amax, std::fabs(w[(size_t)r * cols + c]));
            }
            const float scale = (amax == 0.0f) ? 1.0f : (amax / fp8_max_finite(fmt));
            out.scales[(size_t)r * out.num_k_chunks + chunk] = scale;

            for (int c = k0; c < k1; ++c) {
                const float x = w[(size_t)r * cols + c];
                out.weights[(size_t)r * cols + c] = fp32_to_fp8(x, scale, fmt);
            }
        }
    }
    return out;
}

static PackedRowMajorMatrix upload_packed(const HostPackedMatrix& hp) {
    PackedRowMajorMatrix dev{};
    dev.rows = hp.rows;
    dev.cols = hp.cols;
    dev.k_chunk = hp.k_chunk;
    dev.num_k_chunks = hp.num_k_chunks;
    dev.fp8_format = hp.fmt;

    uint8_t* d_weights = nullptr;
    float* d_scales = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_weights), hp.weights.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_scales), hp.scales.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_weights, hp.weights.data(), hp.weights.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, hp.scales.data(), hp.scales.size() * sizeof(float), cudaMemcpyHostToDevice));

    dev.weights = d_weights;
    dev.scales = d_scales;
    return dev;
}

static void free_packed(PackedRowMajorMatrix& m) {
    if (m.weights) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(const_cast<uint8_t*>(m.weights))));
    if (m.scales) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(const_cast<float*>(m.scales))));
    m.weights = nullptr;
    m.scales = nullptr;
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

static float max_rel_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float denom = std::max(1e-6f, std::fabs(b[i]));
        m = std::max(m, std::fabs(a[i] - b[i]) / denom);
    }
    return m;
}

static void cpu_matvec_ref(const std::vector<float>& w, int rows, int cols,
                           const std::vector<float>& x, int B,
                           std::vector<float>& y) {
    y.assign((size_t)B * rows, 0.f);
    for (int b = 0; b < B; ++b) {
        for (int r = 0; r < rows; ++r) {
            float acc = 0.f;
            for (int c = 0; c < cols; ++c) {
                acc += w[(size_t)r * cols + c] * x[(size_t)b * cols + c];
            }
            y[(size_t)b * rows + r] = acc;
        }
    }
}

static void cpu_mlp_ref(const std::vector<float>& w_up,
                        const std::vector<float>& w_gate,
                        const std::vector<float>& w_down,
                        int hidden, int inter, int B,
                        const std::vector<float>& x,
                        std::vector<float>& y) {
    std::vector<float> up, gate;
    cpu_matvec_ref(w_up, inter, hidden, x, B, up);
    cpu_matvec_ref(w_gate, inter, hidden, x, B, gate);

    std::vector<float> fused((size_t)B * inter, 0.f);
    for (size_t i = 0; i < fused.size(); ++i) fused[i] = silu(up[i]) * gate[i];

    cpu_matvec_ref(w_down, hidden, inter, fused, B, y);
}

static std::vector<float> rand_uniform(size_t n, float lo, float hi, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

static bool run_case(int B, int hidden, int inter, int k_chunk, Fp8Format fmt) {
    std::printf("\n=== correctness batch=%d hidden=%d inter=%d k_chunk=%d fmt=%s ===\n",
                B, hidden, inter, k_chunk, (fmt == Fp8Format::E4M3 ? "E4M3" : "E5M2"));

    MlpShape shape{};
    shape.num_tokens = B;
    shape.hidden_dim = hidden;
    shape.inter_dim = inter;
    shape.k_chunk = k_chunk;
    shape.rows_per_cta = 8;
    shape.fp8_format = fmt;

    // Host weights/input
    const auto w_up_f = rand_uniform((size_t)inter * hidden, -1.0f, 1.0f, 1);
    const auto w_gate_f = rand_uniform((size_t)inter * hidden, -1.0f, 1.0f, 2);
    const auto w_down_f = rand_uniform((size_t)hidden * inter, -1.0f, 1.0f, 3);
    const auto x_f = rand_uniform((size_t)B * hidden, -1.0f, 1.0f, 4);

    std::vector<__half> x_h((size_t)B * hidden);
    for (size_t i = 0; i < x_h.size(); ++i) x_h[i] = f32_to_half(x_f[i]);

    // Pack/upload
    const auto hp_up = pack_fp8_rowmajor_ref(w_up_f, inter, hidden, k_chunk, fmt);
    const auto hp_gate = pack_fp8_rowmajor_ref(w_gate_f, inter, hidden, k_chunk, fmt);
    const auto hp_down = pack_fp8_rowmajor_ref(w_down_f, hidden, inter, k_chunk, fmt);

    auto d_up = upload_packed(hp_up);
    auto d_gate = upload_packed(hp_gate);
    auto d_down = upload_packed(hp_down);

    DeviceMlpView mlp{};
    mlp.w_up = d_up;
    mlp.w_gate = d_gate;
    mlp.w_down = d_down;

    // Device buffers
    __half* d_x_h = nullptr;
    float* d_x_f = nullptr;
    __half* d_y_h = nullptr;
    float* d_y_f = nullptr;
    float* d_workspace = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x_h, x_h.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_x_f, x_f.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_h, (size_t)B * std::max(hidden, inter) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_y_f, (size_t)B * std::max(hidden, inter) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_workspace, workspace_num_bytes(shape)));

    CUDA_CHECK(cudaMemcpy(d_x_h, x_h.data(), x_h.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_f, x_f.data(), x_f.size() * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ---- matvec half path: up ----
    bool ok = launch_matvec_decode(d_up, d_x_h, d_y_h, shape, stream);
    if (!ok) {
        std::fprintf(stderr, "launch_matvec_decode(half) failed\n");
        return false;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<__half> y_half((size_t)B * inter);
    CUDA_CHECK(cudaMemcpy(y_half.data(), d_y_h, y_half.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    std::vector<float> y_half_f(y_half.size());
    for (size_t i = 0; i < y_half.size(); ++i) y_half_f[i] = half_to_f32(y_half[i]);

    std::vector<float> y_ref;
    cpu_matvec_ref(w_up_f, inter, hidden, x_f, B, y_ref);

    const float mat_half_abs = max_abs_diff(y_half_f, y_ref);
    const float mat_half_rel = max_rel_diff(y_half_f, y_ref);
    std::printf("matvec half-input  : max_abs=%g max_rel=%g\n", mat_half_abs, mat_half_rel);

    // ---- matvec float path: down ----
    std::vector<float> fused_in = rand_uniform((size_t)B * inter, -1.0f, 1.0f, 5);
    CUDA_CHECK(cudaMemcpy(d_x_f, fused_in.data(), fused_in.size() * sizeof(float), cudaMemcpyHostToDevice));

    MlpShape down_shape = shape;
    down_shape.hidden_dim = inter;
    down_shape.inter_dim = hidden;

    ok = launch_matvec_decode_from_float(d_down, d_x_f, d_y_f, down_shape, stream);
    if (!ok) {
        std::fprintf(stderr, "launch_matvec_decode_from_float failed\n");
        return false;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> y_down((size_t)B * hidden);
    CUDA_CHECK(cudaMemcpy(y_down.data(), d_y_f, y_down.size() * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> y_down_ref;
    cpu_matvec_ref(w_down_f, hidden, inter, fused_in, B, y_down_ref);

    const float mat_float_abs = max_abs_diff(y_down, y_down_ref);
    const float mat_float_rel = max_rel_diff(y_down, y_down_ref);
    std::printf("matvec float-input : max_abs=%g max_rel=%g\n", mat_float_abs, mat_float_rel);

    // ---- full MLP ----
    CUDA_CHECK(cudaMemcpy(d_x_h, x_h.data(), x_h.size() * sizeof(__half), cudaMemcpyHostToDevice));

    ok = launch_mlp(mlp, d_x_h, d_y_h, d_workspace, shape, stream);
    if (!ok) {
        std::fprintf(stderr, "launch_mlp failed\n");
        return false;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<__half> y_mlp_h((size_t)B * hidden);
    CUDA_CHECK(cudaMemcpy(y_mlp_h.data(), d_y_h, y_mlp_h.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    std::vector<float> y_mlp(y_mlp_h.size());
    for (size_t i = 0; i < y_mlp_h.size(); ++i) y_mlp[i] = half_to_f32(y_mlp_h[i]);

    std::vector<float> y_mlp_ref;
    cpu_mlp_ref(w_up_f, w_gate_f, w_down_f, hidden, inter, B, x_f, y_mlp_ref);

    const float mlp_abs = max_abs_diff(y_mlp, y_mlp_ref);
    const float mlp_rel = max_rel_diff(y_mlp, y_mlp_ref);
    std::printf("full mlp           : max_abs=%g max_rel=%g\n", mlp_abs, mlp_rel);

    free_packed(d_up);
    free_packed(d_gate);
    free_packed(d_down);
    CUDA_CHECK(cudaFree(d_x_h));
    CUDA_CHECK(cudaFree(d_x_f));
    CUDA_CHECK(cudaFree(d_y_h));
    CUDA_CHECK(cudaFree(d_y_f));
    CUDA_CHECK(cudaFree(d_workspace));
    CUDA_CHECK(cudaStreamDestroy(stream));

    // FP8 quantization + half cast means this should not be tiny.
    const bool pass =
        (mat_half_abs < 1.5f && mat_half_rel < 0.25f) &&
        (mat_float_abs < 1.5f && mat_float_rel < 0.25f) &&
        (mlp_abs < 3.0f && mlp_rel < 0.35f);

    std::printf("result             : %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

int main() {
    // Use supported k_chunk values only.
    // Small odd shapes still exercise tails; packed k_chunk stays legal.
    bool ok = true;
    ok &= run_case(1, 269, 131, 256, Fp8Format::E4M3);
    ok &= run_case(2, 269, 131, 256, Fp8Format::E4M3);
    ok &= run_case(4, 269, 131, 256, Fp8Format::E4M3);

    ok &= run_case(1, 269, 131, 256, Fp8Format::E5M2);
    ok &= run_case(2, 269, 131, 256, Fp8Format::E5M2);
    ok &= run_case(4, 269, 131, 256, Fp8Format::E5M2);

    // One real-shape sanity point with the current tuned path.
    ok &= run_case(4, 7168, 2048, 1024, Fp8Format::E4M3);

    if (!ok) {
        std::fprintf(stderr, "\nTEST FAILED\n");
        return 1;
    }
    std::printf("\nALL TESTS PASSED\n");
    return 0;
}
