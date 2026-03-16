#include "expert.h"

#include <cassert>
#include <cmath>
#include <cstdio>

namespace {

float silu(float x) {
    return x / (1.0f + std::exp(-x));
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
    if (fp8_format == expert::FP8_E5M2) {
        return decode_fp8_e5m2_host(x);
    }
    return decode_fp8_e4m3_host(x);
}

bool nearly_equal(float a, float b, float atol = 1e-3f, float rtol = 1e-3f) {
    const float diff = std::fabs(a - b);
    const float tol = atol + rtol * std::fabs(b);
    return diff <= tol;
}

void test_single_element_e4m3() {
    constexpr int hidden_dim = 1;
    constexpr int inter_dim = 1;
    constexpr int num_tokens = 1;
    constexpr int group_size = 1;

    const uint8_t w_up_fp8   = 0x38;
    const uint8_t w_gate_fp8 = 0x40;
    const uint8_t w_down_fp8 = 0x34;

    const float s_up[1]   = {1.0f};
    const float s_gate[1] = {1.0f};
    const float s_down[1] = {1.0f};

    const uint8_t w_up_data[1]   = {w_up_fp8};
    const uint8_t w_gate_data[1] = {w_gate_fp8};
    const uint8_t w_down_data[1] = {w_down_fp8};

    expert::HostExpertWeights host_w{};
    host_w.hidden_dim = hidden_dim;
    host_w.inter_dim = inter_dim;

    host_w.w_up.data = w_up_data;
    host_w.w_up.scales = s_up;
    host_w.w_up.rows = inter_dim;
    host_w.w_up.cols = hidden_dim;
    host_w.w_up.group_size = group_size;
    host_w.w_up.fp8_format = expert::FP8_E4M3;

    host_w.w_gate.data = w_gate_data;
    host_w.w_gate.scales = s_gate;
    host_w.w_gate.rows = inter_dim;
    host_w.w_gate.cols = hidden_dim;
    host_w.w_gate.group_size = group_size;
    host_w.w_gate.fp8_format = expert::FP8_E4M3;

    host_w.w_down.data = w_down_data;
    host_w.w_down.scales = s_down;
    host_w.w_down.rows = hidden_dim;
    host_w.w_down.cols = inter_dim;
    host_w.w_down.group_size = group_size;
    host_w.w_down.fp8_format = expert::FP8_E4M3;

    assert(expert::init_expert_runtime(0));
    assert(expert::load_expert_weights(0, host_w));

    const expert::ExpertWeights* w = expert::get_expert_weights(0);
    assert(w != nullptr);

    half* d_input = nullptr;
    half* d_output = nullptr;
    float* d_fused = nullptr;

    assert(cudaSuccess == cudaMalloc(&d_input, num_tokens * hidden_dim * sizeof(half)));
    assert(cudaSuccess == cudaMalloc(&d_output, num_tokens * hidden_dim * sizeof(half)));
    assert(cudaSuccess == cudaMalloc(&d_fused, num_tokens * inter_dim * sizeof(float)));

    const float x = 1.5f;
    half h_input[1];
    h_input[0] = __float2half(x);

    assert(cudaSuccess == cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    assert(cudaSuccess == cudaStreamCreate(&stream));

    assert(expert::launch_expert_mlp(*w, d_input, d_output, d_fused, num_tokens, stream));
    assert(cudaSuccess == cudaStreamSynchronize(stream));

    half h_output[1];
    assert(cudaSuccess == cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost));

    const float w_up   = decode_fp8_host(w_up_fp8, expert::FP8_E4M3) * s_up[0];
    const float w_gate = decode_fp8_host(w_gate_fp8, expert::FP8_E4M3) * s_gate[0];
    const float w_down = decode_fp8_host(w_down_fp8, expert::FP8_E4M3) * s_down[0];

    const float up = x * w_up;
    const float gate = x * w_gate;
    const float fused = up * silu(gate);
    const float expected = fused * w_down;

    const float got = __half2float(h_output[0]);

    std::printf("E4M3 single-element test\n");
    std::printf("x=%f w_up=%f w_gate=%f w_down=%f\n", x, w_up, w_gate, w_down);
    std::printf("expected=%f got=%f\n", expected, got);

    assert(nearly_equal(got, expected, 1e-2f, 1e-2f));

    assert(cudaSuccess == cudaStreamDestroy(stream));
    assert(cudaSuccess == cudaFree(d_input));
    assert(cudaSuccess == cudaFree(d_output));
    assert(cudaSuccess == cudaFree(d_fused));

    expert::shutdown_expert_runtime();
}

void test_single_element_e5m2() {
    constexpr int hidden_dim = 1;
    constexpr int inter_dim = 1;
    constexpr int num_tokens = 1;
    constexpr int group_size = 1;

    const uint8_t w_up_fp8   = 0x3C;
    const uint8_t w_gate_fp8 = 0x40;
    const uint8_t w_down_fp8 = 0x38;

    const float s_up[1]   = {0.5f};
    const float s_gate[1] = {1.0f};
    const float s_down[1] = {2.0f};

    const uint8_t w_up_data[1]   = {w_up_fp8};
    const uint8_t w_gate_data[1] = {w_gate_fp8};
    const uint8_t w_down_data[1] = {w_down_fp8};

    expert::HostExpertWeights host_w{};
    host_w.hidden_dim = hidden_dim;
    host_w.inter_dim = inter_dim;

    host_w.w_up.data = w_up_data;
    host_w.w_up.scales = s_up;
    host_w.w_up.rows = inter_dim;
    host_w.w_up.cols = hidden_dim;
    host_w.w_up.group_size = group_size;
    host_w.w_up.fp8_format = expert::FP8_E5M2;

    host_w.w_gate.data = w_gate_data;
    host_w.w_gate.scales = s_gate;
    host_w.w_gate.rows = inter_dim;
    host_w.w_gate.cols = hidden_dim;
    host_w.w_gate.group_size = group_size;
    host_w.w_gate.fp8_format = expert::FP8_E5M2;

    host_w.w_down.data = w_down_data;
    host_w.w_down.scales = s_down;
    host_w.w_down.rows = hidden_dim;
    host_w.w_down.cols = inter_dim;
    host_w.w_down.group_size = group_size;
    host_w.w_down.fp8_format = expert::FP8_E5M2;

    assert(expert::init_expert_runtime(0));
    assert(expert::load_expert_weights(1, host_w));

    const expert::ExpertWeights* w = expert::get_expert_weights(1);
    assert(w != nullptr);

    half* d_input = nullptr;
    half* d_output = nullptr;
    float* d_fused = nullptr;

    assert(cudaSuccess == cudaMalloc(&d_input, num_tokens * hidden_dim * sizeof(half)));
    assert(cudaSuccess == cudaMalloc(&d_output, num_tokens * hidden_dim * sizeof(half)));
    assert(cudaSuccess == cudaMalloc(&d_fused, num_tokens * inter_dim * sizeof(float)));

    const float x = 0.75f;
    half h_input[1];
    h_input[0] = __float2half(x);

    assert(cudaSuccess == cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    assert(cudaSuccess == cudaStreamCreate(&stream));

    assert(expert::launch_expert_mlp(*w, d_input, d_output, d_fused, num_tokens, stream));
    assert(cudaSuccess == cudaStreamSynchronize(stream));

    half h_output[1];
    assert(cudaSuccess == cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost));

    const float w_up   = decode_fp8_host(w_up_fp8, expert::FP8_E5M2) * s_up[0];
    const float w_gate = decode_fp8_host(w_gate_fp8, expert::FP8_E5M2) * s_gate[0];
    const float w_down = decode_fp8_host(w_down_fp8, expert::FP8_E5M2) * s_down[0];

    const float up = x * w_up;
    const float gate = x * w_gate;
    const float fused = up * silu(gate);
    const float expected = fused * w_down;

    const float got = __half2float(h_output[0]);

    std::printf("E5M2 single-element test\n");
    std::printf("x=%f w_up=%f w_gate=%f w_down=%f\n", x, w_up, w_gate, w_down);
    std::printf("expected=%f got=%f\n", expected, got);

    assert(nearly_equal(got, expected, 1e-2f, 1e-2f));

    assert(cudaSuccess == cudaStreamDestroy(stream));
    assert(cudaSuccess == cudaFree(d_input));
    assert(cudaSuccess == cudaFree(d_output));
    assert(cudaSuccess == cudaFree(d_fused));

    expert::shutdown_expert_runtime();
}

void test_small_matrix_e4m3_grouped() {
    constexpr int hidden_dim = 4;
    constexpr int inter_dim = 4;
    constexpr int num_tokens = 2;
    constexpr int group_size = 2;

    const float s_up[8] = {
        1.0f, 0.5f,
        1.0f, 0.5f,
        1.0f, 0.5f,
        1.0f, 0.5f,
    };
    const float s_gate[8] = {
        0.5f, 1.0f,
        0.5f, 1.0f,
        0.5f, 1.0f,
        0.5f, 1.0f,
    };
    const float s_down[8] = {
        1.0f, 1.0f,
        0.5f, 0.5f,
        1.0f, 1.0f,
        0.5f, 0.5f,
    };

    const uint8_t w_up_data[16] = {
        0x38, 0x40, 0x34, 0x30,
        0x40, 0x38, 0x30, 0x34,
        0x34, 0x30, 0x38, 0x40,
        0x30, 0x34, 0x40, 0x38,
    };

    const uint8_t w_gate_data[16] = {
        0x30, 0x34, 0x38, 0x40,
        0x34, 0x30, 0x40, 0x38,
        0x38, 0x40, 0x30, 0x34,
        0x40, 0x38, 0x34, 0x30,
    };

    const uint8_t w_down_data[16] = {
        0x38, 0x34, 0x40, 0x30,
        0x34, 0x38, 0x30, 0x40,
        0x40, 0x30, 0x38, 0x34,
        0x30, 0x40, 0x34, 0x38,
    };

    expert::HostExpertWeights host_w{};
    host_w.hidden_dim = hidden_dim;
    host_w.inter_dim = inter_dim;

    host_w.w_up.data = w_up_data;
    host_w.w_up.scales = s_up;
    host_w.w_up.rows = inter_dim;
    host_w.w_up.cols = hidden_dim;
    host_w.w_up.group_size = group_size;
    host_w.w_up.fp8_format = expert::FP8_E4M3;

    host_w.w_gate.data = w_gate_data;
    host_w.w_gate.scales = s_gate;
    host_w.w_gate.rows = inter_dim;
    host_w.w_gate.cols = hidden_dim;
    host_w.w_gate.group_size = group_size;
    host_w.w_gate.fp8_format = expert::FP8_E4M3;

    host_w.w_down.data = w_down_data;
    host_w.w_down.scales = s_down;
    host_w.w_down.rows = hidden_dim;
    host_w.w_down.cols = inter_dim;
    host_w.w_down.group_size = group_size;
    host_w.w_down.fp8_format = expert::FP8_E4M3;

    assert(expert::init_expert_runtime(0));
    assert(expert::load_expert_weights(2, host_w));

    const expert::ExpertWeights* w = expert::get_expert_weights(2);
    assert(w != nullptr);

    half* d_input = nullptr;
    half* d_output = nullptr;
    float* d_fused = nullptr;

    assert(cudaSuccess == cudaMalloc(&d_input, num_tokens * hidden_dim * sizeof(half)));
    assert(cudaSuccess == cudaMalloc(&d_output, num_tokens * hidden_dim * sizeof(half)));
    assert(cudaSuccess == cudaMalloc(&d_fused, num_tokens * inter_dim * sizeof(float)));

    const float x_host[num_tokens * hidden_dim] = {
        1.0f,  0.5f, -1.0f,  2.0f,
        0.25f, -0.5f, 1.5f, -2.0f,
    };

    half h_input[num_tokens * hidden_dim];
    for (int i = 0; i < num_tokens * hidden_dim; ++i) {
        h_input[i] = __float2half(x_host[i]);
    }

    assert(cudaSuccess == cudaMemcpy(
        d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    assert(cudaSuccess == cudaStreamCreate(&stream));

    assert(expert::launch_expert_mlp(*w, d_input, d_output, d_fused, num_tokens, stream));
    assert(cudaSuccess == cudaStreamSynchronize(stream));

    half h_output[num_tokens * hidden_dim];
    assert(cudaSuccess == cudaMemcpy(
        h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost));

    float w_up_ref[inter_dim][hidden_dim];
    float w_gate_ref[inter_dim][hidden_dim];
    float w_down_ref[hidden_dim][inter_dim];

    auto load_grouped = [](const uint8_t* data,
                           const float* scales,
                           int rows,
                           int cols,
                           int group_size,
                           int row,
                           int col) -> float {
        (void)rows;
        const int idx = row * cols + col;
        const int groups_per_row = (cols + group_size - 1) / group_size;
        const int group = row * groups_per_row + (col / group_size);
        return decode_fp8_e4m3_host(data[idx]) * scales[group];
    };

    for (int r = 0; r < inter_dim; ++r) {
        for (int c = 0; c < hidden_dim; ++c) {
            w_up_ref[r][c] = load_grouped(w_up_data, s_up, inter_dim, hidden_dim, group_size, r, c);
            w_gate_ref[r][c] = load_grouped(w_gate_data, s_gate, inter_dim, hidden_dim, group_size, r, c);
        }
    }

    for (int r = 0; r < hidden_dim; ++r) {
        for (int c = 0; c < inter_dim; ++c) {
            w_down_ref[r][c] = load_grouped(w_down_data, s_down, hidden_dim, inter_dim, group_size, r, c);
        }
    }

    float expected[num_tokens][hidden_dim];

    for (int t = 0; t < num_tokens; ++t) {
        float up[inter_dim];
        float gate[inter_dim];
        float fused[inter_dim];

        for (int i = 0; i < inter_dim; ++i) {
            up[i] = 0.0f;
            gate[i] = 0.0f;
            for (int k = 0; k < hidden_dim; ++k) {
                const float x = x_host[t * hidden_dim + k];
                up[i] += x * w_up_ref[i][k];
                gate[i] += x * w_gate_ref[i][k];
            }
            fused[i] = up[i] * silu(gate[i]);
        }

        for (int o = 0; o < hidden_dim; ++o) {
            expected[t][o] = 0.0f;
            for (int k = 0; k < inter_dim; ++k) {
                expected[t][o] += fused[k] * w_down_ref[o][k];
            }
        }
    }

    std::printf("Small matrix grouped E4M3 test\n");
    for (int t = 0; t < num_tokens; ++t) {
        for (int o = 0; o < hidden_dim; ++o) {
            const float got = __half2float(h_output[t * hidden_dim + o]);
            const float exp = expected[t][o];
            std::printf("token=%d out=%d expected=%f got=%f\n", t, o, exp, got);
            assert(nearly_equal(got, exp, 2e-2f, 2e-2f));
        }
    }

    assert(cudaSuccess == cudaStreamDestroy(stream));
    assert(cudaSuccess == cudaFree(d_input));
    assert(cudaSuccess == cudaFree(d_output));
    assert(cudaSuccess == cudaFree(d_fused));

    expert::shutdown_expert_runtime();
}

}  // namespace

int main() {
    test_single_element_e4m3();
    test_single_element_e5m2();
    test_small_matrix_e4m3_grouped();
    std::printf("All tests passed.\n");
    return 0;
}
