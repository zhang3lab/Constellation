#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>

#include "expert_node_v2/backend/activation_codec_v2.h"
#include "expert_node_v2/backend/cpu/down_cpu_v2.h"
#include "expert_node_v2/backend/cpu/fused_up_gate_cpu_v2.h"
#include "expert_node_v2/backend/cpu/backend_cpu_v2.h"
#include "expert_node_v2/backend/dummy_expert_data_v2.h"
#include "expert_node_v2/backend/expert_reference_v2.h"
#include "expert_node_v2/backend/fp8_lut_v2.h"
#include "expert_node_v2/expert_format_v2.h"

bool RunDownCpuPredecodeBlockLocalV2(
    const MatrixBlockScaleViewV2& w_down,
    const float* h,
    float* y) {
    if (h == nullptr || y == nullptr) return false;

    const int rows = w_down.matrix.rows;
    const int cols = w_down.matrix.cols;
    if (rows <= 0 || cols <= 0) return false;

    const float* lut = GetHostFp8LutV2(w_down.matrix.fp8_format);
    if (lut == nullptr) return false;

    const auto weights = w_down.weight.data;
    const auto scales = w_down.scale.data;

    const int row_block = w_down.scale_meta.row_block;
    const int col_block = w_down.scale_meta.col_block;
    const int num_col_blocks = w_down.scale_meta.num_col_blocks;

    std::vector<float> decoded_block(static_cast<std::size_t>(col_block), 0.0f);

    for (int row = 0; row < rows; ++row) {
        const int rb = row / row_block;
        const std::size_t row_base =
            static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);

        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;

        for (int k0 = 0; k0 < cols; k0 += col_block) {
            const int k1 = std::min(k0 + col_block, cols);
            const int cb = k0 / col_block;

            const std::size_t s_idx =
                static_cast<std::size_t>(rb) *
                    static_cast<std::size_t>(num_col_blocks) +
                static_cast<std::size_t>(cb);

            const float scale = scales[s_idx];
            const int block_len = k1 - k0;

            for (int t = 0; t < block_len; ++t) {
                const std::size_t w_idx =
                    row_base + static_cast<std::size_t>(k0 + t);
                decoded_block[static_cast<std::size_t>(t)] =
                    lut[weights[w_idx]] * scale;
            }

            int t = 0;
            for (; t + 3 < block_len; t += 4) {
                const int k = k0 + t;

                const float w0 = decoded_block[static_cast<std::size_t>(t + 0)];
                const float w1 = decoded_block[static_cast<std::size_t>(t + 1)];
                const float w2 = decoded_block[static_cast<std::size_t>(t + 2)];
                const float w3 = decoded_block[static_cast<std::size_t>(t + 3)];

                sum0 += w0 * h[k + 0];
                sum1 += w1 * h[k + 1];
                sum2 += w2 * h[k + 2];
                sum3 += w3 * h[k + 3];
            }

            for (; t < block_len; ++t) {
                const int k = k0 + t;
                const float w = decoded_block[static_cast<std::size_t>(t)];
                switch (t & 3) {
                    case 0:
                        sum0 += w * h[k];
                        break;
                    case 1:
                        sum1 += w * h[k];
                        break;
                    case 2:
                        sum2 += w * h[k];
                        break;
                    default:
                        sum3 += w * h[k];
                        break;
                }
            }
        }

        y[row] = (sum0 + sum1) + (sum2 + sum3);
    }

    return true;
}

bool RunDownCpuTileDecodeLocalV2(
    const MatrixBlockScaleViewV2& w_down,
    const float* h,
    float* y) {
    if (h == nullptr || y == nullptr) return false;

    const int rows = w_down.matrix.rows;
    const int cols = w_down.matrix.cols;
    if (rows <= 0 || cols <= 0) return false;

    const float* lut = GetHostFp8LutV2(w_down.matrix.fp8_format);
    if (lut == nullptr) return false;

    const auto weights = w_down.weight.data;
    const auto scales = w_down.scale.data;

    const int row_block = w_down.scale_meta.row_block;
    const int col_block = w_down.scale_meta.col_block;
    const int num_col_blocks = w_down.scale_meta.num_col_blocks;

    constexpr int TILE = 16;
    float tmp[TILE];

    for (int row = 0; row < rows; ++row) {
        const int rb = row / row_block;
        const std::size_t row_base =
            static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);

        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;

        for (int k0 = 0; k0 < cols; k0 += col_block) {
            const int k1 = std::min(k0 + col_block, cols);
            const int cb = k0 / col_block;

            const std::size_t s_idx =
                static_cast<std::size_t>(rb) *
                    static_cast<std::size_t>(num_col_blocks) +
                static_cast<std::size_t>(cb);

            const float scale = scales[s_idx];

            for (int t0 = k0; t0 < k1; t0 += TILE) {
                const int t1 = std::min(t0 + TILE, k1);
                const int tile_len = t1 - t0;

                for (int t = 0; t < tile_len; ++t) {
                    const std::size_t w_idx =
                        row_base + static_cast<std::size_t>(t0 + t);
                    tmp[t] = lut[weights[w_idx]] * scale;
                }

                int t = 0;
                for (; t + 3 < tile_len; t += 4) {
                    const int k = t0 + t;

                    const float w0 = tmp[t + 0];
                    const float w1 = tmp[t + 1];
                    const float w2 = tmp[t + 2];
                    const float w3 = tmp[t + 3];

                    sum0 += w0 * h[k + 0];
                    sum1 += w1 * h[k + 1];
                    sum2 += w2 * h[k + 2];
                    sum3 += w3 * h[k + 3];
                }

                for (; t < tile_len; ++t) {
                    const int k = t0 + t;
                    const float w = tmp[t];
                    switch (t & 3) {
                        case 0:
                            sum0 += w * h[k];
                            break;
                        case 1:
                            sum1 += w * h[k];
                            break;
                        case 2:
                            sum2 += w * h[k];
                            break;
                        default:
                            sum3 += w * h[k];
                            break;
                    }
                }
            }
        }

        y[row] = (sum0 + sum1) + (sum2 + sum3);
    }

    return true;
}

bool RunFusedUpGateTile8LocalV2(
    const MatrixBlockScaleViewV2& w_up,
    const MatrixBlockScaleViewV2& w_gate,
    const void* x,
    common::ActivationDType input_dtype,
    float* h) {
    if (x == nullptr || h == nullptr) return false;

    const int rows = w_up.matrix.rows;
    const int cols = w_up.matrix.cols;
    if (rows <= 0 || cols <= 0) return false;

    if (w_gate.matrix.rows != rows || w_gate.matrix.cols != cols) {
        return false;
    }

    switch (input_dtype) {
        case common::ActivationDType::FP16:
        case common::ActivationDType::BF16:
            break;
        default:
            return false;
    }

    const float* lut_up = GetHostFp8LutV2(w_up.matrix.fp8_format);
    const float* lut_gate = GetHostFp8LutV2(w_gate.matrix.fp8_format);
    if (lut_up == nullptr || lut_gate == nullptr) return false;

    const auto up_weights = w_up.weight.data;
    const auto up_scales = w_up.scale.data;
    const auto gate_weights = w_gate.weight.data;
    const auto gate_scales = w_gate.scale.data;
    const auto* x_u16 = static_cast<const std::uint16_t*>(x);

    const int up_row_block = w_up.scale_meta.row_block;
    const int up_col_block = w_up.scale_meta.col_block;
    const int up_num_col_blocks = w_up.scale_meta.num_col_blocks;

    const int gate_row_block = w_gate.scale_meta.row_block;
    const int gate_col_block = w_gate.scale_meta.col_block;
    const int gate_num_col_blocks = w_gate.scale_meta.num_col_blocks;

    if (up_col_block != gate_col_block) {
        return false;
    }

    std::vector<float> x_f32(static_cast<std::size_t>(cols));
    for (int k = 0; k < cols; ++k) {
        x_f32[static_cast<std::size_t>(k)] =
            DecodeActivationToFloatV2(input_dtype, x_u16[k]);
    }

    constexpr int TILE = 8;
    float up_tmp[TILE];
    float gate_tmp[TILE];

    for (int row = 0; row < rows; ++row) {
        const int rb_up = row / up_row_block;
        const int rb_gate = row / gate_row_block;
        const std::size_t row_base =
            static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);

        float up_sum0 = 0.0f;
        float up_sum1 = 0.0f;
        float up_sum2 = 0.0f;
        float up_sum3 = 0.0f;

        float gate_sum0 = 0.0f;
        float gate_sum1 = 0.0f;
        float gate_sum2 = 0.0f;
        float gate_sum3 = 0.0f;

        for (int k0 = 0; k0 < cols; k0 += up_col_block) {
            const int k1 = std::min(k0 + up_col_block, cols);

            const int cb_up = k0 / up_col_block;
            const int cb_gate = k0 / gate_col_block;

            const std::size_t up_s_idx =
                static_cast<std::size_t>(rb_up) *
                    static_cast<std::size_t>(up_num_col_blocks) +
                static_cast<std::size_t>(cb_up);

            const std::size_t gate_s_idx =
                static_cast<std::size_t>(rb_gate) *
                    static_cast<std::size_t>(gate_num_col_blocks) +
                static_cast<std::size_t>(cb_gate);

            const float up_scale = up_scales[up_s_idx];
            const float gate_scale = gate_scales[gate_s_idx];

            for (int t0 = k0; t0 < k1; t0 += TILE) {
                const int t1 = std::min(t0 + TILE, k1);
                const int tile_len = t1 - t0;

                for (int t = 0; t < tile_len; ++t) {
                    const std::size_t w_idx =
                        row_base + static_cast<std::size_t>(t0 + t);
                    up_tmp[t] = lut_up[up_weights[w_idx]] * up_scale;
                    gate_tmp[t] = lut_gate[gate_weights[w_idx]] * gate_scale;
                }

                int t = 0;
                for (; t + 3 < tile_len; t += 4) {
                    const int k = t0 + t;

                    const float x0 = x_f32[static_cast<std::size_t>(k + 0)];
                    const float x1 = x_f32[static_cast<std::size_t>(k + 1)];
                    const float x2 = x_f32[static_cast<std::size_t>(k + 2)];
                    const float x3 = x_f32[static_cast<std::size_t>(k + 3)];

                    up_sum0 += up_tmp[t + 0] * x0;
                    up_sum1 += up_tmp[t + 1] * x1;
                    up_sum2 += up_tmp[t + 2] * x2;
                    up_sum3 += up_tmp[t + 3] * x3;

                    gate_sum0 += gate_tmp[t + 0] * x0;
                    gate_sum1 += gate_tmp[t + 1] * x1;
                    gate_sum2 += gate_tmp[t + 2] * x2;
                    gate_sum3 += gate_tmp[t + 3] * x3;
                }

                for (; t < tile_len; ++t) {
                    const int k = t0 + t;
                    const float x_val = x_f32[static_cast<std::size_t>(k)];

                    switch (t & 3) {
                        case 0:
                            up_sum0 += up_tmp[t] * x_val;
                            gate_sum0 += gate_tmp[t] * x_val;
                            break;
                        case 1:
                            up_sum1 += up_tmp[t] * x_val;
                            gate_sum1 += gate_tmp[t] * x_val;
                            break;
                        case 2:
                            up_sum2 += up_tmp[t] * x_val;
                            gate_sum2 += gate_tmp[t] * x_val;
                            break;
                        default:
                            up_sum3 += up_tmp[t] * x_val;
                            gate_sum3 += gate_tmp[t] * x_val;
                            break;
                    }
                }
            }
        }

        const float up_sum = (up_sum0 + up_sum1) + (up_sum2 + up_sum3);
        const float gate_sum = (gate_sum0 + gate_sum1) + (gate_sum2 + gate_sum3);

        const float silu_gate = gate_sum / (1.0f + std::exp(-gate_sum));
        h[row] = silu_gate * up_sum;
    }

    return true;
}

namespace {

struct Args {
    std::string dtype = "fp16";
    int iters = 20;
};

Args parse_args(int argc, char** argv) {
    Args args;

    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];

        if (s == "--dtype") {
            if (i + 1 >= argc) {
                std::printf("missing value for --dtype\n");
                std::exit(1);
            }
            args.dtype = argv[++i];
        } else if (s == "--iters") {
            if (i + 1 >= argc) {
                std::printf("missing value for --iters\n");
                std::exit(1);
            }
            args.iters = std::atoi(argv[++i]);
        } else {
            std::printf("unknown arg: %s\n", s.c_str());
            std::exit(1);
        }
    }

    if (args.dtype != "fp16" && args.dtype != "bf16") {
        std::printf("unsupported dtype: %s\n", args.dtype.c_str());
        std::exit(1);
    }
    if (args.iters <= 0) {
        std::printf("iters must be > 0\n");
        std::exit(1);
    }

    return args;
}

float percentile(std::vector<float> values, float q) {
    if (values.empty()) return 0.0f;
    if (q <= 0.0f) q = 0.0f;
    if (q >= 1.0f) q = 1.0f;

    std::sort(values.begin(), values.end());

    if (values.size() == 1) return values[0];

    const float pos = q * static_cast<float>(values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
    if (lo == hi) return values[lo];

    const float w = pos - static_cast<float>(lo);
    return values[lo] * (1.0f - w) + values[hi] * w;
}

void print_stats(const char* name, const std::vector<float>& ms_list) {
    float sum = 0.0f;
    for (float v : ms_list) sum += v;
    const float mean =
        ms_list.empty() ? 0.0f : sum / static_cast<float>(ms_list.size());
    const float p50 = percentile(ms_list, 0.50f);
    const float p90 = percentile(ms_list, 0.90f);
    const float p99 = percentile(ms_list, 0.99f);

    std::printf(
        "PROFILE,%s,mean_ms=%g,p50_ms=%g,p90_ms=%g,p99_ms=%g\n",
        name, mean, p50, p90, p99);
}

float ms_since(std::clock_t t0, std::clock_t t1) {
    return 1000.0f *
           static_cast<float>(t1 - t0) /
           static_cast<float>(CLOCKS_PER_SEC);
}

struct TestContext {
    common::ActivationDType act_dtype = common::ActivationDType::FP16;

    ExpertTensorBundleV2 bundle;
    ExpertWeightsViewV2 host_view;
    ExpertDeviceStorageV2 storage;
    ExpertWorkspaceConfigV2 cfg;
    ExpertWorkspaceCpuV2 ws;

    std::vector<float> x_float;
    std::vector<std::uint16_t> x_act;

    bool storage_ready = false;
    bool ws_ready = false;
};

void CleanupTestContext(TestContext* ctx) {
    if (ctx == nullptr) return;

    if (ctx->ws_ready) {
        FreeExpertWorkspaceCpuV2(&ctx->ws);
        ctx->ws_ready = false;
    }
    if (ctx->storage_ready) {
        FreeExpertWeightsCpuV2(&ctx->storage);
        ctx->storage_ready = false;
    }
}

bool InitTestContext(const Args& args, TestContext* ctx) {
    if (ctx == nullptr) return false;

    if (args.dtype == "fp16") {
        ctx->act_dtype = common::ActivationDType::FP16;
    } else if (args.dtype == "bf16") {
        ctx->act_dtype = common::ActivationDType::BF16;
    } else {
        std::printf("unsupported dtype: %s\n", args.dtype.c_str());
        return false;
    }

    FillDummyExpertBundleV2(&ctx->bundle);

    if (!BuildExpertWeightsViewV2(ctx->bundle, &ctx->host_view)) {
        std::printf("BuildExpertWeightsViewV2 failed\n");
        return false;
    }

    if (!UploadExpertCpuV2(0, ctx->bundle, &ctx->storage)) {
        std::printf("UploadExpertCpuV2 failed\n");
        return false;
    }
    ctx->storage_ready = true;

    ctx->cfg.hidden_dim = 7168;
    ctx->cfg.inter_dim = 2048;

    if (!InitExpertWorkspaceCpuV2(ctx->cfg, &ctx->ws)) {
        std::printf("InitExpertWorkspaceCpuV2 failed\n");
        return false;
    }
    ctx->ws_ready = true;

    FillDummyInputActivationV2(
        ctx->cfg.hidden_dim,
        ctx->act_dtype,
        &ctx->x_float,
        &ctx->x_act);

    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    TestContext ctx;
    if (!InitTestContext(args, &ctx)) {
        CleanupTestContext(&ctx);
        return 1;
    }

    std::vector<float> h_mid(static_cast<std::size_t>(ctx.cfg.inter_dim), 0.0f);
    std::vector<std::uint16_t> y_u16(static_cast<std::size_t>(ctx.cfg.hidden_dim), 0);
    std::vector<float> y_f32(static_cast<std::size_t>(ctx.cfg.hidden_dim), 0.0f);

    std::vector<float> up_gate_ms_list;
    std::vector<float> down_u16_ms_list;
    std::vector<float> down_predecode_f32_ms_list;

    up_gate_ms_list.reserve(static_cast<std::size_t>(args.iters));
    down_u16_ms_list.reserve(static_cast<std::size_t>(args.iters));
    down_predecode_f32_ms_list.reserve(static_cast<std::size_t>(args.iters));

    std::vector<float> down_tile16_f32_ms_list;
    down_tile16_f32_ms_list.reserve(static_cast<std::size_t>(args.iters));

    std::vector<float> up_gate_tile8_ms_list;
    up_gate_tile8_ms_list.reserve(static_cast<std::size_t>(args.iters));

    for (int i = 0; i < args.iters; ++i) {
        {
            const std::clock_t t0 = std::clock();
            if (!RunFusedUpGateCpuV2(
                    ctx.storage.view().w_up,
                    ctx.storage.view().w_gate,
                    ctx.x_act.data(),
                    ctx.act_dtype,
                    h_mid.data())) {
                std::printf("RunFusedUpGateCpuV2 failed at iter=%d\n", i);
                CleanupTestContext(&ctx);
                return 1;
            }
            const std::clock_t t1 = std::clock();
            up_gate_ms_list.push_back(ms_since(t0, t1));
        }

        {
            const std::clock_t t0 = std::clock();
            if (!RunDownCpuV2(
                    ctx.storage.view().w_down,
                    h_mid.data(),
                    y_u16.data(),
                    ctx.act_dtype)) {
                std::printf("RunDownCpuV2 failed at iter=%d\n", i);
                CleanupTestContext(&ctx);
                return 1;
            }
            const std::clock_t t1 = std::clock();
            down_u16_ms_list.push_back(ms_since(t0, t1));
        }

        {
            const std::clock_t t0 = std::clock();
            if (!RunDownCpuPredecodeBlockLocalV2(
                    ctx.storage.view().w_down,
                    h_mid.data(),
                    y_f32.data())) {
                std::printf("RunDownCpuPredecodeBlockLocalV2 failed at iter=%d\n", i);
                CleanupTestContext(&ctx);
                return 1;
            }
            const std::clock_t t1 = std::clock();
            down_predecode_f32_ms_list.push_back(ms_since(t0, t1));
        }

	{
    const std::clock_t t0 = std::clock();
    if (!RunDownCpuTileDecodeLocalV2(
            ctx.storage.view().w_down,
            h_mid.data(),
            y_f32.data())) {
        std::printf("RunDownCpuTileDecodeLocalV2 failed at iter=%d\n", i);
        CleanupTestContext(&ctx);
        return 1;
    }
    const std::clock_t t1 = std::clock();
    down_tile16_f32_ms_list.push_back(ms_since(t0, t1));
}

{
    const std::clock_t t0 = std::clock();
    if (!RunFusedUpGateTile8LocalV2(
            ctx.storage.view().w_up,
            ctx.storage.view().w_gate,
            ctx.x_act.data(),
            ctx.act_dtype,
            h_mid.data())) {
        std::printf("RunFusedUpGateTile8LocalV2 failed at iter=%d\n", i);
        CleanupTestContext(&ctx);
        return 1;
    }
    const std::clock_t t1 = std::clock();
    up_gate_tile8_ms_list.push_back(ms_since(t0, t1));
}
    }

    print_stats("up_gate", up_gate_ms_list);
    print_stats("up_gate_tile8", up_gate_tile8_ms_list);
    print_stats("down_u16_out", down_u16_ms_list);
    print_stats("down_predecode_f32", down_predecode_f32_ms_list);
    print_stats("down_tile16_f32", down_tile16_f32_ms_list);

    CleanupTestContext(&ctx);
    return 0;
}
