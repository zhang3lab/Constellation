#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "expert_node_v2/backend/activation_codec_v2.h"
#include "expert_node_v2/backend/cpu/backend_cpu_v2.h"
#include "expert_node_v2/backend/cpu/tests/cpu_benchmark_support_v2.h"
#include "expert_node_v2/backend/expert_reference_v2.h"

namespace {

bool RunCorrectness(TestContext* ctx) {
    if (ctx == nullptr) return false;

    if (!RunExpertCpuV2(
            ctx->storage.view(),
            &ctx->ws,
            ctx->x_act.data(),
            ctx->act_dtype,
            ctx->y_cpu_bytes.data(),
            ctx->act_dtype)) {
        std::printf("RunExpertCpuV2 failed\n");
        return false;
    }

    std::vector<std::uint8_t> y_ref_bytes;
    std::vector<float> h_ref_debug;
    if (!RunExpertReferenceV2(
            ctx->host_view,
            ctx->x_act.data(),
            ctx->act_dtype,
            ctx->act_dtype,
            &y_ref_bytes,
            &h_ref_debug)) {
        std::printf("RunExpertReferenceV2 failed\n");
        return false;
    }

    const auto* y_ref_u16 =
        reinterpret_cast<const std::uint16_t*>(y_ref_bytes.data());
    const auto* y_cpu_u16 =
        reinterpret_cast<const std::uint16_t*>(ctx->y_cpu_bytes.data());

    std::vector<float> y_ref(ctx->cfg.hidden_dim);
    std::vector<float> y_cpu(ctx->cfg.hidden_dim);
    for (int i = 0; i < ctx->cfg.hidden_dim; ++i) {
        y_ref[i] = DecodeActivationToFloatV2(ctx->act_dtype, y_ref_u16[i]);
        y_cpu[i] = DecodeActivationToFloatV2(ctx->act_dtype, y_cpu_u16[i]);
    }

    float max_abs = 0.0f;
    float sum_abs = 0.0f;
    float dot = 0.0f;
    float norm_ref = 0.0f;
    float norm_cpu = 0.0f;

    for (int i = 0; i < ctx->cfg.hidden_dim; ++i) {
        const float ref_v = y_ref[i];
        const float cpu_v = y_cpu[i];
        const float abs_err = std::fabs(ref_v - cpu_v);

        if (abs_err > max_abs) max_abs = abs_err;
        sum_abs += abs_err;
        dot += ref_v * cpu_v;
        norm_ref += ref_v * ref_v;
        norm_cpu += cpu_v * cpu_v;
    }

    const float mean_abs = sum_abs / static_cast<float>(ctx->cfg.hidden_dim);
    const float cos =
        (norm_ref > 0.0f && norm_cpu > 0.0f)
            ? (dot / (std::sqrt(norm_ref) * std::sqrt(norm_cpu)))
            : 0.0f;

    std::printf(
        "correctness: max_abs=%g mean_abs=%g cos=%g\n",
        max_abs, mean_abs, cos);

    for (int i = 0; i < 8; ++i) {
        std::printf("ref[%d]=%g cpu[%d]=%g\n",
                    i, y_ref[i], i, y_cpu[i]);
    }

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

    const bool ok = RunCorrectness(&ctx);

    CleanupTestContext(&ctx);
    return ok ? 0 : 1;
}
