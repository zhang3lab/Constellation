#include <chrono>
#include <cstdio>
#include <vector>

#include "expert_node_v2/backend/cpu/backend_cpu_v2.h"
#include "expert_node_v2/backend/cpu/down_cpu_v2.h"
#include "expert_node_v2/backend/cpu/fused_up_gate_cpu_v2.h"
#include "expert_node_v2/backend/cpu/tests/cpu_benchmark_support_v2.h"
#include "expert_node_v2/backend/cpu_fp16_resident/backend_cpu_fp16_resident_v2.h"
#include "expert_node_v2/backend/cpu_fp16_resident/down_cpu_fp16_resident_v2.h"
#include "expert_node_v2/backend/cpu_fp16_resident/fused_up_gate_cpu_fp16_resident_v2.h"

namespace {

struct DualStorageContext {
    TestContext common;

    ExpertDeviceStorageV2 basic_storage;
    ExpertDeviceStorageV2 fp16_storage;

    bool basic_storage_ready = false;
    bool fp16_storage_ready = false;
};

void CleanupDualStorageContext(DualStorageContext* ctx) {
    if (ctx == nullptr) return;

    if (ctx->basic_storage_ready) {
        FreeExpertWeightsCpuV2(&ctx->basic_storage);
        ctx->basic_storage_ready = false;
    }
    if (ctx->fp16_storage_ready) {
        FreeExpertWeightsCpuFp16ResidentV2(&ctx->fp16_storage);
        ctx->fp16_storage_ready = false;
    }

    CleanupTestContext(&ctx->common);
}

bool InitDualStorageContext(const Args& args, DualStorageContext* ctx) {
    if (ctx == nullptr) return false;

    if (!InitTestContext(args, &ctx->common)) {
        CleanupDualStorageContext(ctx);
        return false;
    }

    if (!UploadExpertCpuV2(0, ctx->common.bundle, &ctx->basic_storage)) {
        std::printf("UploadExpertCpuV2 failed\n");
        CleanupDualStorageContext(ctx);
        return false;
    }
    ctx->basic_storage_ready = true;

    if (!UploadExpertCpuFp16ResidentV2(0, ctx->common.bundle, &ctx->fp16_storage)) {
        std::printf("UploadExpertCpuFp16ResidentV2 failed\n");
        CleanupDualStorageContext(ctx);
        return false;
    }
    ctx->fp16_storage_ready = true;

    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    DualStorageContext ctx;
    if (!InitDualStorageContext(args, &ctx)) {
        CleanupDualStorageContext(&ctx);
        return 1;
    }

    const ExpertWeightsViewV2 basic_view = ctx.basic_storage.view();
    const ExpertWeightsViewV2 fp16_view = ctx.fp16_storage.view();

    for (int omp_threads : args.thread_list) {
        std::vector<float> up_gate_cpu_basic_ms_list;
        std::vector<float> up_gate_cpu_fp16_resident_ms_list;
        std::vector<float> down_cpu_basic_ms_list;
        std::vector<float> down_cpu_fp16_resident_ms_list;

        up_gate_cpu_basic_ms_list.reserve(static_cast<std::size_t>(args.iters));
        up_gate_cpu_fp16_resident_ms_list.reserve(static_cast<std::size_t>(args.iters));
        down_cpu_basic_ms_list.reserve(static_cast<std::size_t>(args.iters));
        down_cpu_fp16_resident_ms_list.reserve(static_cast<std::size_t>(args.iters));

        std::vector<float> h_mid_basic(
            static_cast<std::size_t>(ctx.common.cfg.inter_dim), 0.0f);
        std::vector<float> h_mid_fp16(
            static_cast<std::size_t>(ctx.common.cfg.inter_dim), 0.0f);

        std::vector<std::uint8_t> y_basic_bytes(
            static_cast<std::size_t>(ctx.common.cfg.hidden_dim) * sizeof(std::uint16_t), 0);
        std::vector<std::uint8_t> y_fp16_bytes(
            static_cast<std::size_t>(ctx.common.cfg.hidden_dim) * sizeof(std::uint16_t), 0);

        for (int i = 0; i < args.iters; ++i) {
            {
                if (args.flush_cache) {
                    FlushCpuCachesOmpLocalV2();
                }

                const auto t0 = std::chrono::steady_clock::now();
                if (!RunFusedUpGateCpuV2(
                        basic_view.w_up,
                        basic_view.w_gate,
                        ctx.common.x_act.data(),
                        ctx.common.act_dtype,
                        h_mid_basic.data(),
                        omp_threads)) {
                    std::printf("RunFusedUpGateCpuV2 failed at iter=%d\n", i);
                    CleanupDualStorageContext(&ctx);
                    return 1;
                }
                const auto t1 = std::chrono::steady_clock::now();
                up_gate_cpu_basic_ms_list.push_back(
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
            }

            {
                if (args.flush_cache) {
                    FlushCpuCachesOmpLocalV2();
                }

                const auto t0 = std::chrono::steady_clock::now();
                if (!RunFusedUpGateCpuFp16ResidentV2(
                        fp16_view.w_up,
                        fp16_view.w_gate,
                        ctx.common.x_act.data(),
                        ctx.common.act_dtype,
                        h_mid_fp16.data(),
                        omp_threads)) {
                    std::printf("RunFusedUpGateCpuFp16ResidentV2 failed at iter=%d\n", i);
                    CleanupDualStorageContext(&ctx);
                    return 1;
                }
                const auto t1 = std::chrono::steady_clock::now();
                up_gate_cpu_fp16_resident_ms_list.push_back(
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
            }

            {
                if (args.flush_cache) {
                    FlushCpuCachesOmpLocalV2();
                }

                const auto t0 = std::chrono::steady_clock::now();
                if (!RunDownCpuV2(
                        basic_view.w_down,
                        h_mid_basic.data(),
                        y_basic_bytes.data(),
                        ctx.common.act_dtype,
                        omp_threads)) {
                    std::printf("RunDownCpuV2 failed at iter=%d\n", i);
                    CleanupDualStorageContext(&ctx);
                    return 1;
                }
                const auto t1 = std::chrono::steady_clock::now();
                down_cpu_basic_ms_list.push_back(
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
            }

            {
                if (args.flush_cache) {
                    FlushCpuCachesOmpLocalV2();
                }

                const auto t0 = std::chrono::steady_clock::now();
                if (!RunDownCpuFp16ResidentV2(
                        fp16_view.w_down,
                        h_mid_fp16.data(),
                        y_fp16_bytes.data(),
                        ctx.common.act_dtype,
                        omp_threads)) {
                    std::printf("RunDownCpuFp16ResidentV2 failed at iter=%d\n", i);
                    CleanupDualStorageContext(&ctx);
                    return 1;
                }
                const auto t1 = std::chrono::steady_clock::now();
                down_cpu_fp16_resident_ms_list.push_back(
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
        }

        print_stats_with_threads("up_gate_cpu_basic", omp_threads, up_gate_cpu_basic_ms_list);
        print_stats_with_threads(
            "up_gate_cpu_fp16_resident", omp_threads, up_gate_cpu_fp16_resident_ms_list);
        print_stats_with_threads("down_cpu_basic", omp_threads, down_cpu_basic_ms_list);
        print_stats_with_threads(
            "down_cpu_fp16_resident", omp_threads, down_cpu_fp16_resident_ms_list);
    }

    CleanupDualStorageContext(&ctx);
    return 0;
}
