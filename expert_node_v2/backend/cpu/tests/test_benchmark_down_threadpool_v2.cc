#include <chrono>
#include <cstdio>
#include <vector>

#include "expert_node_v2/backend/cpu/backend_cpu_v2.h"

// 下面这些按你工程实际 include 路径调整
// 如果它们不是头文件里定义的，而是在别的 test.cc 里，建议先抽到公共头里，
// 或者暂时直接把定义拷过来。
// 需要包含：
// - Args
// - parse_args(...)
// - percentile(...)
// - TestContext
// - InitTestContext(...)
// - CleanupTestContext(...)
// - Fp16ResidentMatrixLocalV2
// - BuildDownFp16ResidentLocalV2(...)
// - FixedRangeThreadPoolV2
// - RunDownCpuFp16ResidentF16cAvx2ThreadPoolV2(...)
// - FlushCpuCachesOmpLocalV2(...)

bool RunBenchmarkDownThreadPool(const Args& args, TestContext* ctx) {
    if (ctx == nullptr) return false;

    Fp16ResidentMatrixLocalV2 w_down_fp16;
    if (!BuildDownFp16ResidentLocalV2(ctx->storage.view().w_down, &w_down_fp16)) {
        std::printf("BuildDownFp16ResidentLocalV2 failed\n");
        return false;
    }

    FixedRangeThreadPoolV2 pool;
    if (!pool.init(args.threads)) {
        std::printf("FixedRangeThreadPoolV2 init failed\n");
        return false;
    }

    std::vector<float> h_mid(static_cast<std::size_t>(ctx->cfg.inter_dim), 0.0f);
    std::vector<float> y_f32(static_cast<std::size_t>(ctx->cfg.hidden_dim), 0.0f);

    for (int i = 0; i < args.warmup; ++i) {
        if (args.flush_cache) {
            FlushCpuCachesOmpLocalV2();
        }

        if (!RunDownCpuFp16ResidentF16cAvx2ThreadPoolV2(
                &pool,
                w_down_fp16,
                h_mid.data(),
                y_f32.data())) {
            std::printf("warmup failed at iter=%d\n", i);
            pool.shutdown();
            return false;
        }
    }

    std::vector<float> ms_list;
    ms_list.reserve(static_cast<std::size_t>(args.iters));

    for (int i = 0; i < args.iters; ++i) {
        if (args.flush_cache) {
            FlushCpuCachesOmpLocalV2();
        }

        const auto t0 = std::chrono::steady_clock::now();
        if (!RunDownCpuFp16ResidentF16cAvx2ThreadPoolV2(
                &pool,
                w_down_fp16,
                h_mid.data(),
                y_f32.data())) {
            std::printf("benchmark failed at iter=%d\n", i);
            pool.shutdown();
            return false;
        }
        const auto t1 = std::chrono::steady_clock::now();

        const float ms = static_cast<float>(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
        ms_list.push_back(ms);
    }

    float sum = 0.0f;
    for (float v : ms_list) sum += v;
    const float mean =
        ms_list.empty() ? 0.0f : sum / static_cast<float>(ms_list.size());
    const float p50 = percentile(ms_list, 0.50f);
    const float p90 = percentile(ms_list, 0.90f);
    const float p99 = percentile(ms_list, 0.99f);

    std::printf(
        "CSV_HEADER,kind,threads,flush_cache,warmup,iters,mean_ms,p50_ms,p90_ms,p99_ms\n");
    std::printf(
        "CSV_ROW,down_fp16_f16c_avx2_threadpool,%d,%d,%d,%d,%g,%g,%g,%g\n",
        args.threads,
        args.flush_cache ? 1 : 0,
        args.warmup,
        args.iters,
        mean,
        p50,
        p90,
        p99);

    pool.shutdown();
    return true;
}

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    TestContext ctx;
    if (!InitTestContext(args, &ctx)) {
        CleanupTestContext(&ctx);
        return 1;
    }

    const bool ok = RunBenchmarkDownThreadPool(args, &ctx);
    CleanupTestContext(&ctx);
    return ok ? 0 : 1;
}
