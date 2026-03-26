#include <cstdio>
#include <memory>
#include <thread>
#include <vector>

#include "expert_node_v2/control.h"
#include "expert_node_v2/node_info.h"
#include "expert_node_v2/worker.h"

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    ControlState state;
    if (!BuildStaticNodeInfo(
            "node0",
            "127.0.0.1",
            40000,
            50000,
            &state.static_info)) {
        std::fprintf(stderr, "BuildStaticNodeInfo failed\n");
        return 1;
    }

    std::printf("[main] node_id=%s control_port=%d num_gpus=%zu\n",
                state.static_info.node_id.c_str(),
                state.static_info.control_port,
                state.static_info.gpus.size());

    std::thread control_thread([&state]() {
        RunControlLoop(&state);
    });

    std::vector<std::unique_ptr<GpuWorkerContextV2>> worker_contexts;
    std::vector<std::thread> worker_threads;

    worker_contexts.reserve(state.static_info.gpus.size());
    worker_threads.reserve(state.static_info.gpus.size());

    for (const auto& gpu : state.static_info.gpus) {
        auto ctx = std::make_unique<GpuWorkerContextV2>();
        ctx->worker_id = gpu.worker_id;
        ctx->worker_port = static_cast<int>(gpu.worker_port);
        ctx->state = &state;

        std::printf("[main] starting gpu worker worker_id=%d port=%d name=%s vendor=%u\n",
                    ctx->worker_id,
                    ctx->worker_port,
                    gpu.gpu_name.c_str(),
                    static_cast<unsigned>(gpu.gpu_vendor));

        worker_threads.emplace_back([raw = ctx.get()]() {
            RunGpuWorkerLoopV2(raw);
        });

        worker_contexts.push_back(std::move(ctx));
    }

    control_thread.join();
    for (auto& t : worker_threads) {
        t.join();
    }

    return 0;
}
