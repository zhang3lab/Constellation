#include "expert_node/expert_runtime.h"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace expert_node {

bool ExpertRuntime::register_loaded_expert(const LoadedExpert& expert) {
    if (!expert.ready) return false;
    loaded_experts_[expert.expert_id] = expert;
    return true;
}

const LoadedExpert* ExpertRuntime::find_loaded_expert(int expert_id) const {
    auto it = loaded_experts_.find(expert_id);
    if (it == loaded_experts_.end()) return nullptr;
    return &it->second;
}

bool ExpertRuntime::is_expert_ready(int expert_id) const {
    auto it = loaded_experts_.find(expert_id);
    return it != loaded_experts_.end() && it->second.ready;
}

bool ExpertRuntime::execute_expert_stub_with_activation(
    int expert_id,
    const void* activation_ptr,
    std::uint64_t activation_bytes) const {
    const LoadedExpert* expert = find_loaded_expert(expert_id);
    if (expert == nullptr) {
        std::fprintf(stderr, "[runtime] expert %d not found\n", expert_id);
        return false;
    }

    if (!expert->ready) {
        std::fprintf(stderr, "[runtime] expert %d not ready\n", expert_id);
        return false;
    }

    if (activation_ptr == nullptr || activation_bytes == 0) {
        std::fprintf(stderr, "[runtime] invalid activation for expert %d\n", expert_id);
        return false;
    }

    std::printf("[runtime] execute stub expert=%d gpu=%d ready=%d activation=%p bytes=%llu\n",
                expert->expert_id,
                expert->local_gpu_id,
                static_cast<int>(expert->ready),
                activation_ptr,
                static_cast<unsigned long long>(activation_bytes));

    std::printf("[runtime]   up_w=%p up_s=%p\n",
                expert->mlp.w_up.weights,
                expert->mlp.w_up.scales);
    std::printf("[runtime]   gate_w=%p gate_s=%p\n",
                expert->mlp.w_gate.weights,
                expert->mlp.w_gate.scales);
    std::printf("[runtime]   down_w=%p down_s=%p\n",
                expert->mlp.w_down.weights,
                expert->mlp.w_down.scales);

    return true;
}

std::size_t ExpertRuntime::size() const {
    return loaded_experts_.size();
}

void ExpertRuntime::debug_print() const {
    std::vector<int> ids;
    ids.reserve(loaded_experts_.size());
    for (const auto& kv : loaded_experts_) {
        ids.push_back(kv.first);
    }
    std::sort(ids.begin(), ids.end());

    std::printf("[runtime] loaded experts = %zu\n", loaded_experts_.size());
    for (int expert_id : ids) {
        const auto& e = loaded_experts_.at(expert_id);
        std::printf("[runtime]   expert=%d gpu=%d ready=%d up_w=%p gate_w=%p down_w=%p\n",
                    e.expert_id,
                    e.local_gpu_id,
                    static_cast<int>(e.ready),
                    e.mlp.w_up.weights,
                    e.mlp.w_gate.weights,
                    e.mlp.w_down.weights);
    }
}

}  // namespace expert_node
