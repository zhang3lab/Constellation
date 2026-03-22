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

const DeviceExpertWeights* ExpertRuntime::find_device_weights(int expert_id) const {
    auto it = loaded_experts_.find(expert_id);
    if (it == loaded_experts_.end()) return nullptr;
    if (!it->second.ready) return nullptr;
    return &it->second.weights;
}

bool ExpertRuntime::is_expert_ready(int expert_id) const {
    auto it = loaded_experts_.find(expert_id);
    return it != loaded_experts_.end() && it->second.ready;
}

bool ExpertRuntime::execute_expert_stub(int expert_id) const {
    const LoadedExpert* expert = find_loaded_expert(expert_id);
    if (expert == nullptr) {
        std::fprintf(stderr, "[runtime] expert %d not found\n", expert_id);
        return false;
    }

    if (!expert->ready) {
        std::fprintf(stderr, "[runtime] expert %d not ready\n", expert_id);
        return false;
    }

    const DeviceExpertWeights* w = find_device_weights(expert_id);
    if (w == nullptr) {
        std::fprintf(stderr, "[runtime] device weights missing for expert %d\n", expert_id);
        return false;
    }

    if (w->w_up_ptr == nullptr || w->w_gate_ptr == nullptr || w->w_down_ptr == nullptr) {
        std::fprintf(stderr, "[runtime] incomplete device weights for expert %d\n", expert_id);
        return false;
    }

    std::printf("[runtime] execute stub expert=%d gpu=%d ready=%d\n",
                expert->expert_id,
                expert->local_gpu_id,
                static_cast<int>(expert->ready));

    std::printf("[runtime]   up=%p (%llu bytes)\n",
                w->w_up_ptr,
                static_cast<unsigned long long>(w->w_up_bytes));
    std::printf("[runtime]   gate=%p (%llu bytes)\n",
                w->w_gate_ptr,
                static_cast<unsigned long long>(w->w_gate_bytes));
    std::printf("[runtime]   down=%p (%llu bytes)\n",
                w->w_down_ptr,
                static_cast<unsigned long long>(w->w_down_bytes));

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
        std::printf("[runtime]   expert=%d gpu=%d ready=%d up=%p gate=%p down=%p\n",
                    e.expert_id,
                    e.local_gpu_id,
                    static_cast<int>(e.ready),
                    e.weights.w_up_ptr,
                    e.weights.w_gate_ptr,
                    e.weights.w_down_ptr);
    }
}

}  // namespace expert_node
