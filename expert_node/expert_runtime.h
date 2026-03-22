#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace expert_node {

struct DeviceExpertWeights {
    void* w_up_ptr = nullptr;
    void* w_gate_ptr = nullptr;
    void* w_down_ptr = nullptr;

    std::uint64_t w_up_bytes = 0;
    std::uint64_t w_gate_bytes = 0;
    std::uint64_t w_down_bytes = 0;
};

struct LoadedExpert {
    int expert_id = -1;
    int local_gpu_id = -1;
    DeviceExpertWeights weights;
    bool ready = false;
};

class ExpertRuntime {
public:
    bool register_loaded_expert(const LoadedExpert& expert);
    const LoadedExpert* find_loaded_expert(int expert_id) const;
    bool is_expert_ready(int expert_id) const;
    std::size_t size() const;

private:
    std::unordered_map<int, LoadedExpert> loaded_experts_;
};

}  // namespace expert_node
