#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include "expert_node/kernel/expert.h"

namespace expert_node {

struct DevicePackedMatrixOwner {
    PackedRowMajorMatrix view;
    uint8_t* weights = nullptr;
    float* scales = nullptr;
};

struct DevicePackedMlpOwner {
    DevicePackedMatrixOwner w_up;
    DevicePackedMatrixOwner w_gate;
    DevicePackedMatrixOwner w_down;
};

struct LoadedExpert {
    int expert_id = -1;
    int local_gpu_id = -1;
    DevicePackedMlpOwner packed;
    DeviceMlpView mlp;
    bool ready = false;
};

class ExpertRuntime {
public:
    bool register_loaded_expert(const LoadedExpert& expert);
    const LoadedExpert* find_loaded_expert(int expert_id) const;
    bool is_expert_ready(int expert_id) const;
    bool execute_expert_stub_with_activation(
        int expert_id,
        const void* activation_ptr,
        std::uint64_t activation_bytes) const;
    std::size_t size() const;
    void debug_print() const;

private:
    std::unordered_map<int, LoadedExpert> loaded_experts_;
};

}  // namespace expert_node
