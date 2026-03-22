#include "expert_node/expert_runtime.h"

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

std::size_t ExpertRuntime::size() const {
    return loaded_experts_.size();
}

}  // namespace expert_node
