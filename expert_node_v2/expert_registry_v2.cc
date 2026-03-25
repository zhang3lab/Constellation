#include "expert_node_v2/expert_registry_v2.h"

#include <cstdio>
#include <utility>

namespace expert_node_v2 {
namespace {

HostTensorV2* select_incoming_slot(
    ExpertEntryV2* entry,
    common::TensorKind tensor_kind) {
    if (entry == nullptr) return nullptr;

    switch (tensor_kind) {
        case common::TensorKind::WUp:
            return &entry->incoming.w_up;
        case common::TensorKind::WUpScale:
            return &entry->incoming.w_up_scale;
        case common::TensorKind::WGate:
            return &entry->incoming.w_gate;
        case common::TensorKind::WGateScale:
            return &entry->incoming.w_gate_scale;
        case common::TensorKind::WDown:
            return &entry->incoming.w_down;
        case common::TensorKind::WDownScale:
            return &entry->incoming.w_down_scale;
        default:
            return nullptr;
    }
}

}  // namespace

void ExpertRegistryV2::clear() {
    entries_.clear();
}

ExpertEntryV2* ExpertRegistryV2::find_or_create_entry(int expert_id) {
    if (expert_id < 0) return nullptr;
    auto& e = entries_[expert_id];
    if (e.expert_id < 0) {
        e.expert_id = expert_id;
    }
    return &e;
}

ExpertEntryV2* ExpertRegistryV2::find_entry(int expert_id) {
    auto it = entries_.find(expert_id);
    if (it == entries_.end()) return nullptr;
    return &it->second;
}

const ExpertEntryV2* ExpertRegistryV2::find_entry(int expert_id) const {
    auto it = entries_.find(expert_id);
    if (it == entries_.end()) return nullptr;
    return &it->second;
}

bool ExpertRegistryV2::store_incoming_tensor(
    int expert_id,
    common::TensorKind tensor_kind,
    std::uint64_t total_bytes,
    std::string&& bytes) {
    ExpertEntryV2* entry = find_entry(expert_id);
    if (entry == nullptr) return false;

    HostTensorV2* slot = select_incoming_slot(entry, tensor_kind);
    if (slot == nullptr) return false;

    slot->bytes = std::move(bytes);
    slot->ready = true;

    if (static_cast<std::uint64_t>(slot->bytes.size()) != total_bytes) {
        slot->ready = false;
        return false;
    }

    entry->incoming_ready = entry->incoming.all_ready();
    return true;
}

bool ExpertRegistryV2::is_incoming_ready(int expert_id) const {
    const ExpertEntryV2* e = find_entry(expert_id);
    return e != nullptr && e->incoming_ready;
}

bool ExpertRegistryV2::is_resident_ready(int expert_id) const {
    const ExpertEntryV2* e = find_entry(expert_id);
    return e != nullptr && e->resident_ready;
}

std::size_t ExpertRegistryV2::size() const {
    return entries_.size();
}

void ExpertRegistryV2::debug_print() const {
    std::printf("[expert_registry_v2] num_entries=%zu\n", entries_.size());
    for (const auto& kv : entries_) {
        const auto& e = kv.second;
        std::printf("[expert_registry_v2] expert=%d gpu=%d incoming_ready=%d resident_ready=%d\n",
                    e.expert_id,
                    e.local_gpu_id,
                    e.incoming_ready ? 1 : 0,
                    e.resident_ready ? 1 : 0);
    }
}

}  // namespace expert_node_v2
