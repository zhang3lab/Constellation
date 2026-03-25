#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/protocol.h"
#include "expert_node_v2/expert_format_v2.h"

namespace expert_node_v2 {

struct ExpertEntryV2 {
    int expert_id = -1;
    int local_gpu_id = -1;

    ExpertTensorBundleV2 incoming;
    bool incoming_ready = false;

    ExpertDeviceStorageV2 storage;
    bool resident_ready = false;
};

class ExpertRegistryV2 {
public:
    void clear();

    ExpertEntryV2* find_or_create_entry(int expert_id);
    ExpertEntryV2* find_entry(int expert_id);
    const ExpertEntryV2* find_entry(int expert_id) const;

    bool store_incoming_tensor(
        int expert_id,
        common::TensorKind tensor_kind,
        std::uint64_t total_bytes,
        std::string&& bytes);

    bool is_incoming_ready(int expert_id) const;
    bool is_resident_ready(int expert_id) const;

    std::size_t size() const;
    void debug_print() const;

private:
    std::unordered_map<int, ExpertEntryV2> entries_;
};

}  // namespace expert_node_v2
