#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "expert_node_v2/expert_format_v2.h"
#include "common/protocol.h"

using common::TensorKind;

struct ExpertTensorStateV2 {
    ExpertTensorBundleV2 bundle;

    bool all_ready() const {
        return bundle.all_ready();
    }

    void clear() {
        bundle.clear();
    }
};

class ExpertTensorStoreV2 {
public:
    ExpertTensorStoreV2() = default;
    ~ExpertTensorStoreV2() = default;

    ExpertTensorStateV2& get_or_create(int expert_id);

    ExpertTensorStateV2* find(int expert_id);
    const ExpertTensorStateV2* find(int expert_id) const;

    void erase(int expert_id);
    void clear();

    // Store raw tensor content into the slot selected by tensor_kind.
    // This does not automatically mark the tensor as ready.
    bool store_tensor(
        int expert_id,
        TensorKind tensor_kind,
        std::vector<std::uint8_t> bytes,
        std::vector<std::uint64_t> shape,
        std::string dtype);

    // Mark one tensor as fully received.
    bool mark_ready(int expert_id, TensorKind tensor_kind);

    bool is_tensor_ready(int expert_id, TensorKind tensor_kind) const;
    bool is_expert_ready(int expert_id) const;

    HostTensorV2* get_tensor_slot(int expert_id, TensorKind tensor_kind);
    const HostTensorV2* get_tensor_slot(int expert_id, TensorKind tensor_kind) const;

private:
    static HostTensorV2* select_slot(ExpertTensorBundleV2* bundle, TensorKind tensor_kind);
    static const HostTensorV2* select_slot(const ExpertTensorBundleV2* bundle, TensorKind tensor_kind);

private:
    std::unordered_map<int, ExpertTensorStateV2> experts_;
};
