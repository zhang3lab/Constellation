#include "expert_node_v2/expert_tensor_store_v2.h"

#include <utility>

ExpertTensorStateV2& ExpertTensorStoreV2::get_or_create(int expert_id) {
    auto [it, inserted] = experts_.try_emplace(expert_id);
    return it->second;
}

ExpertTensorStateV2* ExpertTensorStoreV2::find(int expert_id) {
    auto it = experts_.find(expert_id);
    if (it == experts_.end()) return nullptr;
    return &it->second;
}

const ExpertTensorStateV2* ExpertTensorStoreV2::find(int expert_id) const {
    auto it = experts_.find(expert_id);
    if (it == experts_.end()) return nullptr;
    return &it->second;
}

void ExpertTensorStoreV2::erase(int expert_id) {
    experts_.erase(expert_id);
}

void ExpertTensorStoreV2::clear() {
    experts_.clear();
}

bool ExpertTensorStoreV2::store_tensor(
    int expert_id,
    common::TensorKind tensor_kind,
    std::vector<std::uint8_t> bytes,
    std::vector<std::uint64_t> shape,
    std::string dtype) {
    ExpertTensorBundleV2* bundle = find_or_create_bundle(expert_id);
    if (bundle == nullptr) return false;

    HostTensorV2* slot = select_slot(bundle, tensor_kind);
    if (slot == nullptr) return false;

    slot->bytes = std::move(bytes);
    slot->meta.shape = std::move(shape);
    slot->meta.dtype = std::move(dtype);
    return true;
}

bool ExpertTensorStoreV2::mark_ready(int expert_id, common::TensorKind tensor_kind) {
    ExpertTensorBundleV2* bundle = find_or_create_bundle(expert_id);
    if (bundle == nullptr) return false;

    HostTensorV2* slot = select_slot(bundle, tensor_kind);
    if (slot == nullptr) return false;

    if (slot->bytes.empty() || slot->meta.shape.empty() || slot->meta.dtype.empty()) {
        return false;
    }

    slot->ready = true;
    return true;
}

bool ExpertTensorStoreV2::is_tensor_ready(int expert_id, TensorKind tensor_kind) const {
    const HostTensorV2* slot = get_tensor_slot(expert_id, tensor_kind);
    if (slot == nullptr) return false;
    return slot->ready;
}

bool ExpertTensorStoreV2::is_expert_ready(int expert_id) const {
    const ExpertTensorStateV2* state = find(expert_id);
    if (state == nullptr) return false;
    return state->all_ready();
}

HostTensorV2* ExpertTensorStoreV2::get_tensor_slot(int expert_id, TensorKind tensor_kind) {
    ExpertTensorStateV2* state = find(expert_id);
    if (state == nullptr) return nullptr;
    return select_slot(&state->bundle, tensor_kind);
}

const HostTensorV2* ExpertTensorStoreV2::get_tensor_slot(int expert_id, TensorKind tensor_kind) const {
    const ExpertTensorStateV2* state = find(expert_id);
    if (state == nullptr) return nullptr;
    return select_slot(&state->bundle, tensor_kind);
}

HostTensorV2* ExpertTensorStoreV2::select_slot(ExpertTensorBundleV2* bundle, TensorKind tensor_kind) {
    if (bundle == nullptr) return nullptr;

    switch (tensor_kind) {
        case TensorKind::WUp:
            return &bundle->w_up;
        case TensorKind::WUpScale:
            return &bundle->w_up_scale;
        case TensorKind::WGate:
            return &bundle->w_gate;
        case TensorKind::WGateScale:
            return &bundle->w_gate_scale;
        case TensorKind::WDown:
            return &bundle->w_down;
        case TensorKind::WDownScale:
            return &bundle->w_down_scale;
        default:
            return nullptr;
    }
}

const HostTensorV2* ExpertTensorStoreV2::select_slot(const ExpertTensorBundleV2* bundle, TensorKind tensor_kind) {
    if (bundle == nullptr) return nullptr;

    switch (tensor_kind) {
        case TensorKind::WUp:
            return &bundle->w_up;
        case TensorKind::WUpScale:
            return &bundle->w_up_scale;
        case TensorKind::WGate:
            return &bundle->w_gate;
        case TensorKind::WGateScale:
            return &bundle->w_gate_scale;
        case TensorKind::WDown:
            return &bundle->w_down;
        case TensorKind::WDownScale:
            return &bundle->w_down_scale;
        default:
            return nullptr;
    }
}
