#include "expert_node_v2/expert_registry_v2.h"

#include <cstdio>
#include <utility>

#include "expert_node_v2/build_config_v2.h"

#if EXPERT_NODE_V2_ENABLE_CUDA
#include "expert_node_v2/backend/cuda/backend_cuda_v2.h"
#endif

#if EXPERT_NODE_V2_ENABLE_AMD
#include "expert_node_v2/backend/amd/backend_amd_v2.h"
#endif

#if EXPERT_NODE_V2_ENABLE_INTEL
#include "expert_node_v2/backend/intel/backend_intel_v2.h"
#endif

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

ExpertEntryV2* ExpertRegistryV2::FindOrCreateEntry_(int expert_id) {
    if (expert_id < 0) return nullptr;

    auto& e = entries_[expert_id];
    if (e.expert_id < 0) {
        e.expert_id = expert_id;
    }
    return &e;
}

ExpertResidentSlotV2* ExpertRegistryV2::FindOrCreateResident_(
    int expert_id,
    int worker_id) {
    if (worker_id < 0) return nullptr;

    ExpertEntryV2* entry = FindOrCreateEntry_(expert_id);
    if (entry == nullptr) return nullptr;

    auto& slot = entry->residents[worker_id];
    if (slot.worker_id < 0) {
        slot.worker_id = worker_id;
    }
    return &slot;
}

const ExpertResidentSlotV2* ExpertRegistryV2::FindResident_(
    int expert_id,
    int worker_id) const {
    auto it = entries_.find(expert_id);
    if (it == entries_.end()) return nullptr;

    auto jt = it->second.residents.find(worker_id);
    if (jt == it->second.residents.end()) return nullptr;

    return &jt->second;
}

bool ExpertRegistryV2::StoreIncomingTensor(
    int expert_id,
    common::TensorKind tensor_kind,
    std::uint64_t total_bytes,
    std::vector<std::uint8_t>&& bytes,
    common::TensorMeta&& meta) {
    ExpertEntryV2* entry = FindOrCreateEntry_(expert_id);
    if (entry == nullptr) return false;

    HostTensorV2* slot = select_incoming_slot(entry, tensor_kind);
    if (slot == nullptr) return false;

    if (static_cast<std::uint64_t>(bytes.size()) != total_bytes) {
        return false;
    }

    slot->bytes = std::move(bytes);
    slot->meta = std::move(meta);
    slot->ready = true;

    entry->incoming_ready = entry->incoming.all_ready();
    return true;
}

bool ExpertRegistryV2::Update(
    int expert_id,
    int worker_id,
    common::GpuVendor vendor,
    const std::array<common::VendorWorkerSpan, 256>& vendor_spans) {
    if (expert_id < 0 || worker_id < 0) return false;

    const expert_node_v2::BackendRegistryEntryV2* backend =
        expert_node_v2::FindBackendRegistryEntryV2(vendor);
    if (backend == nullptr) {
        return false;
    }
    if (backend->upload_expert == nullptr ||
        backend->free_expert_weights == nullptr) {
        return false;
    }

    const common::VendorWorkerSpan& vendor_span =
        vendor_spans[static_cast<std::size_t>(vendor)];

    if (worker_id < vendor_span.worker_id_begin) return false;
    if (worker_id >= vendor_span.worker_id_begin + vendor_span.worker_count) {
        return false;
    }

    const int local_gpu_id = worker_id - vendor_span.worker_id_begin;

    ExpertEntryV2* entry = FindOrCreateEntry_(expert_id);
    if (entry == nullptr) return false;
    if (!entry->incoming_ready) return false;

    ExpertResidentSlotV2* slot = FindOrCreateResident_(expert_id, worker_id);
    if (slot == nullptr) return false;

    if (slot->resident_ready) {
        backend->free_expert_weights(&slot->storage);
        slot->storage.clear();
        slot->resident_ready = false;
    }

    const bool ok =
        backend->upload_expert(local_gpu_id, entry->incoming, &slot->storage);

    if (!ok) {
        std::fprintf(stderr,
                     "[expert_registry_v2] upload failed expert=%d worker=%d vendor=%u\n",
                     expert_id,
                     worker_id,
                     static_cast<unsigned>(vendor));
        slot->storage.clear();
        slot->resident_ready = false;
        return false;
    }

    slot->resident_ready = true;
    return true;
}

const ExpertEntryV2* ExpertRegistryV2::FindEntry(int expert_id) const {
    auto it = entries_.find(expert_id);
    if (it == entries_.end()) return nullptr;
    return &it->second;
}

const ExpertDeviceStorageV2* ExpertRegistryV2::FindDeviceStorage(
    int expert_id,
    int worker_id) const {
    const ExpertResidentSlotV2* slot = FindResident_(expert_id, worker_id);
    if (slot == nullptr) return nullptr;
    if (!slot->resident_ready) return nullptr;
    return &slot->storage;
}

std::size_t ExpertRegistryV2::size() const {
    return entries_.size();
}

void ExpertRegistryV2::DebugPrint() const {
    std::printf("[expert_registry_v2] num_entries=%zu\n", entries_.size());
    for (const auto& kv : entries_) {
        const auto& e = kv.second;
        std::printf("[expert_registry_v2] expert=%d incoming_ready=%d residents=%zu\n",
                    e.expert_id,
                    e.incoming_ready ? 1 : 0,
                    e.residents.size());
        for (const auto& rv : e.residents) {
            const auto& slot = rv.second;
            std::printf("[expert_registry_v2]   worker=%d resident_ready=%d\n",
                        slot.worker_id,
                        slot.resident_ready ? 1 : 0);
        }
    }
}

}  // namespace expert_node_v2
