#include "expert_node_v2/expert_registry_v2.h"

#include <algorithm>
#include <cstdio>
#include <unordered_set>
#include <utility>

#include "expert_node_v2/backend/backend_registry_v2.h"
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

void ExpertRegistryV2::Reset() {
    for (auto& kv : entries_) {
        ExpertEntryV2& entry = kv.second;

        for (auto& rk : entry.residents) {
            ExpertResidentSlotV2& slot = rk.second;

            if (slot.resident_ready) {
                const expert_node_v2::BackendRegistryEntryV2* backend =
                    expert_node_v2::FindBackendRegistryEntryV2(slot.vendor);
                if (backend == nullptr || backend->free_expert_weights == nullptr) {
                    std::fprintf(stderr,
                                 "[expert_registry_v2] missing backend free during reset "
                                 "expert=%d worker=%d vendor=%u\n",
                                 entry.expert_id,
                                 slot.worker_id,
                                 static_cast<unsigned>(slot.vendor));
                    std::abort();
                }

                backend->free_expert_weights(&slot.storage);
                slot.resident_ready = false;
                slot.vendor = common::GpuVendor::Unknown;
            }
        }

        entry.residents.clear();
        entry.incoming.clear();
        entry.incoming_ready = false;
    }

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

bool ExpertRegistryV2::ClearIncoming(int expert_id) {
    if (expert_id < 0) return false;

    auto it = entries_.find(expert_id);
    if (it == entries_.end()) return false;

    ExpertEntryV2& entry = it->second;
    entry.incoming.clear();
    entry.incoming_ready = false;
    return true;
}

bool ExpertRegistryV2::Update(
    int expert_id,
    int worker_id,
    common::GpuVendor vendor,
    const std::array<common::VendorWorkerSpan, common::kGpuVendorCount>& vendor_spans) {
    if (expert_id < 0 || worker_id < 0) {
        std::fprintf(stderr,
                     "[expert_registry_v2] Update invalid ids "
                     "expert=%d worker=%d vendor=%u(%s)\n",
                     expert_id,
                     worker_id,
                     static_cast<unsigned>(vendor),
                     common::gpu_vendor_name(vendor));
        return false;
    }

    const expert_node_v2::BackendRegistryEntryV2* backend =
        expert_node_v2::FindBackendRegistryEntryV2(vendor);
    if (backend == nullptr) {
        std::fprintf(stderr,
                     "[expert_registry_v2] Update backend not found "
                     "expert=%d worker=%d vendor=%u(%s)\n",
                     expert_id,
                     worker_id,
                     static_cast<unsigned>(vendor),
                     common::gpu_vendor_name(vendor));
        return false;
    }
    if (backend->upload_expert == nullptr ||
        backend->free_expert_weights == nullptr) {
        std::fprintf(stderr,
                     "[expert_registry_v2] Update backend missing callbacks "
                     "expert=%d worker=%d vendor=%u(%s) "
                     "upload_expert=%p free_expert_weights=%p\n",
                     expert_id,
                     worker_id,
                     static_cast<unsigned>(vendor),
                     common::gpu_vendor_name(vendor),
                     reinterpret_cast<const void*>(backend->upload_expert),
                     reinterpret_cast<const void*>(backend->free_expert_weights));
        return false;
    }

    const common::VendorWorkerSpan& vendor_span =
        vendor_spans[static_cast<std::size_t>(vendor)];

    if (worker_id < vendor_span.worker_id_begin) {
        std::fprintf(stderr,
                     "[expert_registry_v2] Update worker below vendor span "
                     "expert=%d worker=%d vendor=%u(%s) "
                     "span_begin=%d span_count=%d\n",
                     expert_id,
                     worker_id,
                     static_cast<unsigned>(vendor),
                     common::gpu_vendor_name(vendor),
                     vendor_span.worker_id_begin,
                     vendor_span.worker_count);
        return false;
    }
    if (worker_id >= vendor_span.worker_id_begin + vendor_span.worker_count) {
        std::fprintf(stderr,
                     "[expert_registry_v2] Update worker above vendor span "
                     "expert=%d worker=%d vendor=%u(%s) "
                     "span_begin=%d span_count=%d\n",
                     expert_id,
                     worker_id,
                     static_cast<unsigned>(vendor),
                     common::gpu_vendor_name(vendor),
                     vendor_span.worker_id_begin,
                     vendor_span.worker_count);
        return false;
    }

    const int local_gpu_id = worker_id - vendor_span.worker_id_begin;

    ExpertEntryV2* entry = FindOrCreateEntry_(expert_id);
    if (entry == nullptr) {
        std::fprintf(stderr,
                     "[expert_registry_v2] Update FindOrCreateEntry_ failed "
                     "expert=%d worker=%d vendor=%u(%s)\n",
                     expert_id,
                     worker_id,
                     static_cast<unsigned>(vendor),
                     common::gpu_vendor_name(vendor));
        return false;
    }
    if (!entry->incoming_ready) {
        std::fprintf(stderr,
                     "[expert_registry_v2] Update incoming not ready "
                     "expert=%d worker=%d vendor=%u(%s)\n",
                     expert_id,
                     worker_id,
                     static_cast<unsigned>(vendor),
                     common::gpu_vendor_name(vendor));
        return false;
    }

    ExpertResidentSlotV2* slot = FindOrCreateResident_(expert_id, worker_id);
    if (slot == nullptr) {
        std::fprintf(stderr,
                     "[expert_registry_v2] Update FindOrCreateResident_ failed "
                     "expert=%d worker=%d vendor=%u(%s)\n",
                     expert_id,
                     worker_id,
                     static_cast<unsigned>(vendor),
                     common::gpu_vendor_name(vendor));
        return false;
    }

    if (slot->resident_ready) {
        std::fprintf(stderr,
                     "[expert_registry_v2] Update replacing existing resident "
                     "expert=%d worker=%d old_vendor=%u(%s) new_vendor=%u(%s)\n",
                     expert_id,
                     worker_id,
                     static_cast<unsigned>(slot->vendor),
                     common::gpu_vendor_name(slot->vendor),
                     static_cast<unsigned>(vendor),
                     common::gpu_vendor_name(vendor));

        backend->free_expert_weights(&slot->storage);
        slot->resident_ready = false;
        slot->vendor = common::GpuVendor::Unknown;
    }

    const bool ok =
        backend->upload_expert(local_gpu_id, entry->incoming, &slot->storage);

    if (!ok) {
        std::fprintf(stderr,
                     "[expert_registry_v2] Update upload failed "
                     "expert=%d worker=%d local_gpu_id=%d vendor=%u(%s)\n",
                     expert_id,
                     worker_id,
                     local_gpu_id,
                     static_cast<unsigned>(vendor),
                     common::gpu_vendor_name(vendor));
        slot->resident_ready = false;
        slot->vendor = common::GpuVendor::Unknown;
        return false;
    }

    slot->resident_ready = true;
    slot->vendor = vendor;

    return true;
}

std::vector<common::ResidentInventoryWorkerInfo>
ExpertRegistryV2::BuildResidentInventory(
    const common::StaticNodeInfo& static_info) const {
    std::vector<common::ResidentInventoryWorkerInfo> workers;
    workers.reserve(static_info.gpus.size());

    std::unordered_map<int, std::size_t> worker_index;
    worker_index.reserve(static_info.gpus.size());

    for (const auto& gpu : static_info.gpus) {
        const std::size_t idx = workers.size();
        workers.push_back(common::ResidentInventoryWorkerInfo{
            .worker_id = gpu.worker_id,
            .expert_ids = {},
        });
        worker_index.emplace(gpu.worker_id, idx);
    }

    for (const auto& [expert_id, entry] : entries_) {
        (void)expert_id;
        for (const auto& [worker_id, slot] : entry.residents) {
            if (!slot.resident_ready) {
                continue;
            }

            auto it = worker_index.find(worker_id);
            if (it == worker_index.end()) {
                continue;
            }

            workers[it->second].expert_ids.push_back(entry.expert_id);
        }
    }

    std::sort(
        workers.begin(),
        workers.end(),
        [](const common::ResidentInventoryWorkerInfo& a,
           const common::ResidentInventoryWorkerInfo& b) {
            return a.worker_id < b.worker_id;
        });

    for (auto& worker : workers) {
        std::sort(worker.expert_ids.begin(), worker.expert_ids.end());
    }

    return workers;
}

std::size_t ExpertRegistryV2::DropNonTargetResidents(
    const std::vector<common::PlacementAssignment>& assignments) {
    auto make_key = [](int expert_id, int worker_id) -> std::uint64_t {
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(expert_id)) << 32) |
               static_cast<std::uint32_t>(worker_id);
    };

    std::unordered_set<std::uint64_t> keep;
    keep.reserve(assignments.size());

    for (const auto& a : assignments) {
        if (a.expert_id < 0 || a.worker_id < 0) {
            continue;
        }
        keep.insert(make_key(static_cast<int>(a.expert_id),
                             static_cast<int>(a.worker_id)));
    }

    std::size_t num_dropped = 0;

    for (auto& [expert_id, entry] : entries_) {
        (void)expert_id;

        for (auto it = entry.residents.begin(); it != entry.residents.end();) {
            const int worker_id = it->first;
            ExpertResidentSlotV2& slot = it->second;

            if (!slot.resident_ready) {
                it = entry.residents.erase(it);
                continue;
            }

            const std::uint64_t key = make_key(entry.expert_id, worker_id);
            if (keep.find(key) != keep.end()) {
                ++it;
                continue;
            }

            const expert_node_v2::BackendRegistryEntryV2* backend =
                expert_node_v2::FindBackendRegistryEntryV2(slot.vendor);

            if (backend == nullptr || backend->free_expert_weights == nullptr) {
                std::fprintf(stderr,
                             "[expert_registry_v2] DropNonTargetResidents missing backend "
                             "expert=%d worker=%d vendor=%u(%s)\n",
                             entry.expert_id,
                             worker_id,
                             static_cast<unsigned>(slot.vendor),
                             common::gpu_vendor_name(slot.vendor));
                slot.resident_ready = false;
                slot.vendor = common::GpuVendor::Unknown;
                it = entry.residents.erase(it);
                ++num_dropped;
                continue;
            }

            std::fprintf(stderr,
                         "[expert_registry_v2] DropNonTargetResidents dropping "
                         "expert=%d worker=%d vendor=%u(%s)\n",
                         entry.expert_id,
                         worker_id,
                         static_cast<unsigned>(slot.vendor),
                         common::gpu_vendor_name(slot.vendor));

            backend->free_expert_weights(&slot.storage);
            slot.resident_ready = false;
            slot.vendor = common::GpuVendor::Unknown;

            it = entry.residents.erase(it);
            ++num_dropped;
        }
    }

    return num_dropped;
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
