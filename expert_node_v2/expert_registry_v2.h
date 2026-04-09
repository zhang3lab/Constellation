#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "common/protocol.h"
#include "common/types.h"
#include "expert_node_v2/expert_format_v2.h"

namespace expert_node_v2 {

struct ExpertResidentSlotV2 {
    int worker_id = -1;
    common::GpuVendor vendor = common::GpuVendor::Unknown;
    ExpertDeviceStorageV2 storage;
    bool resident_ready = false;
};

struct ExpertEntryV2 {
    int expert_id = -1;

    ExpertTensorBundleV2 incoming;
    bool incoming_ready = false;

    std::unordered_map<int, ExpertResidentSlotV2> residents;
};

class ExpertRegistryV2 {
public:
    void Reset();

    bool StoreIncomingTensor(
        int expert_id,
        common::TensorKind tensor_kind,
        std::uint64_t total_bytes,
        std::vector<std::uint8_t>&& bytes,
        common::TensorMeta&& meta);

    bool ClearIncoming(int expert_id);

    bool Update(
        int expert_id,
        int worker_id,
        common::GpuVendor vendor,
        const std::array<common::VendorWorkerSpan, common::kGpuVendorCount>& vendor_spans);

    const ExpertEntryV2* FindEntry(int expert_id) const;
    const ExpertDeviceStorageV2* FindDeviceStorage(
        int expert_id,
        int worker_id) const;

    std::size_t size() const;
    void DebugPrint() const;

private:
    ExpertEntryV2* FindOrCreateEntry_(int expert_id);
    ExpertResidentSlotV2* FindOrCreateResident_(int expert_id, int worker_id);
    const ExpertResidentSlotV2* FindResident_(int expert_id, int worker_id) const;

private:
    std::unordered_map<int, ExpertEntryV2> entries_;
};

}  // namespace expert_node_v2
