#include "node_info.h"

#include <cstdio>
#include <utility>
#include <vector>

#include "expert_node_v2/cuda/gpu_info_cuda_v2.h"
#include "expert_node_v2/amd/gpu_info_amd_v2.h"
#include "expert_node_v2/intel/gpu_info_intel_v2.h"

namespace {

template <class TGpu>
bool ValidateWorkerIndexedGpus(
    const std::string& node_id,
    const std::vector<TGpu>& gpus) {
    for (std::size_t i = 0; i < gpus.size(); ++i) {
        if (gpus[i].worker_id != static_cast<std::int32_t>(i)) {
            std::fprintf(stderr,
                         "[%s] gpu index mismatch: i=%zu worker_id=%d\n",
                         node_id.c_str(),
                         i,
                         gpus[i].worker_id);
            return false;
        }
    }
    return true;
}

}  // namespace

bool BuildStaticNodeInfo(
    const std::string& node_id,
    const std::string& host,
    int control_port,
    int worker_port_base,
    common::StaticNodeInfo* out) {
    if (out == nullptr) return false;

    common::StaticNodeInfo node;
    node.node_id = node_id;
    node.host = host;
    node.control_port = control_port;

    for (auto& span : node.vendor_spans) {
        span.worker_id_begin = -1;
        span.worker_count = 0;
    }

    std::int32_t next_worker_id = 0;

#if EXPERT_NODE_V2_ENABLE_CUDA
    {
        std::vector<common::StaticGpuInfo> gpus;
        if (!BuildLocalCudaGpuInfosV2(
                next_worker_id,
                static_cast<std::uint32_t>(worker_port_base),
                &gpus)) {
            std::fprintf(stderr, "BuildLocalCudaGpuInfosV2 failed\n");
        } else {
            auto& span =
                node.vendor_spans[static_cast<std::size_t>(common::GpuVendor::Nvidia)];
            span.worker_id_begin = next_worker_id;
            span.worker_count = static_cast<std::int32_t>(gpus.size());

            next_worker_id += static_cast<std::int32_t>(gpus.size());

            node.gpus.insert(
                node.gpus.end(),
                std::make_move_iterator(gpus.begin()),
                std::make_move_iterator(gpus.end()));
        }
    }
#endif

#if EXPERT_NODE_V2_ENABLE_AMD
    {
        std::vector<common::StaticGpuInfo> gpus;
        // TODO: replace stub with real AMD probe.
        // bool ok = BuildLocalAmdGpuInfosV2(
        //     next_worker_id,
        //     static_cast<std::uint32_t>(worker_port_base),
        //     &gpus);
        bool ok = true;

        if (!ok) {
            std::fprintf(stderr, "BuildLocalAmdGpuInfosV2 failed\n");
        } else {
            auto& span =
                node.vendor_spans[static_cast<std::size_t>(common::GpuVendor::AMD)];
            span.worker_id_begin = next_worker_id;
            span.worker_count = static_cast<std::int32_t>(gpus.size());

            next_worker_id += static_cast<std::int32_t>(gpus.size());

            node.gpus.insert(
                node.gpus.end(),
                std::make_move_iterator(gpus.begin()),
                std::make_move_iterator(gpus.end()));
        }
    }
#endif

#if EXPERT_NODE_V2_ENABLE_INTEL
    {
        std::vector<common::StaticGpuInfo> gpus;
        // TODO: replace stub with real Intel probe.
        // bool ok = BuildLocalIntelGpuInfosV2(
        //     next_worker_id,
        //     static_cast<std::uint32_t>(worker_port_base),
        //     &gpus);
        bool ok = true;

        if (!ok) {
            std::fprintf(stderr, "BuildLocalIntelGpuInfosV2 failed\n");
        } else {
            auto& span =
                node.vendor_spans[static_cast<std::size_t>(common::GpuVendor::Intel)];
            span.worker_id_begin = next_worker_id;
            span.worker_count = static_cast<std::int32_t>(gpus.size());

            next_worker_id += static_cast<std::int32_t>(gpus.size());

            node.gpus.insert(
                node.gpus.end(),
                std::make_move_iterator(gpus.begin()),
                std::make_move_iterator(gpus.end()));
        }
    }
#endif

    if (!ValidateWorkerIndexedGpus(node.node_id, node.gpus)) {
        return false;
    }

    *out = std::move(node);
    return true;
}

bool BuildDynamicNodeInfo(
    const common::StaticNodeInfo& static_info,
    common::NodeStatus node_status,
    common::DynamicNodeInfo* out) {
    if (out == nullptr) return false;

    common::DynamicNodeInfo node;
    node.node_id = static_info.node_id;
    node.node_status = node_status;

#if EXPERT_NODE_V2_ENABLE_CUDA
    {
        const auto& span =
            static_info.vendor_spans[static_cast<std::size_t>(common::GpuVendor::Nvidia)];

        if (span.worker_id_begin >= 0 && span.worker_count > 0) {
            std::vector<common::DynamicGpuInfo> gpus;
            if (!BuildLocalCudaDynamicGpuInfosV2(
                    span.worker_id_begin,
                    &gpus)) {
                std::fprintf(stderr, "BuildLocalCudaDynamicGpuInfosV2 failed\n");
            } else {
                node.gpus.insert(
                    node.gpus.end(),
                    std::make_move_iterator(gpus.begin()),
                    std::make_move_iterator(gpus.end()));
            }
        }
    }
#endif

#if EXPERT_NODE_V2_ENABLE_AMD
    {
        const auto& span =
            static_info.vendor_spans[static_cast<std::size_t>(common::GpuVendor::AMD)];

        if (span.worker_id_begin >= 0 && span.worker_count > 0) {
            std::vector<common::DynamicGpuInfo> gpus;
            if (!BuildLocalAmdDynamicGpuInfosV2(
                    span.worker_id_begin,
                    &gpus)) {
                std::fprintf(stderr, "BuildLocalAmdDynamicGpuInfosV2 failed\n");
            } else {
                node.gpus.insert(
                    node.gpus.end(),
                    std::make_move_iterator(gpus.begin()),
                    std::make_move_iterator(gpus.end()));
            }
        }
    }
#endif

#if EXPERT_NODE_V2_ENABLE_INTEL
    {
        const auto& span =
            static_info.vendor_spans[static_cast<std::size_t>(common::GpuVendor::Intel)];

        if (span.worker_id_begin >= 0 && span.worker_count > 0) {
            std::vector<common::DynamicGpuInfo> gpus;
            if (!BuildLocalIntelDynamicGpuInfosV2(
                    span.worker_id_begin,
                    &gpus)) {
                std::fprintf(stderr, "BuildLocalIntelDynamicGpuInfosV2 failed\n");
            } else {
                node.gpus.insert(
                    node.gpus.end(),
                    std::make_move_iterator(gpus.begin()),
                    std::make_move_iterator(gpus.end()));
            }
        }
    }
#endif

    if (!ValidateWorkerIndexedGpus(node.node_id, node.gpus)) {
        return false;
    }

    *out = std::move(node);
    return true;
}
