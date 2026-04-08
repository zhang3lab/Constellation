#include "node_info.h"

#include <cstdio>
#include <utility>
#include <vector>

#include "expert_node_v2/backend/backend_registry_v2.h"

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

template <class TGpu>
void AppendGpus(
    std::vector<TGpu>* dst,
    std::vector<TGpu>* src) {
    dst->insert(
        dst->end(),
        std::make_move_iterator(src->begin()),
        std::make_move_iterator(src->end()));
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

    for (const auto& entry : expert_node_v2::GetBackendRegistryV2()) {
        if (entry.build_static == nullptr) {
            continue;
        }

        std::vector<common::StaticGpuInfo> gpus;
        if (!entry.build_static(
                next_worker_id,
                static_cast<std::uint32_t>(worker_port_base),
                &gpus)) {
            std::fprintf(stderr,
                         "[%s] static gpu probe failed for vendor=%s\n",
                         node.node_id.c_str(),
                         common::gpu_vendor_name(entry.vendor));
            continue;
        }

        auto& span =
            node.vendor_spans[static_cast<std::size_t>(entry.vendor)];
        span.worker_id_begin = next_worker_id;
        span.worker_count = static_cast<std::int32_t>(gpus.size());

        next_worker_id += static_cast<std::int32_t>(gpus.size());
        AppendGpus(&node.gpus, &gpus);
    }

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

    for (const auto& entry : expert_node_v2::GetBackendRegistryV2()) {
        if (entry.build_dynamic == nullptr) {
            continue;
        }

        const auto& span =
            static_info.vendor_spans[static_cast<std::size_t>(entry.vendor)];
        if (span.worker_id_begin < 0 || span.worker_count <= 0) {
            continue;
        }

        std::vector<common::DynamicGpuInfo> gpus;
        if (!entry.build_dynamic(span.worker_id_begin, &gpus)) {
            std::fprintf(stderr,
                         "[%s] dynamic gpu probe failed for vendor=%s\n",
                         node.node_id.c_str(),
                         common::gpu_vendor_name(entry.vendor));
            continue;
        }

        AppendGpus(&node.gpus, &gpus);
    }

    if (!ValidateWorkerIndexedGpus(node.node_id, node.gpus)) {
        return false;
    }

    *out = std::move(node);
    return true;
}
