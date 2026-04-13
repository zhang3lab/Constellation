#include "control.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/header_codec.h"
#include "common/infer_codec.h"
#include "common/inventory_codec.h"
#include "common/placement_codec.h"
#include "common/protocol.h"
#include "common/resident_inventory_codec.h"
#include "common/socket_utils.h"
#include "common/types.h"
#include "common/weight_codec.h"
#include "expert_node_v2/expert_registry_v2.h"
#include "expert_node_v2/node_info.h"

namespace {

int ListenTcp(int port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(port));

    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        return -1;
    }
    if (::listen(fd, 16) != 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}

const char* TensorKindName(common::TensorKind k) {
    switch (k) {
        case common::TensorKind::WUp:
            return "w_up";
        case common::TensorKind::WGate:
            return "w_gate";
        case common::TensorKind::WDown:
            return "w_down";
        case common::TensorKind::WUpScale:
            return "w_up_scale";
        case common::TensorKind::WGateScale:
            return "w_gate_scale";
        case common::TensorKind::WDownScale:
            return "w_down_scale";
        default:
            return "unknown";
    }
}

std::uint64_t MakeExpertWorkerKey(int expert_id, int worker_id) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(expert_id)) << 32) |
           static_cast<std::uint32_t>(worker_id);

bool EnqueueResidentBuild(
    ControlState* state,
    int expert_id,
    int worker_id,
    common::GpuVendor vendor,
    bool* out_enqueued,
    std::size_t* out_queue_size) {
    if (out_enqueued != nullptr) {
        *out_enqueued = false;
    }
    if (out_queue_size != nullptr) {
        *out_queue_size = 0;
    }

    if (state == nullptr) return false;
    if (expert_id < 0 || worker_id < 0) return false;
    if (vendor == common::GpuVendor::Unknown) return false;

    const std::uint64_t key = MakeExpertWorkerKey(expert_id, worker_id);

    {
        std::lock_guard<std::mutex> lock(state->resident_build.mu);

        if (state->resident_build.pending_build_keys.find(key) !=
            state->resident_build.pending_build_keys.end()) {
            if (out_queue_size != nullptr) {
                *out_queue_size = state->resident_build.pending_builds.size();
            }
            return true;
        }

        state->resident_build.pending_builds.push_back(PendingResidentBuild{
            .expert_id = expert_id,
            .worker_id = worker_id,
            .vendor = vendor,
        });
        state->resident_build.pending_build_keys.insert(key);

        if (out_enqueued != nullptr) {
            *out_enqueued = true;
        }
        if (out_queue_size != nullptr) {
            *out_queue_size = state->resident_build.pending_builds.size();
        }
    }

    state->resident_build.cv.notify_one();
    return true;
}

void RunResidentBuildWorker(ControlState* state) {
    if (state == nullptr) return;

    for (;;) {
        PendingResidentBuild task;
        bool have_task = false;
        std::size_t queue_size_after_pop = 0;

        {
            std::unique_lock<std::mutex> lock(state->resident_build.mu);
            state->resident_build.cv.wait(lock, [&] {
                return state->resident_build.stop_worker ||
                       !state->resident_build.pending_builds.empty();
            });

            if (state->resident_build.stop_worker &&
                state->resident_build.pending_builds.empty()) {
                return;
            }

            task = state->resident_build.pending_builds.front();
            state->resident_build.pending_builds.erase(
                state->resident_build.pending_builds.begin());
            queue_size_after_pop = state->resident_build.pending_builds.size();
            have_task = true;
        }

        if (!have_task) {
            continue;
        }

        const auto t0 = std::chrono::steady_clock::now();

        bool update_ok = false;
        bool cleared = false;
        bool incoming_ready_before = false;
        bool incoming_ready_after = false;
        bool resident_ready_after = false;

        {
            std::unique_lock<std::shared_mutex> lock(state->mu);

            const expert_node_v2::ExpertEntryV2* entry =
                state->registry.FindEntry(task.expert_id);
            if (entry == nullptr) {
                std::fprintf(stderr,
                             "[%s] background build missing entry "
                             "expert=%d worker=%d vendor=%u(%s)\n",
                             state->static_info.node_id.c_str(),
                             task.expert_id,
                             task.worker_id,
                             static_cast<unsigned>(task.vendor),
                             common::gpu_vendor_name(task.vendor));
            } else {
                incoming_ready_before = entry->incoming_ready;

                if (!entry->incoming_ready) {
                    std::fprintf(stderr,
                                 "[%s] background build skipped: incoming not ready "
                                 "expert=%d worker=%d vendor=%u(%s)\n",
                                 state->static_info.node_id.c_str(),
                                 task.expert_id,
                                 task.worker_id,
                                 static_cast<unsigned>(task.vendor),
                                 common::gpu_vendor_name(task.vendor));
                } else {
                    update_ok = state->registry.Update(
                        task.expert_id,
                        task.worker_id,
                        task.vendor,
                        state->static_info.vendor_spans);

                    if (!update_ok) {
                        std::fprintf(stderr,
                                     "[%s] background Update failed "
                                     "expert=%d worker=%d vendor=%u(%s)\n",
                                     state->static_info.node_id.c_str(),
                                     task.expert_id,
                                     task.worker_id,
                                     static_cast<unsigned>(task.vendor),
                                     common::gpu_vendor_name(task.vendor));
                    } else {
                        cleared = state->registry.ClearIncoming(task.expert_id);
                        if (!cleared) {
                            std::fprintf(stderr,
                                         "[%s] background ClearIncoming failed "
                                         "expert=%d worker=%d vendor=%u(%s)\n",
                                         state->static_info.node_id.c_str(),
                                         task.expert_id,
                                         task.worker_id,
                                         static_cast<unsigned>(task.vendor),
                                         common::gpu_vendor_name(task.vendor));
                        }
                    }

                    entry = state->registry.FindEntry(task.expert_id);
                    if (entry != nullptr) {
                        incoming_ready_after = entry->incoming_ready;
                    }

                    const ExpertDeviceStorageV2* storage =
                        state->registry.FindDeviceStorage(
                            task.expert_id, task.worker_id);
                    resident_ready_after = (storage != nullptr);
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(state->resident_build.mu);
            state->resident_build.pending_build_keys.erase(
                MakeExpertWorkerKey(task.expert_id, task.worker_id));
        }

        const auto t1 = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::printf("[%s] background resident build "
                    "expert=%d worker=%d vendor=%u(%s) "
                    "incoming_ready_before=%d update_ok=%d cleared=%d "
                    "incoming_ready_after=%d resident_ready_after=%d "
                    "queue_size_after_pop=%zu ms=%.3f\n",
                    state->static_info.node_id.c_str(),
                    task.expert_id,
                    task.worker_id,
                    static_cast<unsigned>(task.vendor),
                    common::gpu_vendor_name(task.vendor),
                    static_cast<int>(incoming_ready_before),
                    static_cast<int>(update_ok),
                    static_cast<int>(cleared),
                    static_cast<int>(incoming_ready_after),
                    static_cast<int>(resident_ready_after),
                    queue_size_after_pop,
                    ms);
    }
}

bool SendEmptyAck(
    int fd,
    common::MsgType ack_type,
    std::uint32_t request_id) {
    common::MsgHeader resp{};
    resp.magic = common::kMagic;
    resp.version = common::kVersion;
    resp.msg_type = static_cast<std::uint16_t>(ack_type);
    resp.request_id = request_id;
    resp.body_len = 0;
    return common::SendMessage(fd, resp, std::string());
}

common::PlacementAck BuildPlacementAck(
    const ControlState& state,
    const common::PlacementPlan& plan) {
    common::PlacementAck ack{};
    ack.status_code = 0;

    for (const auto& a : plan.assignments) {
        ++ack.num_target_experts;

        if (state.registry.FindDeviceStorage(
                static_cast<int>(a.expert_id),
                static_cast<int>(a.worker_id)) != nullptr) {
            ++ack.num_ready_experts;
        }
    }

    ack.all_ready = (ack.num_ready_experts == ack.num_target_experts);
    ack.needs_load = !ack.all_ready;
    return ack;
}

bool SendPlacementAck(
    int fd,
    std::uint32_t request_id,
    const common::PlacementAck& ack) {
    std::string body;
    if (!common::EncodePlacementAck(ack, &body)) {
        return false;
    }

    common::MsgHeader hdr{};
    hdr.magic = common::kMagic;
    hdr.version = common::kVersion;
    hdr.msg_type = static_cast<std::uint16_t>(common::MsgType::PlacementAck);
    hdr.request_id = request_id;
    hdr.body_len = static_cast<std::uint32_t>(body.size());

    return common::SendMessage(fd, hdr, body);
}

bool SendPlacementAckError(
    int fd,
    std::uint32_t request_id,
    std::uint32_t status_code = 1) {
    common::PlacementAck ack{};
    ack.status_code = status_code;
    ack.needs_load = true;
    ack.all_ready = false;
    ack.num_target_experts = 0;
    ack.num_ready_experts = 0;
    return SendPlacementAck(fd, request_id, ack);
}

bool HandleInventoryRequest(
    int fd,
    ControlState* state,
    const common::MsgHeader& req,
    const std::string& req_body) {
    if (state == nullptr) {
        std::fprintf(stderr, "HandleInventoryRequest: state is null\n");
        return false;
    }

    if (!req_body.empty()) {
        std::fprintf(stderr, "InventoryRequest body must be empty\n");
        return false;
    }

    common::DynamicNodeInfo dynamic_info;
    if (!BuildDynamicNodeInfo(
            state->static_info,
            state->node_status,
            &dynamic_info)) {
        std::fprintf(stderr, "[%s] BuildDynamicNodeInfo failed\n",
                     state->static_info.node_id.c_str());
        return false;
    }

    std::printf("[%s] received InventoryRequest rid=%u\n",
                state->static_info.node_id.c_str(), req.request_id);

    std::string body;
    if (!common::EncodeInventoryReplyBody(
            state->static_info,
            dynamic_info,
            &body)) {
        std::fprintf(stderr, "[%s] EncodeInventoryReplyBody failed\n",
                     state->static_info.node_id.c_str());
        return false;
    }

    common::MsgHeader resp{};
    resp.magic = common::kMagic;
    resp.version = common::kVersion;
    resp.msg_type = static_cast<std::uint16_t>(common::MsgType::InventoryReply);
    resp.request_id = req.request_id;
    resp.body_len = static_cast<std::uint32_t>(body.size());

    bool ok = common::SendMessage(fd, resp, body);
    if (ok) {
        std::printf("[%s] sent InventoryReply rid=%u body_len=%u\n",
                    state->static_info.node_id.c_str(),
                    resp.request_id,
                    resp.body_len);
    }
    return ok;
}

bool HandleResidentInventoryRequest(
    int fd,
    ControlState* state,
    const common::MsgHeader& req,
    const std::string& req_body) {
    if (state == nullptr) {
        std::fprintf(stderr, "HandleResidentInventoryRequest: state is null\n");
        return false;
    }

    if (!req_body.empty()) {
        std::fprintf(stderr, "ResidentInventoryRequest body must be empty\n");
        return false;
    }

    std::printf("[%s] received ResidentInventoryRequest rid=%u\n",
                state->static_info.node_id.c_str(), req.request_id);

    std::vector<common::ResidentInventoryWorkerInfo> workers =
        state->registry.BuildResidentInventory(state->static_info);

    std::size_t total_resident_experts = 0;
    for (const auto& worker : workers) {
        total_resident_experts += worker.expert_ids.size();
    }

    std::printf("[%s] ResidentInventoryRequest rid=%u workers=%zu total_resident_experts=%zu\n",
                state->static_info.node_id.c_str(),
                req.request_id,
                workers.size(),
                total_resident_experts);

    std::string body;
    if (!common::EncodeResidentInventoryReplyBody(workers, &body)) {
        std::fprintf(stderr, "[%s] EncodeResidentInventoryReplyBody failed\n",
                     state->static_info.node_id.c_str());
        return false;
    }

    common::MsgHeader resp{};
    resp.magic = common::kMagic;
    resp.version = common::kVersion;
    resp.msg_type =
        static_cast<std::uint16_t>(common::MsgType::ResidentInventoryReply);
    resp.request_id = req.request_id;
    resp.body_len = static_cast<std::uint32_t>(body.size());

    const bool ok = common::SendMessage(fd, resp, body);
    if (ok) {
        std::printf("[%s] sent ResidentInventoryReply rid=%u body_len=%u workers=%zu total_resident_experts=%zu\n",
                    state->static_info.node_id.c_str(),
                    resp.request_id,
                    resp.body_len,
                    workers.size(),
                    total_resident_experts);
    }
    return ok;
}

bool HandleHeartbeatRequest(
    int fd,
    ControlState* state,
    const common::MsgHeader& req,
    const std::string& req_body) {
    if (state == nullptr) {
        std::fprintf(stderr, "HandleHeartbeatRequest: state is null\n");
        return false;
    }

    if (!req_body.empty()) {
        std::fprintf(stderr, "HeartbeatRequest body must be empty\n");
        return false;
    }

    std::printf("[%s] received HeartbeatRequest rid=%u\n",
                state->static_info.node_id.c_str(),
                req.request_id);

    const bool ok =
        SendEmptyAck(fd, common::MsgType::HeartbeatReply, req.request_id);
    if (ok) {
        std::printf("[%s] sent HeartbeatReply rid=%u\n",
                    state->static_info.node_id.c_str(),
                    req.request_id);
    }
    return ok;
}

bool HandlePlacementPlan(
    int fd,
    ControlState* state,
    const common::MsgHeader& req,
    const std::string& req_body) {
    if (state == nullptr) {
        std::fprintf(stderr, "HandlePlacementPlan: state is null\n");
        return false;
    }

    common::PlacementPlan plan;
    if (!common::DecodePlacementPlanBody(req_body, &plan)) {
        std::fprintf(stderr,
                     "[%s] failed to decode PlacementPlan\n",
                     state->static_info.node_id.c_str());
        return SendPlacementAckError(fd, req.request_id);
    }

    for (const auto& a : plan.assignments) {
        if (a.expert_id < 0) {
            std::fprintf(stderr,
                         "[%s] invalid expert_id=%d in PlacementPlan\n",
                         state->static_info.node_id.c_str(),
                         a.expert_id);
            return SendPlacementAckError(fd, req.request_id);
        }

        if (a.worker_id < 0 ||
            a.worker_id >= static_cast<std::int32_t>(state->static_info.gpus.size())) {
            std::fprintf(stderr,
                         "[%s] invalid worker_id=%d for expert=%d\n",
                         state->static_info.node_id.c_str(),
                         a.worker_id,
                         a.expert_id);
            return SendPlacementAckError(fd, req.request_id);
        }
    }

    std::printf("[%s] received PlacementPlan rid=%u assignments=%zu drop_non_target_residents=%d\n",
                state->static_info.node_id.c_str(),
                req.request_id,
                plan.assignments.size(),
                static_cast<int>(plan.drop_non_target_residents));

    common::PlacementAck ack{};
    std::size_t num_dropped = 0;
    bool had_active_load = false;
    bool cleared_active_load = false;
    {
        std::unique_lock<std::shared_mutex> lock(state->mu);

        had_active_load = state->active_load.active;

        if (plan.drop_non_target_residents) {
            num_dropped = state->registry.DropNonTargetResidents(plan.assignments);
        }

        ack = BuildPlacementAck(*state, plan);

        // Keep resident experts unless explicitly dropped above.
        // Always clear partial in-flight upload state.
        state->active_load = ActiveLoad{};
        cleared_active_load = had_active_load;
    }

    const bool ok = SendPlacementAck(fd, req.request_id, ack);
    if (ok) {
	std::printf("[%s] sent PlacementAck rid=%u drop_non_target_residents=%d dropped=%zu "
                    "had_active_load=%d cleared_active_load=%d "
                    "needs_load=%d all_ready=%d target=%u ready=%u\n",
                    state->static_info.node_id.c_str(),
                    req.request_id,
                    static_cast<int>(plan.drop_non_target_residents),
                    num_dropped,
                    static_cast<int>(had_active_load),
                    static_cast<int>(cleared_active_load),
                    static_cast<int>(ack.needs_load),
                    static_cast<int>(ack.all_ready),
                    ack.num_target_experts,
                    ack.num_ready_experts);
    }
    return ok;
}

bool HandleLoadWeightsBegin(
    int fd,
    ControlState* state,
    const common::MsgHeader& req,
    const std::string& req_body) {
    if (state == nullptr) {
        std::fprintf(stderr, "HandleLoadWeightsBegin: state is null\n");
        return false;
    }

    common::LoadWeightsBeginMsg msg;
    if (!common::DecodeLoadWeightsBeginBody(req_body, &msg)) {
        std::fprintf(stderr, "[%s] failed to decode LoadWeightsBegin\n",
                     state->static_info.node_id.c_str());
        return false;
    }

    if (msg.expert_id < 0) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsBegin invalid expert_id=%d\n",
                     state->static_info.node_id.c_str(),
                     msg.expert_id);
        return false;
    }

    if (msg.worker_id < 0) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsBegin invalid worker_id=%d for expert=%d\n",
                     state->static_info.node_id.c_str(),
                     msg.worker_id,
                     msg.expert_id);
        return false;
    }

    if (state->active_load.active) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsBegin while another load is active: "
                     "active expert=%d tensor_kind=%s total=%llu received=%llu\n",
                     state->static_info.node_id.c_str(),
                     state->active_load.expert_id,
                     TensorKindName(state->active_load.tensor_kind),
                     static_cast<unsigned long long>(state->active_load.total_bytes),
                     static_cast<unsigned long long>(state->active_load.received_bytes));
        return false;
    }

    state->active_load.active = true;
    state->active_load.expert_id = msg.expert_id;
    state->active_load.worker_id = msg.worker_id;
    state->active_load.tensor_kind = msg.tensor_kind;
    state->active_load.total_bytes = msg.total_bytes;
    state->active_load.received_bytes = 0;
    state->active_load.buffer.clear();
    state->active_load.buffer.reserve(static_cast<std::size_t>(msg.total_bytes));
    state->active_load.meta = msg.meta;

    std::printf("[%s] received LoadWeightsBegin rid=%u "
                "expert=%d worker_id=%d tensor_kind=%s total_bytes=%llu "
                "shape=[%llu,%llu] dtype=%s\n",
                state->static_info.node_id.c_str(),
                req.request_id,
                msg.expert_id,
                msg.worker_id,
                TensorKindName(msg.tensor_kind),
                static_cast<unsigned long long>(msg.total_bytes),
                msg.meta.shape.size() > 0 ? static_cast<unsigned long long>(msg.meta.shape[0]) : 0ULL,
                msg.meta.shape.size() > 1 ? static_cast<unsigned long long>(msg.meta.shape[1]) : 0ULL,
                msg.meta.dtype.c_str());

    const bool ok =
        SendEmptyAck(fd, common::MsgType::LoadWeightsAck, req.request_id);
    if (ok) {
        std::printf("[%s] sent LoadWeightsAck rid=%u\n",
                    state->static_info.node_id.c_str(),
                    req.request_id);
    }
    return ok;
}

bool HandleLoadWeightsChunk(
    int fd,
    ControlState* state,
    const common::MsgHeader& req,
    const std::string& req_body) {
    (void)fd;

    if (state == nullptr) {
        std::fprintf(stderr, "HandleLoadWeightsChunk: state is null\n");
        return false;
    }

    common::LoadWeightsChunkMsg msg;
    if (!common::DecodeLoadWeightsChunkBody(req_body, &msg)) {
        std::fprintf(stderr, "[%s] failed to decode LoadWeightsChunk\n",
                     state->static_info.node_id.c_str());
        return false;
    }

    if (!state->active_load.active) {
        std::fprintf(stderr, "[%s] LoadWeightsChunk with no active load\n",
                     state->static_info.node_id.c_str());
        return false;
    }

    if (state->active_load.expert_id != msg.expert_id) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsChunk expert mismatch: got=%d expected=%d\n",
                     state->static_info.node_id.c_str(),
                     msg.expert_id,
                     state->active_load.expert_id);
        return false;
    }

    if (state->active_load.worker_id != msg.worker_id) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsChunk worker mismatch: got=%d expected=%d\n",
                     state->static_info.node_id.c_str(),
                     msg.worker_id,
                     state->active_load.worker_id);
        return false;
    }

    if (state->active_load.tensor_kind != msg.tensor_kind) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsChunk tensor_kind mismatch: got=%s expected=%s\n",
                     state->static_info.node_id.c_str(),
                     TensorKindName(msg.tensor_kind),
                     TensorKindName(state->active_load.tensor_kind));
        return false;
    }

    if (msg.chunk_offset != state->active_load.received_bytes) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsChunk offset mismatch: got=%llu expected=%llu\n",
                     state->static_info.node_id.c_str(),
                     static_cast<unsigned long long>(msg.chunk_offset),
                     static_cast<unsigned long long>(state->active_load.received_bytes));
        return false;
    }

    const std::uint64_t new_received =
        state->active_load.received_bytes +
        static_cast<std::uint64_t>(msg.chunk_data.size());
    if (new_received > state->active_load.total_bytes) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsChunk overflow: chunk would make received=%llu > total=%llu\n",
                     state->static_info.node_id.c_str(),
                     static_cast<unsigned long long>(new_received),
                     static_cast<unsigned long long>(state->active_load.total_bytes));
        return false;
    }

    state->active_load.buffer.insert(
        state->active_load.buffer.end(),
        msg.chunk_data.begin(),
        msg.chunk_data.end());
    state->active_load.received_bytes =
        static_cast<std::uint64_t>(state->active_load.buffer.size());

    std::printf("[%s] received LoadWeightsChunk rid=%u "
                "expert=%d worker_id=%d tensor_kind=%s chunk_offset=%llu chunk_size=%zu "
                "received=%llu/%llu\n",
                state->static_info.node_id.c_str(),
                req.request_id,
                msg.expert_id,
                msg.worker_id,
                TensorKindName(msg.tensor_kind),
                static_cast<unsigned long long>(msg.chunk_offset),
                msg.chunk_data.size(),
                static_cast<unsigned long long>(state->active_load.received_bytes),
                static_cast<unsigned long long>(state->active_load.total_bytes));

    return true;
}

bool HandleLoadWeightsEnd(
    int fd,
    ControlState* state,
    const common::MsgHeader& req,
    const std::string& req_body) {
    if (state == nullptr) {
        std::fprintf(stderr, "HandleLoadWeightsEnd: state is null\n");
        return false;
    }

    common::LoadWeightsEndMsg msg;
    if (!common::DecodeLoadWeightsEndBody(req_body, &msg)) {
        std::fprintf(stderr, "[%s] failed to decode LoadWeightsEnd\n",
                     state->static_info.node_id.c_str());
        return false;
    }

    std::uint64_t total_bytes = 0;
    std::size_t final_buffer_size = 0;
    common::GpuVendor vendor = common::GpuVendor::Unknown;
    bool incoming_ready_before_enqueue = false;
    bool incoming_ready_after_enqueue = false;
    bool incoming_cleared_before_ack = false;
    bool resident_ready_before_ack = false;
    bool enqueued_background_build = false;

    {
        std::unique_lock<std::shared_mutex> lock(state->mu);

        if (!state->active_load.active) {
            std::fprintf(stderr, "[%s] LoadWeightsEnd with no active load\n",
                         state->static_info.node_id.c_str());
            return false;
        }

        if (state->active_load.expert_id != msg.expert_id) {
            std::fprintf(stderr,
                         "[%s] LoadWeightsEnd expert mismatch: got=%d expected=%d\n",
                         state->static_info.node_id.c_str(),
                         msg.expert_id,
                         state->active_load.expert_id);
            return false;
        }

        if (state->active_load.worker_id != msg.worker_id) {
            std::fprintf(stderr,
                         "[%s] LoadWeightsEnd worker mismatch: got=%d expected=%d\n",
                         state->static_info.node_id.c_str(),
                         msg.worker_id,
                         state->active_load.worker_id);
            return false;
        }

        if (state->active_load.tensor_kind != msg.tensor_kind) {
            std::fprintf(stderr,
                         "[%s] LoadWeightsEnd tensor_kind mismatch: got=%s expected=%s\n",
                         state->static_info.node_id.c_str(),
                         TensorKindName(msg.tensor_kind),
                         TensorKindName(state->active_load.tensor_kind));
            return false;
        }

        if (state->active_load.received_bytes != state->active_load.total_bytes) {
            std::fprintf(stderr,
                         "[%s] LoadWeightsEnd byte mismatch: received=%llu expected=%llu\n",
                         state->static_info.node_id.c_str(),
                         static_cast<unsigned long long>(state->active_load.received_bytes),
                         static_cast<unsigned long long>(state->active_load.total_bytes));
            return false;
        }

        if (state->active_load.buffer.size() !=
            static_cast<std::size_t>(state->active_load.total_bytes)) {
            std::fprintf(stderr,
                         "[%s] LoadWeightsEnd buffer size mismatch: buffer=%zu expected=%llu\n",
                         state->static_info.node_id.c_str(),
                         state->active_load.buffer.size(),
                         static_cast<unsigned long long>(state->active_load.total_bytes));
            return false;
        }

        if (msg.worker_id < 0 ||
            static_cast<std::size_t>(msg.worker_id) >= state->static_info.gpus.size()) {
            std::fprintf(stderr,
                         "[%s] LoadWeightsEnd invalid worker_id=%d, gpus.size()=%zu\n",
                         state->static_info.node_id.c_str(),
                         msg.worker_id,
                         state->static_info.gpus.size());
            return false;
        }

        const common::StaticGpuInfo& gpu =
            state->static_info.gpus[static_cast<std::size_t>(msg.worker_id)];
        vendor = gpu.gpu_vendor;

        total_bytes = state->active_load.total_bytes;
        final_buffer_size = state->active_load.buffer.size();

        if (!state->registry.StoreIncomingTensor(
                msg.expert_id,
                msg.tensor_kind,
                total_bytes,
                std::move(state->active_load.buffer),
                std::move(state->active_load.meta))) {
            std::fprintf(stderr,
                         "[%s] failed to store incoming tensor for expert=%d tensor_kind=%s\n",
                         state->static_info.node_id.c_str(),
                         msg.expert_id,
                         TensorKindName(msg.tensor_kind));
            return false;
        }

        const expert_node_v2::ExpertEntryV2* entry =
            state->registry.FindEntry(msg.expert_id);
        if (entry == nullptr) {
            std::fprintf(stderr,
                         "[%s] missing entry after StoreIncomingTensor for expert=%d\n",
                         state->static_info.node_id.c_str(),
                         msg.expert_id);
            return false;
        }

        incoming_ready_before_enqueue = entry->incoming_ready;

	std::size_t pending_build_queue_size = 0;

        if (entry->incoming_ready) {
            if (!EnqueueResidentBuild(
                    state,
                    msg.expert_id,
                    msg.worker_id,
                    vendor,
                    &enqueued_background_build,
                    &pending_build_queue_size)) {
                std::fprintf(stderr,
                             "[%s] failed to enqueue resident build for expert=%d worker=%d vendor=%u(%s)\n",
                             state->static_info.node_id.c_str(),
                             msg.expert_id,
                             msg.worker_id,
                             static_cast<unsigned>(vendor),
                             common::gpu_vendor_name(vendor));
                return false;
            }
        }

        const ExpertDeviceStorageV2* storage =
            state->registry.FindDeviceStorage(msg.expert_id, msg.worker_id);

        entry = state->registry.FindEntry(msg.expert_id);
        if (entry == nullptr) {
            std::fprintf(stderr,
                         "[%s] missing entry before ack for expert=%d\n",
                         state->static_info.node_id.c_str(),
                         msg.expert_id);
            return false;
        }

        incoming_ready_after_enqueue = entry->incoming_ready;
        incoming_cleared_before_ack = false;
        resident_ready_before_ack = (storage != nullptr);

        state->registry.DebugPrint();
        state->active_load = ActiveLoad{};
    }

    std::printf("[%s] received LoadWeightsEnd rid=%u "
                "expert=%d worker_id=%d vendor=%u tensor_kind=%s total_bytes=%llu buffer_size=%zu "
                "incoming_ready_before_enqueue=%d incoming_ready_after_enqueue=%d "
                "incoming_cleared_before_ack=%d resident_ready_before_ack=%d "
		"enqueued_background_build=%d pending_build_queue_size=%zu\n",
                state->static_info.node_id.c_str(),
                req.request_id,
                msg.expert_id,
                msg.worker_id,
                static_cast<unsigned>(vendor),
                TensorKindName(msg.tensor_kind),
                static_cast<unsigned long long>(total_bytes),
                final_buffer_size,
                static_cast<int>(incoming_ready_before_enqueue),
                static_cast<int>(incoming_ready_after_enqueue),
                static_cast<int>(incoming_cleared_before_ack),
                static_cast<int>(resident_ready_before_ack),
                static_cast<int>(enqueued_background_build),
                pending_build_queue_size);

    const bool ok =
        SendEmptyAck(fd, common::MsgType::LoadWeightsAck, req.request_id);
    if (ok) {
        std::printf("[%s] sent LoadWeightsAck rid=%u\n",
                    state->static_info.node_id.c_str(),
                    req.request_id);
    }
    return ok;
}

bool DispatchRequest(
    int fd,
    ControlState* state,
    const common::MsgHeader& req,
    const std::string& req_body) {
    if (state == nullptr) {
        std::fprintf(stderr, "DispatchRequest: state is null\n");
        return false;
    }

    auto msg_type = static_cast<common::MsgType>(req.msg_type);

    const bool profile_load_path =
        msg_type == common::MsgType::LoadWeightsBegin ||
        msg_type == common::MsgType::LoadWeightsChunk ||
        msg_type == common::MsgType::LoadWeightsEnd;

    const auto t0 = profile_load_path ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};

    bool ok = false;
    switch (msg_type) {
        case common::MsgType::InventoryRequest:
            ok = HandleInventoryRequest(fd, state, req, req_body);
            break;
        case common::MsgType::ResidentInventoryRequest:
            ok = HandleResidentInventoryRequest(fd, state, req, req_body);
            break;
        case common::MsgType::HeartbeatRequest:
            ok = HandleHeartbeatRequest(fd, state, req, req_body);
            break;
        case common::MsgType::PlacementPlan:
            ok = HandlePlacementPlan(fd, state, req, req_body);
            break;
        case common::MsgType::LoadWeightsBegin:
            ok = HandleLoadWeightsBegin(fd, state, req, req_body);
            break;
        case common::MsgType::LoadWeightsChunk:
            ok = HandleLoadWeightsChunk(fd, state, req, req_body);
            break;
        case common::MsgType::LoadWeightsEnd:
            ok = HandleLoadWeightsEnd(fd, state, req, req_body);
            break;
        case common::MsgType::InferRequest:
            std::fprintf(stderr,
                         "[%s] InferRequest should not be handled on control port\n",
                         state->static_info.node_id.c_str());
            ok = false;
            break;
        default:
            std::fprintf(stderr, "[%s] unsupported msg_type: %u\n",
                         state->static_info.node_id.c_str(),
                         req.msg_type);
            ok = false;
            break;
    }

    if (profile_load_path) {
        const auto t1 = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::fprintf(stderr,
                     "[dispatch_profile] node=%s msg_type=%u rid=%u body_len=%u ok=%d ms=%.3f\n",
                     state->static_info.node_id.c_str(),
                     static_cast<unsigned>(req.msg_type),
                     req.request_id,
                     req.body_len,
                     static_cast<int>(ok),
                     ms);
    }

    return ok;
}

bool HandleOneRequest(
    int fd,
    ControlState* state) {
    std::uint8_t hdr_buf[16];
    if (!common::RecvAll(fd, hdr_buf, sizeof(hdr_buf))) {
        return false;
    }

    common::MsgHeader req{};
    if (!common::DecodeHeader(hdr_buf, sizeof(hdr_buf), &req)) {
        std::fprintf(stderr, "failed to decode request header\n");
        return false;
    }

    if (req.magic != common::kMagic) {
        std::fprintf(stderr, "bad magic: 0x%x\n", req.magic);
        return false;
    }
    if (req.version != common::kVersion) {
        std::fprintf(stderr, "bad version: %u\n", req.version);
        return false;
    }

    std::string req_body;
    req_body.resize(req.body_len);
    if (req.body_len > 0) {
        if (!common::RecvAll(fd, req_body.data(), req.body_len)) {
            std::fprintf(stderr, "failed to read request body (%u bytes)\n", req.body_len);
            return false;
        }
    }

    return DispatchRequest(fd, state, req, req_body);
}

}  // namespace

void RunControlLoop(ControlState* state) {
    if (state == nullptr) {
        std::fprintf(stderr, "[control] RunControlLoop: state is null\n");
        return;
    }

    state->resident_build.stop_worker = false;
    state->resident_build.worker = std::thread(RunResidentBuildWorker, state);

    int listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        std::perror("socket");
        {
            std::lock_guard<std::mutex> lock(state->resident_build.mu);
            state->resident_build.stop_worker = true;
        }
        state->resident_build.cv.notify_all();
        if (state->resident_build.worker.joinable()) {
            state->resident_build.worker.join();
        }
        return;
    }

    int opt = 1;
    if (::setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) != 0) {
        std::perror("setsockopt(SO_REUSEADDR)");
        ::close(listen_fd);
        {
            std::lock_guard<std::mutex> lock(state->resident_build.mu);
            state->resident_build.stop_worker = true;
        }
        state->resident_build.cv.notify_all();
        if (state->resident_build.worker.joinable()) {
            state->resident_build.worker.join();
        }
        return;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<std::uint16_t>(state->static_info.control_port));

    if (::bind(listen_fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::perror("bind");
        ::close(listen_fd);
        {
            std::lock_guard<std::mutex> lock(state->resident_build.mu);
            state->resident_build.stop_worker = true;
        }
        state->resident_build.cv.notify_all();
        if (state->resident_build.worker.joinable()) {
            state->resident_build.worker.join();
        }
        return;
    }

    if (::listen(listen_fd, 128) != 0) {
        std::perror("listen");
        ::close(listen_fd);
        {
            std::lock_guard<std::mutex> lock(state->resident_build.mu);
            state->resident_build.stop_worker = true;
        }
        state->resident_build.cv.notify_all();
        if (state->resident_build.worker.joinable()) {
            state->resident_build.worker.join();
        }
        return;
    }

    std::printf("[%s] control loop started on port=%d\n",
                state->static_info.node_id.c_str(),
                state->static_info.control_port);

    for (;;) {
        int fd = ::accept(listen_fd, nullptr, nullptr);
        if (fd < 0) {
            std::perror("accept");
            continue;
        }

        std::printf("[%s] control accepted client fd=%d\n",
                    state->static_info.node_id.c_str(),
                    fd);

        while (true) {
            if (!HandleOneRequest(fd, state)) {
                break;
            }
        }

        ::close(fd);
    }

    ::close(listen_fd);

    {
        std::lock_guard<std::mutex> lock(state->resident_build.mu);
        state->resident_build.stop_worker = true;
    }
    state->resident_build.cv.notify_all();
    if (state->resident_build.worker.joinable()) {
        state->resident_build.worker.join();
    }
}
