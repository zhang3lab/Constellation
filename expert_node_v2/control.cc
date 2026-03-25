#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "common/header_codec.h"
#include "common/infer_codec.h"
#include "common/inventory_codec.h"
#include "common/placement_codec.h"
#include "common/protocol.h"
#include "common/socket_utils.h"
#include "common/types.h"
#include "common/weight_codec.h"
#include "expert_node_v2/cuda/gpu_info_cuda_v2.h"
#include "expert_node_v2/expert_backend_v2.h"
#include "expert_node_v2/expert_registry_v2.h"

namespace {

struct ActiveLoad {
    bool active = false;
    int expert_id = -1;
    common::TensorKind tensor_kind = common::TensorKind::WUp;
    std::uint64_t total_bytes = 0;
    std::uint64_t received_bytes = 0;
    std::string buffer;
};

struct ControlState {
    common::NodeStatus node_status = common::NodeStatus::Booting;

    std::string node_id = "node0";
    int control_port = 0;
    int worker_port_base = 0;

    ActiveLoad active_load;
    expert_node_v2::ExpertRegistryV2 registry;
};

bool StoreIncomingTensorToEntry(
    expert_node_v2::ExpertEntryV2* entry,
    common::TensorKind tensor_kind,
    std::uint64_t total_bytes,
    std::string&& bytes) {
    if (entry == nullptr) return false;

    expert_node_v2::HostTensorV2* slot = nullptr;

    switch (tensor_kind) {
        case common::TensorKind::WUp:
            slot = &entry->incoming.w_up;
            break;
        case common::TensorKind::WUpScale:
            slot = &entry->incoming.w_up_scale;
            break;
        case common::TensorKind::WGate:
            slot = &entry->incoming.w_gate;
            break;
        case common::TensorKind::WGateScale:
            slot = &entry->incoming.w_gate_scale;
            break;
        case common::TensorKind::WDown:
            slot = &entry->incoming.w_down;
            break;
        case common::TensorKind::WDownScale:
            slot = &entry->incoming.w_down_scale;
            break;
        default:
            return false;
    }

    slot->bytes = std::move(bytes);
    slot->ready = true;

    // 暂时只填 bytes/ready。shape/dtype 后面如果协议里没有，
    // 就在 Upload 前按固定 DeepSeek expert 形状补。
    return static_cast<std::uint64_t>(slot->bytes.size()) == total_bytes;
}

void FillFixedTensorMetaForEntry(expert_node_v2::ExpertEntryV2* entry) {
    if (entry == nullptr) return;

    entry->incoming.w_up.shape = {2048, 7168};
    entry->incoming.w_up.dtype = "torch.float8_e4m3fn";

    entry->incoming.w_gate.shape = {2048, 7168};
    entry->incoming.w_gate.dtype = "torch.float8_e4m3fn";

    entry->incoming.w_down.shape = {7168, 2048};
    entry->incoming.w_down.dtype = "torch.float8_e4m3fn";

    entry->incoming.w_up_scale.shape = {16, 56};
    entry->incoming.w_up_scale.dtype = "torch.float32";

    entry->incoming.w_gate_scale.shape = {16, 56};
    entry->incoming.w_gate_scale.dtype = "torch.float32";

    entry->incoming.w_down_scale.shape = {56, 16};
    entry->incoming.w_down_scale.dtype = "torch.float32";
}

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

common::NodeInfo BuildRealInventory(
    const std::string& node_id,
    int control_port,
    int worker_port_base) {
    common::NodeInfo node;
    node.node_id = "node0";
    node.host = "127.0.0.1";
    node.control_port = control_port;

    if (!BuildLocalCudaGpuInfosV2(
            static_cast<std::uint32_t>(worker_port_base),
            &node.gpus)) {
        std::fprintf(stderr, "BuildLocalCudaGpuInfosV2 failed\n");
    }

    return node;
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

    common::NodeInfo info =
        BuildRealInventory(state->node_id, state->control_port, state->worker_port_base);

    std::printf("[%s] received InventoryRequest rid=%u\n",
                info.node_id.c_str(), req.request_id);

    std::string body =
        common::EncodeInventoryReplyBody(info, state->node_status);

    common::MsgHeader resp{};
    resp.magic = common::kMagic;
    resp.version = common::kVersion;
    resp.msg_type = static_cast<std::uint16_t>(common::MsgType::InventoryReply);
    resp.request_id = req.request_id;
    resp.body_len = static_cast<std::uint32_t>(body.size());

    bool ok = common::SendMessage(fd, resp, body);
    if (ok) {
        std::printf("[%s] sent InventoryReply rid=%u body_len=%u\n",
                    info.node_id.c_str(), resp.request_id, resp.body_len);
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
                state->node_id.c_str(), req.request_id);

    bool ok = SendEmptyAck(fd, common::MsgType::HeartbeatReply, req.request_id);
    if (ok) {
        std::printf("[%s] sent HeartbeatReply rid=%u\n",
                    state->node_id.c_str(), req.request_id);
    }
    return ok;
}

bool HandlePlacementPlan(
    int fd,
    const common::NodeInfo& info,
    ControlState* state,
    const common::MsgHeader& req,
    const std::string& req_body) {
    if (state == nullptr) {
        std::fprintf(stderr, "[%s] HandlePlacementPlan: state is null\n",
                     info.node_id.c_str());
        return false;
    }

    std::vector<common::PlacementAssignment> assignments;
    if (!common::DecodePlacementPlanBody(req_body, &assignments)) {
        std::fprintf(stderr, "[%s] failed to decode PlacementPlan\n",
                     info.node_id.c_str());
        return false;
    }

    for (const auto& a : assignments) {
        if (a.local_gpu_id < 0 ||
            a.local_gpu_id >= static_cast<std::int32_t>(info.gpus.size())) {
            std::fprintf(stderr,
                         "[%s] invalid local_gpu_id=%d for expert=%d\n",
                         info.node_id.c_str(), a.local_gpu_id, a.expert_id);
            return false;
        }
    }

    std::printf("[%s] received PlacementPlan rid=%u assignments=%zu\n",
                info.node_id.c_str(), req.request_id, assignments.size());

    state->registry.clear();

    for (const auto& a : assignments) {
        auto* entry = state->registry.find_or_create_entry(a.expert_id);
        if (entry == nullptr) {
            std::fprintf(stderr,
                         "[%s] failed to create registry entry for expert=%d\n",
                         info.node_id.c_str(), a.expert_id);
            return false;
        }

        entry->expert_id = a.expert_id;
        entry->local_gpu_id = a.local_gpu_id;
        entry->incoming = expert_node_v2::ExpertTensorBundleV2{};
        entry->incoming_ready = false;
        entry->storage.clear();
        entry->resident_ready = false;
    }

    state->active_load = ActiveLoad{};
    state->registry.debug_print();

    bool ok = SendEmptyAck(fd, common::MsgType::PlacementAck, req.request_id);
    if (ok) {
        std::printf("[%s] sent PlacementAck rid=%u\n",
                    info.node_id.c_str(), req.request_id);
    }
    return ok;
}

bool HandleLoadWeightsBegin(
    int fd,
    const common::NodeInfo& info,
    ControlState* state,
    const common::MsgHeader& req,
    const std::string& req_body) {
    if (state == nullptr) {
        std::fprintf(stderr, "[%s] HandleLoadWeightsBegin: state is null\n",
                     info.node_id.c_str());
        return false;
    }

    common::LoadWeightsBeginMsg msg;
    if (!common::DecodeLoadWeightsBeginBody(req_body, &msg)) {
        std::fprintf(stderr, "[%s] failed to decode LoadWeightsBegin\n",
                     info.node_id.c_str());
        return false;
    }

    auto* entry = state->registry.find_entry(msg.expert_id);
    if (entry == nullptr) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsBegin for unknown expert=%d\n",
                     info.node_id.c_str(), msg.expert_id);
        return false;
    }

    if (entry->local_gpu_id != msg.local_gpu_id) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsBegin gpu mismatch for expert=%d: "
                     "got local_gpu_id=%d expected=%d\n",
                     info.node_id.c_str(),
                     msg.expert_id,
                     msg.local_gpu_id,
                     entry->local_gpu_id);
        return false;
    }

    if (state->active_load.active) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsBegin while another load is active\n",
                     info.node_id.c_str());
        return false;
    }

    state->active_load.active = true;
    state->active_load.expert_id = msg.expert_id;
    state->active_load.tensor_kind = msg.tensor_kind;
    state->active_load.total_bytes = msg.total_bytes;
    state->active_load.received_bytes = 0;
    state->active_load.buffer.clear();
    state->active_load.buffer.reserve(static_cast<std::size_t>(msg.total_bytes));

    std::printf("[%s] received LoadWeightsBegin rid=%u "
                "expert=%d local_gpu_id=%d tensor_kind=%s total_bytes=%llu\n",
                info.node_id.c_str(),
                req.request_id,
                msg.expert_id,
                msg.local_gpu_id,
                TensorKindName(msg.tensor_kind),
                static_cast<unsigned long long>(msg.total_bytes));

    std::printf("[%s] active_load armed: expert=%d tensor_kind=%s total=%llu\n",
                info.node_id.c_str(),
                state->active_load.expert_id,
                TensorKindName(state->active_load.tensor_kind),
                static_cast<unsigned long long>(state->active_load.total_bytes));

    bool ok = SendEmptyAck(fd, common::MsgType::LoadWeightsAck, req.request_id);
    if (ok) {
        std::printf("[%s] sent LoadWeightsAck rid=%u\n",
                    info.node_id.c_str(), req.request_id);
    }
    return ok;
}

bool HandleLoadWeightsChunk(
    int fd,
    const common::NodeInfo& info,
    ControlState* state,
    const common::MsgHeader& req,
    const std::string& req_body) {
    if (state == nullptr) {
        std::fprintf(stderr, "[%s] HandleLoadWeightsChunk: state is null\n",
                     info.node_id.c_str());
        return false;
    }

    common::LoadWeightsChunkMsg msg;
    if (!common::DecodeLoadWeightsChunkBody(req_body, &msg)) {
        std::fprintf(stderr, "[%s] failed to decode LoadWeightsChunk\n",
                     info.node_id.c_str());
        return false;
    }

    if (!state->active_load.active) {
        std::fprintf(stderr, "[%s] LoadWeightsChunk with no active load\n",
                     info.node_id.c_str());
        return false;
    }

    auto* entry = state->registry.find_entry(msg.expert_id);
    if (entry == nullptr) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsChunk for unknown expert=%d\n",
                     info.node_id.c_str(), msg.expert_id);
        return false;
    }

    if (entry->local_gpu_id != msg.local_gpu_id) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsChunk gpu mismatch for expert=%d: "
                     "got local_gpu_id=%d expected=%d\n",
                     info.node_id.c_str(),
                     msg.expert_id,
                     msg.local_gpu_id,
                     entry->local_gpu_id);
        return false;
    }

    if (state->active_load.expert_id != msg.expert_id) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsChunk expert mismatch: got=%d expected=%d\n",
                     info.node_id.c_str(),
                     msg.expert_id,
                     state->active_load.expert_id);
        return false;
    }

    if (state->active_load.tensor_kind != msg.tensor_kind) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsChunk tensor_kind mismatch\n",
                     info.node_id.c_str());
        return false;
    }

    if (msg.chunk_offset != state->active_load.received_bytes) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsChunk offset mismatch: got=%llu expected=%llu\n",
                     info.node_id.c_str(),
                     static_cast<unsigned long long>(msg.chunk_offset),
                     static_cast<unsigned long long>(state->active_load.received_bytes));
        return false;
    }

    state->active_load.buffer.append(msg.chunk_data);
    state->active_load.received_bytes =
        static_cast<std::uint64_t>(state->active_load.buffer.size());

    std::printf("[%s] received LoadWeightsChunk rid=%u "
                "expert=%d local_gpu_id=%d tensor_kind=%s chunk_offset=%llu chunk_size=%zu "
                "received=%llu/%llu\n",
                info.node_id.c_str(),
                req.request_id,
                msg.expert_id,
                msg.local_gpu_id,
                TensorKindName(msg.tensor_kind),
                static_cast<unsigned long long>(msg.chunk_offset),
                msg.chunk_data.size(),
                static_cast<unsigned long long>(state->active_load.received_bytes),
                static_cast<unsigned long long>(state->active_load.total_bytes));

    bool ok = SendEmptyAck(fd, common::MsgType::LoadWeightsAck, req.request_id);
    if (ok) {
        std::printf("[%s] sent LoadWeightsAck rid=%u\n",
                    info.node_id.c_str(), req.request_id);
    }
    return ok;
}

bool HandleLoadWeightsEnd(
    int fd,
    const common::NodeInfo& info,
    ControlState* state,
    const common::MsgHeader& req,
    const std::string& req_body) {
    if (state == nullptr) {
        std::fprintf(stderr, "[%s] HandleLoadWeightsEnd: state is null\n",
                     info.node_id.c_str());
        return false;
    }

    common::LoadWeightsEndMsg msg;
    if (!common::DecodeLoadWeightsEndBody(req_body, &msg)) {
        std::fprintf(stderr, "[%s] failed to decode LoadWeightsEnd\n",
                     info.node_id.c_str());
        return false;
    }

    if (!state->active_load.active) {
        std::fprintf(stderr, "[%s] LoadWeightsEnd with no active load\n",
                     info.node_id.c_str());
        return false;
    }

    auto* entry = state->registry.find_entry(msg.expert_id);
    if (entry == nullptr) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsEnd for unknown expert=%d\n",
                     info.node_id.c_str(), msg.expert_id);
        return false;
    }

    if (entry->local_gpu_id != msg.local_gpu_id) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsEnd gpu mismatch for expert=%d: "
                     "got local_gpu_id=%d expected=%d\n",
                     info.node_id.c_str(),
                     msg.expert_id,
                     msg.local_gpu_id,
                     entry->local_gpu_id);
        return false;
    }

    if (state->active_load.expert_id != msg.expert_id) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsEnd expert mismatch: got=%d expected=%d\n",
                     info.node_id.c_str(),
                     msg.expert_id,
                     state->active_load.expert_id);
        return false;
    }

    if (state->active_load.tensor_kind != msg.tensor_kind) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsEnd tensor_kind mismatch\n",
                     info.node_id.c_str());
        return false;
    }

    if (state->active_load.received_bytes != state->active_load.total_bytes) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsEnd byte mismatch: received=%llu expected=%llu\n",
                     info.node_id.c_str(),
                     static_cast<unsigned long long>(state->active_load.received_bytes),
                     static_cast<unsigned long long>(state->active_load.total_bytes));
        return false;
    }

    if (state->active_load.buffer.size() !=
        static_cast<std::size_t>(state->active_load.total_bytes)) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsEnd buffer size mismatch: buffer=%zu expected=%llu\n",
                     info.node_id.c_str(),
                     state->active_load.buffer.size(),
                     static_cast<unsigned long long>(state->active_load.total_bytes));
        return false;
    }

    const std::size_t final_buffer_size = state->active_load.buffer.size();

    if (!state->registry.store_incoming_tensor(
            msg.expert_id,
            msg.tensor_kind,
            state->active_load.total_bytes,
            std::move(state->active_load.buffer))) {
        std::fprintf(stderr,
                     "[%s] failed to store incoming tensor for expert=%d tensor_kind=%s\n",
                     info.node_id.c_str(),
                     msg.expert_id,
                     TensorKindName(msg.tensor_kind));
        return false;
    }

    FillFixedTensorMetaForEntry(entry);

    if (entry->incoming_ready) {
        entry->resident_ready = false;
        expert_node_v2::FreeExpertWeightsV2(&entry->storage);

        if (!expert_node_v2::UploadExpertForGpuV2(
                entry->local_gpu_id,
                entry->incoming,
                &entry->storage)) {
            std::fprintf(stderr,
                         "[%s] UploadExpertForGpuV2 failed for expert=%d gpu=%d\n",
                         info.node_id.c_str(),
                         entry->expert_id,
                         entry->local_gpu_id);
            return false;
        }

        entry->resident_ready = true;
    }

    std::printf("[%s] received LoadWeightsEnd rid=%u "
                "expert=%d local_gpu_id=%d tensor_kind=%s total_bytes=%llu buffer_size=%zu "
                "incoming_ready=%d resident_ready=%d\n",
                info.node_id.c_str(),
                req.request_id,
                msg.expert_id,
                msg.local_gpu_id,
                TensorKindName(msg.tensor_kind),
                static_cast<unsigned long long>(state->active_load.total_bytes),
                final_buffer_size,
                static_cast<int>(entry->incoming_ready),
                static_cast<int>(entry->resident_ready));

    state->registry.debug_print();
    state->active_load = ActiveLoad{};

    bool ok = SendEmptyAck(fd, common::MsgType::LoadWeightsAck, req.request_id);
    if (ok) {
        std::printf("[%s] sent LoadWeightsAck rid=%u\n",
                    info.node_id.c_str(), req.request_id);
    }
    return ok;
}

bool HandleInferRequest(
    int fd,
    const common::NodeInfo& info,
    ControlState* state,
    const common::MsgHeader& req,
    const std::string& req_body) 

bool DispatchRequest(
    int fd,
    const common::NodeInfo& info,
    ControlState* state,
    const common::MsgHeader& req,
    const std::string& req_body) {
    auto msg_type = static_cast<common::MsgType>(req.msg_type);

    switch (msg_type) {
        case common::MsgType::InventoryRequest:
            return HandleInventoryRequest(fd, state, req, req_body);
        case common::MsgType::HeartbeatRequest:
            return HandleHeartbeatRequest(fd, state, req, req_body);
        case common::MsgType::PlacementPlan:
            return HandlePlacementPlan(fd, info, state, req, req_body);
        case common::MsgType::LoadWeightsBegin:
            return HandleLoadWeightsBegin(fd, info, state, req, req_body);
        case common::MsgType::LoadWeightsChunk:
            return HandleLoadWeightsChunk(fd, info, state, req, req_body);
        case common::MsgType::LoadWeightsEnd:
            return HandleLoadWeightsEnd(fd, info, state, req, req_body);
        case common::MsgType::InferRequest:
            return HandleInferRequest(fd, info, state, req, req_body);
        default:
            std::fprintf(stderr, "[%s] unsupported msg_type: %u\n",
                         info.node_id.c_str(), req.msg_type);
            return false;
    }
}

bool HandleOneRequest(
    int fd,
    const common::NodeInfo& info,
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

    return DispatchRequest(fd, info, state, req, req_body);
}

bool HandleClientLoop(
    int fd,
    const common::NodeInfo& info,
    ControlState* state) {
    while (true) {
        if (!HandleOneRequest(fd, info, state)) {
            return false;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    int control_port = 5000;
    int worker_port_base = 6000;

    if (argc >= 2) {
        control_port = std::atoi(argv[1]);
    }
    if (argc >= 3) {
        worker_port_base = std::atoi(argv[2]);
    }

    common::NodeInfo info = BuildRealInventory(control_port, worker_port_base);

    ControlState state;
    state.node_id = "node0";
    state.control_port = control_port;
    state.worker_port_base = worker_port_base;
    state.node_status = common::NodeStatus::Registered;

    int listen_fd = ListenTcp(control_port);
    if (listen_fd < 0) {
        std::fprintf(stderr, "failed to listen on port %d\n", control_port);
        return 1;
    }

    std::printf("[%s] control stub listening on port %d\n",
                info.node_id.c_str(), control_port);

    while (true) {
        int fd = ::accept(listen_fd, nullptr, nullptr);
        if (fd < 0) {
            continue;
        }

        std::printf("[%s] client connected\n", info.node_id.c_str());
        bool ok = HandleClientLoop(fd, info, &state);
        if (!ok) {
            std::printf("[%s] client disconnected or handler failed\n",
                        info.node_id.c_str());
        }
        ::close(fd);
    }

    return 0;
}
