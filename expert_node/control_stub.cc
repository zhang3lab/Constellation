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
#include "common/inventory_codec.h"
#include "common/placement_codec.h"
#include "common/protocol.h"
#include "common/socket_utils.h"
#include "common/types.h"
#include "common/weight_codec.h"

namespace {

struct ExpertResidency {
    int expert_id = -1;
    int local_gpu_id = -1;
    bool ready = false;
};

struct ActiveLoad {
    bool active = false;
    int expert_id = -1;
    common::TensorKind tensor_kind = common::TensorKind::WUp;
    std::uint64_t total_bytes = 0;
    std::uint64_t received_bytes = 0;
};

struct ControlState {
    common::NodeStatus node_status = common::NodeStatus::Booting;
    std::unordered_map<int, ExpertResidency> expert_table;
    ActiveLoad active_load;
};

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
        default:
            return "unknown";
    }
}

common::NodeInfo BuildRealInventory(int control_port, int worker_port_base) {
    common::NodeInfo node;
    node.node_id = "node0";
    node.host = "127.0.0.1";
    node.control_port = control_port;

    int num_devices = 0;
    cudaError_t err = cudaGetDeviceCount(&num_devices);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return node;
    }

    for (int i = 0; i < num_devices; ++i) {
        cudaDeviceProp prop{};
        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "cudaGetDeviceProperties(%d) failed: %s\n",
                         i, cudaGetErrorString(err));
            continue;
        }

        err = cudaSetDevice(i);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "cudaSetDevice(%d) failed: %s\n",
                         i, cudaGetErrorString(err));
            continue;
        }

        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        err = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "cudaMemGetInfo(%d) failed: %s\n",
                         i, cudaGetErrorString(err));
            continue;
        }

        common::GpuInfo gpu;
        gpu.gpu_uid = node.node_id + ":" + std::to_string(i);
        gpu.local_gpu_id = i;
        gpu.gpu_name = prop.name;
        gpu.total_mem_bytes = static_cast<std::uint64_t>(total_bytes);
        gpu.free_mem_bytes = static_cast<std::uint64_t>(free_bytes);
        gpu.worker_port = worker_port_base + i;
        gpu.status = common::GpuStatus::Idle;

        node.gpus.push_back(gpu);
    }

    return node;
}

void PrintExpertTable(const common::NodeInfo& info, const ControlState& state) {
    std::vector<int> expert_ids;
    expert_ids.reserve(state.expert_table.size());

    for (const auto& kv : state.expert_table) {
        expert_ids.push_back(kv.first);
    }
    std::sort(expert_ids.begin(), expert_ids.end());

    std::printf("[%s] expert_table size = %zu\n",
                info.node_id.c_str(),
                state.expert_table.size());

    for (int expert_id : expert_ids) {
        const auto& r = state.expert_table.at(expert_id);
        std::printf("[%s]   expert=%d local_gpu_id=%d ready=%d\n",
                    info.node_id.c_str(),
                    r.expert_id,
                    r.local_gpu_id,
                    static_cast<int>(r.ready));
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

bool HandleInventoryRequest(
    int fd,
    const common::NodeInfo& info,
    ControlState* state,
    const common::MsgHeader& req,
    const std::string& req_body) {
    if (!req_body.empty()) {
        std::fprintf(stderr, "InventoryRequest body must be empty\n");
        return false;
    }

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
    const common::NodeInfo& info,
    const common::MsgHeader& req,
    const std::string& req_body) {
    if (!req_body.empty()) {
        std::fprintf(stderr, "HeartbeatRequest body must be empty\n");
        return false;
    }

    std::printf("[%s] received HeartbeatRequest rid=%u\n",
                info.node_id.c_str(), req.request_id);

    bool ok = SendEmptyAck(fd, common::MsgType::HeartbeatReply, req.request_id);
    if (ok) {
        std::printf("[%s] sent HeartbeatReply rid=%u\n",
                    info.node_id.c_str(), req.request_id);
    }
    return ok;
}

bool HandlePlacementPlan(
    int fd,
    const common::NodeInfo& info,
    ControlState* state,
    const common::MsgHeader& req,
    const std::string& req_body) {
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

    state->expert_table.clear();
    for (const auto& a : assignments) {
        ExpertResidency r;
        r.expert_id = a.expert_id;
        r.local_gpu_id = a.local_gpu_id;
        r.ready = false;
        state->expert_table[a.expert_id] = r;
    }

    state->active_load = ActiveLoad{};
    PrintExpertTable(info, *state);

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
    common::LoadWeightsBeginMsg msg;
    if (!common::DecodeLoadWeightsBeginBody(req_body, &msg)) {
        std::fprintf(stderr, "[%s] failed to decode LoadWeightsBegin\n",
                     info.node_id.c_str());
        return false;
    }

    auto it = state->expert_table.find(msg.expert_id);
    if (it == state->expert_table.end()) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsBegin for unknown expert=%d\n",
                     info.node_id.c_str(), msg.expert_id);
        return false;
    }

    if (it->second.local_gpu_id != msg.local_gpu_id) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsBegin gpu mismatch for expert=%d: "
                     "got local_gpu_id=%d expected=%d\n",
                     info.node_id.c_str(),
                     msg.expert_id,
                     msg.local_gpu_id,
                     it->second.local_gpu_id);
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

    auto it = state->expert_table.find(msg.expert_id);
    if (it == state->expert_table.end()) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsChunk for unknown expert=%d\n",
                     info.node_id.c_str(), msg.expert_id);
        return false;
    }

    if (it->second.local_gpu_id != msg.local_gpu_id) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsChunk gpu mismatch for expert=%d: "
                     "got local_gpu_id=%d expected=%d\n",
                     info.node_id.c_str(),
                     msg.expert_id,
                     msg.local_gpu_id,
                     it->second.local_gpu_id);
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

    state->active_load.received_bytes +=
        static_cast<std::uint64_t>(msg.chunk_data.size());

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

    auto it = state->expert_table.find(msg.expert_id);
    if (it == state->expert_table.end()) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsEnd for unknown expert=%d\n",
                     info.node_id.c_str(), msg.expert_id);
        return false;
    }

    if (it->second.local_gpu_id != msg.local_gpu_id) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsEnd gpu mismatch for expert=%d: "
                     "got local_gpu_id=%d expected=%d\n",
                     info.node_id.c_str(),
                     msg.expert_id,
                     msg.local_gpu_id,
                     it->second.local_gpu_id);
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

    it->second.ready = true;

    std::printf("[%s] received LoadWeightsEnd rid=%u "
                "expert=%d local_gpu_id=%d tensor_kind=%s total_bytes=%llu -> ready=1\n",
                info.node_id.c_str(),
                req.request_id,
                msg.expert_id,
                msg.local_gpu_id,
                TensorKindName(msg.tensor_kind),
                static_cast<unsigned long long>(state->active_load.total_bytes));

    state->active_load = ActiveLoad{};

    bool ok = SendEmptyAck(fd, common::MsgType::LoadWeightsAck, req.request_id);
    if (ok) {
        std::printf("[%s] sent LoadWeightsAck rid=%u\n",
                    info.node_id.c_str(), req.request_id);
    }
    return ok;
}

bool DispatchRequest(
    int fd,
    const common::NodeInfo& info,
    ControlState* state,
    const common::MsgHeader& req,
    const std::string& req_body) {
    auto msg_type = static_cast<common::MsgType>(req.msg_type);

    switch (msg_type) {
        case common::MsgType::InventoryRequest:
            return HandleInventoryRequest(fd, info, state, req, req_body);
        case common::MsgType::HeartbeatRequest:
            return HandleHeartbeatRequest(fd, info, req, req_body);
        case common::MsgType::PlacementPlan:
            return HandlePlacementPlan(fd, info, state, req, req_body);
        case common::MsgType::LoadWeightsBegin:
            return HandleLoadWeightsBegin(fd, info, state, req, req_body);
        case common::MsgType::LoadWeightsChunk:
            return HandleLoadWeightsChunk(fd, info, state, req, req_body);
        case common::MsgType::LoadWeightsEnd:
            return HandleLoadWeightsEnd(fd, info, state, req, req_body);
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
