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
#include "expert_node/expert_runtime.h"
#include "expert_node/kernel/expert.h"

namespace {

struct DeviceTensor {
    common::TensorKind tensor_kind = common::TensorKind::WUp;
    void* device_ptr = nullptr;
    std::uint64_t total_bytes = 0;
};

struct HostTensor {
    common::TensorKind tensor_kind = common::TensorKind::WUp;
    std::string bytes;
    std::uint64_t total_bytes = 0;
};

struct ExpertResidency {
    int expert_id = -1;
    int local_gpu_id = -1;
    bool ready = false;
    std::unordered_map<int, HostTensor> host_tensors;
    std::unordered_map<int, DeviceTensor> device_tensors;
};

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
    std::unordered_map<int, ExpertResidency> expert_table;
    ActiveLoad active_load;
    expert_node::ExpertRuntime runtime;
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

void PrintRuntimeSummary(const common::NodeInfo& info, const ControlState& state) {
    std::printf("[%s] runtime loaded experts = %zu\n",
                info.node_id.c_str(),
                state.runtime.size());

    for (const auto& kv : state.expert_table) {
        int expert_id = kv.first;
        const auto* loaded = state.runtime.find_loaded_expert(expert_id);
        if (loaded == nullptr) {
            continue;
        }

        std::printf("[%s]   loaded expert=%d gpu=%d ready=%d "
                    "up_w=%p gate_w=%p down_w=%p\n",
                    info.node_id.c_str(),
                    loaded->expert_id,
                    loaded->local_gpu_id,
                    static_cast<int>(loaded->ready),
                    loaded->mlp.w_up.weights,
                    loaded->mlp.w_gate.weights,
                    loaded->mlp.w_down.weights);
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

bool HasCompleteExpertTriplet(const ExpertResidency& r) {
    const int up = static_cast<int>(common::TensorKind::WUp);
    const int gate = static_cast<int>(common::TensorKind::WGate);
    const int down = static_cast<int>(common::TensorKind::WDown);

    return r.host_tensors.count(up) > 0 &&
           r.host_tensors.count(gate) > 0 &&
           r.host_tensors.count(down) > 0 &&
           r.device_tensors.count(up) > 0 &&
           r.device_tensors.count(gate) > 0 &&
           r.device_tensors.count(down) > 0;
}

void FreeDevicePackedMatrixOwner(expert_node::DevicePackedMatrixOwner* m) {
    if (m == nullptr) return;
    if (m->weights != nullptr) {
        cudaFree(m->weights);
        m->weights = nullptr;
    }
    if (m->scales != nullptr) {
        cudaFree(m->scales);
        m->scales = nullptr;
    }
    m->view = PackedRowMajorMatrix{};
}

void FreeDevicePackedMlpOwner(expert_node::DevicePackedMlpOwner* mlp) {
    if (mlp == nullptr) return;
    FreeDevicePackedMatrixOwner(&mlp->w_up);
    FreeDevicePackedMatrixOwner(&mlp->w_gate);
    FreeDevicePackedMatrixOwner(&mlp->w_down);
}

bool UploadTensorToDevice(
    int local_gpu_id,
    const HostTensor& host_tensor,
    DeviceTensor* out) {
    if (out == nullptr) return false;

    cudaError_t err = cudaSetDevice(local_gpu_id);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaSetDevice(%d) failed: %s\n",
                     local_gpu_id, cudaGetErrorString(err));
        return false;
    }

    void* ptr = nullptr;
    err = cudaMalloc(&ptr, static_cast<std::size_t>(host_tensor.total_bytes));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc(%llu) failed: %s\n",
                     static_cast<unsigned long long>(host_tensor.total_bytes),
                     cudaGetErrorString(err));
        return false;
    }

    err = cudaMemcpy(
        ptr,
        host_tensor.bytes.data(),
        static_cast<std::size_t>(host_tensor.total_bytes),
        cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy H2D failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(ptr);
        return false;
    }

    out->tensor_kind = host_tensor.tensor_kind;
    out->device_ptr = ptr;
    out->total_bytes = host_tensor.total_bytes;
    return true;
}

bool UploadPackedRowMajorMatrixHostToDevice(
    const PackedRowMajorMatrixHost& src,
    expert_node::DevicePackedMatrixOwner* out) {
    if (out == nullptr) return false;
    if (!src.weights || !src.scales) return false;

    const size_t w_bytes = packed_weights_bytes(src.rows, src.cols);
    const size_t s_bytes = packed_scales_bytes(src.rows, src.cols, src.k_chunk);

    uint8_t* d_weights = nullptr;
    float* d_scales = nullptr;

    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_weights), w_bytes);
    if (err != cudaSuccess) return false;

    err = cudaMalloc(reinterpret_cast<void**>(&d_scales), s_bytes);
    if (err != cudaSuccess) {
        cudaFree(d_weights);
        return false;
    }

    err = cudaMemcpy(d_weights, src.weights, w_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_weights);
        cudaFree(d_scales);
        return false;
    }

    err = cudaMemcpy(d_scales, src.scales, s_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_weights);
        cudaFree(d_scales);
        return false;
    }

    out->weights = d_weights;
    out->scales = d_scales;
    out->view.rows = src.rows;
    out->view.cols = src.cols;
    out->view.k_chunk = src.k_chunk;
    out->view.num_k_chunks = src.num_k_chunks;
    out->view.fp8_format = src.fp8_format;
    out->view.weights = d_weights;
    out->view.scales = d_scales;
    return true;
}

bool BuildPackedHostMatrixFromHostTensor(
    const HostTensor& ht,
    int rows,
    int cols,
    PackedRowMajorMatrixHost* out) {
    if (out == nullptr) return false;
    if (rows <= 0 || cols <= 0) return false;

    const size_t weight_bytes =
        static_cast<size_t>(rows) * static_cast<size_t>(cols);
    if (ht.bytes.size() != weight_bytes) {
        std::fprintf(stderr,
                     "[loader] host tensor size mismatch: got=%zu expected=%zu\n",
                     ht.bytes.size(),
                     weight_bytes);
        return false;
    }

    out->rows = rows;
    out->cols = cols;
    out->fp8_format = Fp8Format::TORCH_E4M3FN;
    out->k_chunk = DefaultConfig::k_chunk;
    out->num_k_chunks = ceil_div_int(cols, out->k_chunk);
    out->weights = nullptr;
    out->scales = nullptr;

    const size_t w_bytes = packed_weights_bytes(rows, cols);
    const size_t s_elems =
        static_cast<size_t>(rows) * static_cast<size_t>(out->num_k_chunks);
    const size_t s_bytes = s_elems * sizeof(float);

    out->weights = static_cast<uint8_t*>(std::malloc(w_bytes));
    out->scales = static_cast<float*>(std::malloc(s_bytes));
    if (!out->weights || !out->scales) {
        if (out->weights) std::free(out->weights);
        if (out->scales) std::free(out->scales);
        out->weights = nullptr;
        out->scales = nullptr;
        return false;
    }

    std::memcpy(out->weights, ht.bytes.data(), w_bytes);
    for (size_t i = 0; i < s_elems; ++i) {
        out->scales[i] = 1.0f;
    }

    std::printf("[loader] packed rows=%d cols=%d k_chunk=%d num_k_chunks=%d fmt=%d\n",
                out->rows, out->cols, out->k_chunk, out->num_k_chunks,
                fp8_format_to_int(out->fp8_format));
    std::printf("[loader] first weight bytes = %u %u %u %u\n",
                static_cast<unsigned>(out->weights[0]),
                static_cast<unsigned>(out->weights[1]),
                static_cast<unsigned>(out->weights[2]),
                static_cast<unsigned>(out->weights[3]));

    return true;
}

bool BuildLoadedExpertFromResidency(
    const ExpertResidency& r,
    expert_node::LoadedExpert* out) {
    if (out == nullptr) return false;
    if (!r.ready) return false;

    const auto& up_ht =
        r.host_tensors.at(static_cast<int>(common::TensorKind::WUp));
    const auto& gate_ht =
        r.host_tensors.at(static_cast<int>(common::TensorKind::WGate));
    const auto& down_ht =
        r.host_tensors.at(static_cast<int>(common::TensorKind::WDown));

    PackedRowMajorMatrixHost up_host{};
    PackedRowMajorMatrixHost gate_host{};
    PackedRowMajorMatrixHost down_host{};

    if (!BuildPackedHostMatrixFromHostTensor(up_ht, 2048, 7168, &up_host)) return false;
    if (!BuildPackedHostMatrixFromHostTensor(gate_ht, 2048, 7168, &gate_host)) {
        free_packed_row_major_matrix_host(&up_host);
        return false;
    }
    if (!BuildPackedHostMatrixFromHostTensor(down_ht, 7168, 2048, &down_host)) {
        free_packed_row_major_matrix_host(&up_host);
        free_packed_row_major_matrix_host(&gate_host);
        return false;
    }

    expert_node::LoadedExpert tmp;
    tmp.expert_id = r.expert_id;
    tmp.local_gpu_id = r.local_gpu_id;

    cudaError_t err = cudaSetDevice(r.local_gpu_id);
    if (err != cudaSuccess) {
        free_packed_row_major_matrix_host(&up_host);
        free_packed_row_major_matrix_host(&gate_host);
        free_packed_row_major_matrix_host(&down_host);
        return false;
    }

    bool ok =
        UploadPackedRowMajorMatrixHostToDevice(up_host, &tmp.packed.w_up) &&
        UploadPackedRowMajorMatrixHostToDevice(gate_host, &tmp.packed.w_gate) &&
        UploadPackedRowMajorMatrixHostToDevice(down_host, &tmp.packed.w_down);

    free_packed_row_major_matrix_host(&up_host);
    free_packed_row_major_matrix_host(&gate_host);
    free_packed_row_major_matrix_host(&down_host);

    if (!ok) {
        FreeDevicePackedMlpOwner(&tmp.packed);
        return false;
    }

    tmp.mlp.w_up = tmp.packed.w_up.view;
    tmp.mlp.w_gate = tmp.packed.w_gate.view;
    tmp.mlp.w_down = tmp.packed.w_down.view;
    tmp.ready = true;

    *out = tmp;
    return true;
}

void FreeDeviceTensor(DeviceTensor* t) {
    if (t == nullptr) return;
    if (t->device_ptr != nullptr) {
        cudaFree(t->device_ptr);
        t->device_ptr = nullptr;
    }
    t->total_bytes = 0;
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

    for (auto& kv : state->expert_table) {
        auto& r = kv.second;
        for (auto& dv : r.device_tensors) {
            FreeDeviceTensor(&dv.second);
        }
    }
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

void PrintExpertTensorSummary(
    const common::NodeInfo& info,
    const ControlState& state) {
    std::vector<int> expert_ids;
    expert_ids.reserve(state.expert_table.size());

    for (const auto& kv : state.expert_table) {
        expert_ids.push_back(kv.first);
    }
    std::sort(expert_ids.begin(), expert_ids.end());

    std::printf("[%s] host tensor summary\n", info.node_id.c_str());

    for (int expert_id : expert_ids) {
        const auto& r = state.expert_table.at(expert_id);
	std::printf("[%s]   expert=%d gpu=%d ready=%d host_tensors=%zu device_tensors=%zu\n",
                    info.node_id.c_str(),
                    r.expert_id,
                    r.local_gpu_id,
                    static_cast<int>(r.ready),
                    r.host_tensors.size(),
                    r.device_tensors.size());

        std::vector<int> tensor_keys;
        tensor_keys.reserve(r.host_tensors.size());
        for (const auto& kv : r.host_tensors) {
            tensor_keys.push_back(kv.first);
        }
        std::sort(tensor_keys.begin(), tensor_keys.end());

        for (int tensor_key : tensor_keys) {
            const auto& ht = r.host_tensors.at(tensor_key);
            std::printf("[%s]     tensor_kind=%s bytes=%llu\n",
                        info.node_id.c_str(),
                        TensorKindName(ht.tensor_kind),
                        static_cast<unsigned long long>(ht.total_bytes));
        }

	std::vector<int> device_keys;
        device_keys.reserve(r.device_tensors.size());
        for (const auto& kv : r.device_tensors) {
            device_keys.push_back(kv.first);
        }
        std::sort(device_keys.begin(), device_keys.end());

        for (int tensor_key : device_keys) {
            const auto& dt = r.device_tensors.at(tensor_key);
            std::printf("[%s]     device_tensor kind=%s bytes=%llu ptr=%p\n",
                        info.node_id.c_str(),
                        TensorKindName(dt.tensor_kind),
                        static_cast<unsigned long long>(dt.total_bytes),
                        dt.device_ptr);
        }
    }
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

    if (state->active_load.buffer.size() !=
        static_cast<std::size_t>(state->active_load.total_bytes)) {
        std::fprintf(stderr,
                     "[%s] LoadWeightsEnd buffer size mismatch: buffer=%zu expected=%llu\n",
                     info.node_id.c_str(),
                     state->active_load.buffer.size(),
                     static_cast<unsigned long long>(state->active_load.total_bytes));
        return false;
    }

    std::size_t final_buffer_size = state->active_load.buffer.size();
    int tensor_key = static_cast<int>(msg.tensor_kind);

    HostTensor ht;
    ht.tensor_kind = msg.tensor_kind;
    ht.total_bytes = state->active_load.total_bytes;
    ht.bytes = std::move(state->active_load.buffer);

    it->second.host_tensors[tensor_key] = std::move(ht);

    auto host_it = it->second.host_tensors.find(tensor_key);
    if (host_it == it->second.host_tensors.end()) {
        std::fprintf(stderr, "[%s] internal error: host tensor missing after store\n",
                     info.node_id.c_str());
        return false;
    }

    auto dev_it = it->second.device_tensors.find(tensor_key);
    if (dev_it != it->second.device_tensors.end()) {
        FreeDeviceTensor(&dev_it->second);
    }

    DeviceTensor dt;
    if (!UploadTensorToDevice(it->second.local_gpu_id, host_it->second, &dt)) {
        std::fprintf(stderr,
                     "[%s] UploadTensorToDevice failed for expert=%d tensor_kind=%s\n",
                     info.node_id.c_str(),
                     it->second.expert_id,
                     TensorKindName(msg.tensor_kind));
        return false;
    }
    it->second.device_tensors[tensor_key] = std::move(dt);

    bool complete = HasCompleteExpertTriplet(it->second);

    it->second.ready = complete;

    if (it->second.ready) {
        expert_node::LoadedExpert le;
        if (!BuildLoadedExpertFromResidency(it->second, &le)) {
            std::fprintf(stderr,
                         "[%s] BuildLoadedExpertFromResidency failed for expert=%d\n",
                         info.node_id.c_str(),
                         it->second.expert_id);
            return false;
        }
     
        if (!state->runtime.register_loaded_expert(le)) {
            std::fprintf(stderr,
                         "[%s] runtime.register_loaded_expert failed for expert=%d\n",
                         info.node_id.c_str(),
                         le.expert_id);
            FreeDevicePackedMlpOwner(&le.packed);
            return false;
        }
    }

    std::printf("[%s] received LoadWeightsEnd rid=%u "
                "expert=%d local_gpu_id=%d tensor_kind=%s total_bytes=%llu buffer_size=%zu -> ready=%d\n",
                info.node_id.c_str(),
                req.request_id,
                msg.expert_id,
                msg.local_gpu_id,
                TensorKindName(msg.tensor_kind),
                static_cast<unsigned long long>(state->active_load.total_bytes),
                final_buffer_size,
                static_cast<int>(it->second.ready));

    PrintExpertTensorSummary(info, *state);
    PrintRuntimeSummary(info, *state);

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
    const std::string& req_body) {
    common::InferRequestMsg msg;
    if (!common::DecodeInferRequestBody(req_body, &msg)) {
        std::fprintf(stderr, "[%s] failed to decode InferRequest\n",
                     info.node_id.c_str());
        return false;
    }

    auto send_infer_response = [&](int status_code,
                                   int batch_size,
                                   int hidden_dim,
                                   const std::string& output) -> bool {
        common::InferResponseMsg resp_msg;
        resp_msg.status_code = status_code;
        resp_msg.batch_size = batch_size;
        resp_msg.hidden_dim = hidden_dim;
        resp_msg.output = output;

        std::string resp_body = common::EncodeInferResponseBody(resp_msg);

        common::MsgHeader resp{};
        resp.magic = common::kMagic;
        resp.version = common::kVersion;
        resp.msg_type = static_cast<std::uint16_t>(common::MsgType::InferResponse);
        resp.request_id = req.request_id;
        resp.body_len = static_cast<std::uint32_t>(resp_body.size());

        bool ok = common::SendMessage(fd, resp, resp_body);
        if (ok) {
            std::printf("[%s] sent InferResponse rid=%u status=%d output_bytes=%zu\n",
                        info.node_id.c_str(),
                        req.request_id,
                        status_code,
                        output.size());
        }
        return ok;
    };

    std::printf("[%s] received InferRequest rid=%u expert=%d batch=%d hidden=%d activation_bytes=%zu\n",
                info.node_id.c_str(),
                req.request_id,
                msg.expert_id,
                msg.batch_size,
                msg.hidden_dim,
                msg.activation.size());

    const expert_node::LoadedExpert* expert =
        state->runtime.find_loaded_expert(msg.expert_id);
    if (expert == nullptr) {
        return send_infer_response(1, msg.batch_size, msg.hidden_dim, std::string());
    }
    if (!expert->ready) {
        return send_infer_response(2, msg.batch_size, msg.hidden_dim, std::string());
    }

    if (msg.batch_size <= 0 || msg.hidden_dim <= 0) {
        return send_infer_response(3, msg.batch_size, msg.hidden_dim, std::string());
    }

    const std::uint64_t expected_nbytes =
        static_cast<std::uint64_t>(msg.batch_size) *
        static_cast<std::uint64_t>(msg.hidden_dim) *
        sizeof(float);

    if (msg.activation.size() != expected_nbytes) {
        std::fprintf(stderr,
                     "[%s] InferRequest activation size mismatch: got=%zu expected=%llu\n",
                     info.node_id.c_str(),
                     msg.activation.size(),
                     static_cast<unsigned long long>(expected_nbytes));
        return send_infer_response(3, msg.batch_size, msg.hidden_dim, std::string());
    }

    cudaError_t err = cudaSetDevice(expert->local_gpu_id);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[%s] cudaSetDevice(%d) failed: %s\n",
                     info.node_id.c_str(),
                     expert->local_gpu_id,
                     cudaGetErrorString(err));
        return send_infer_response(4, msg.batch_size, msg.hidden_dim, std::string());
    }

    const size_t act_elems =
        static_cast<size_t>(msg.batch_size) * static_cast<size_t>(msg.hidden_dim);
    const size_t act_f_bytes = act_elems * sizeof(float);
    const size_t act_h_bytes = act_elems * sizeof(__half);

    MlpShape shape;
    shape.num_tokens = msg.batch_size;
    shape.hidden_dim = DefaultConfig::hidden_dim;
    shape.inter_dim = DefaultConfig::inter_dim;
    shape.k_chunk = DefaultConfig::k_chunk;
    shape.rows_per_cta = DefaultConfig::rows_per_cta;
    shape.fp8_format = expert->mlp.w_up.fp8_format;

    if (msg.hidden_dim != shape.hidden_dim) {
        std::fprintf(stderr,
                     "[%s] InferRequest hidden_dim mismatch: got=%d expected=%d\n",
                     info.node_id.c_str(),
                     msg.hidden_dim,
                     shape.hidden_dim);
        return send_infer_response(3, msg.batch_size, msg.hidden_dim, std::string());
    }

    const size_t workspace_bytes = workspace_num_bytes(shape);

    float* d_input_f = nullptr;
    __half* d_input_h = nullptr;
    __half* d_output_h = nullptr;
    float* d_workspace = nullptr;

    auto cleanup = [&]() {
        if (d_input_f) cudaFree(d_input_f);
        if (d_input_h) cudaFree(d_input_h);
        if (d_output_h) cudaFree(d_output_h);
        if (d_workspace) cudaFree(d_workspace);
        d_input_f = nullptr;
        d_input_h = nullptr;
        d_output_h = nullptr;
        d_workspace = nullptr;
    };

    err = cudaMalloc(reinterpret_cast<void**>(&d_input_f), act_f_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[%s] cudaMalloc d_input_f failed: %s\n",
                     info.node_id.c_str(), cudaGetErrorString(err));
        cleanup();
        return send_infer_response(4, msg.batch_size, msg.hidden_dim, std::string());
    }

    err = cudaMalloc(reinterpret_cast<void**>(&d_input_h), act_h_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[%s] cudaMalloc d_input_h failed: %s\n",
                     info.node_id.c_str(), cudaGetErrorString(err));
        cleanup();
        return send_infer_response(4, msg.batch_size, msg.hidden_dim, std::string());
    }

    err = cudaMalloc(reinterpret_cast<void**>(&d_output_h), act_h_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[%s] cudaMalloc d_output_h failed: %s\n",
                     info.node_id.c_str(), cudaGetErrorString(err));
        cleanup();
        return send_infer_response(4, msg.batch_size, msg.hidden_dim, std::string());
    }

    err = cudaMalloc(reinterpret_cast<void**>(&d_workspace), workspace_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[%s] cudaMalloc d_workspace failed: %s\n",
                     info.node_id.c_str(), cudaGetErrorString(err));
        cleanup();
        return send_infer_response(4, msg.batch_size, msg.hidden_dim, std::string());
    }

    err = cudaMemcpy(d_input_f,
                     msg.activation.data(),
                     act_f_bytes,
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[%s] cudaMemcpy H2D input failed: %s\n",
                     info.node_id.c_str(), cudaGetErrorString(err));
        cleanup();
        return send_infer_response(4, msg.batch_size, msg.hidden_dim, std::string());
    }

    if (!expert::launch_cast_float_to_half(
            d_input_f,
            d_input_h,
            msg.batch_size,
            msg.hidden_dim,
            0)) {
        std::fprintf(stderr, "[%s] launch_cast_float_to_half failed\n",
                     info.node_id.c_str());
        cleanup();
        return send_infer_response(4, msg.batch_size, msg.hidden_dim, std::string());
    }

    bool ok = launch_mlp(
        expert->mlp,
        d_input_h,
        d_output_h,
        d_workspace,
        shape,
        0);

    if (!ok) {
        std::fprintf(stderr, "[%s] launch_mlp failed for expert=%d\n",
                     info.node_id.c_str(), msg.expert_id);
        cleanup();
        return send_infer_response(4, msg.batch_size, msg.hidden_dim, std::string());
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[%s] cudaDeviceSynchronize failed: %s\n",
                     info.node_id.c_str(), cudaGetErrorString(err));
        cleanup();
        return send_infer_response(4, msg.batch_size, msg.hidden_dim, std::string());
    }

    std::string output;
    output.resize(act_h_bytes);

    err = cudaMemcpy(output.data(),
                     d_output_h,
                     act_h_bytes,
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[%s] cudaMemcpy D2H output failed: %s\n",
                     info.node_id.c_str(), cudaGetErrorString(err));
        cleanup();
        return send_infer_response(4, msg.batch_size, msg.hidden_dim, std::string());
    }

    cleanup();
    return send_infer_response(0, msg.batch_size, msg.hidden_dim, output);
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
