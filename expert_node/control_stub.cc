#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/header_codec.h"
#include "common/inventory_codec.h"
#include "common/protocol.h"
#include "common/socket_utils.h"
#include "common/types.h"

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

common::NodeInfo BuildRealInventory(int control_port, int worker_port_base) {
    common::NodeInfo node;
    node.node_id = "node0";
    node.host = "127.0.0.1";
    node.control_port = control_port;
    node.status = common::NodeStatus::Registered;

    int num_devices = 0;
    cudaError_t err = cudaGetDeviceCount(&num_devices);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        node.status = common::NodeStatus::Failed;
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

bool HandleOneRequest(int fd, const common::NodeInfo& info) {
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

    auto msg_type = static_cast<common::MsgType>(req.msg_type);

    if (msg_type == common::MsgType::InventoryRequest) {
        if (!req_body.empty()) {
            std::fprintf(stderr, "InventoryRequest body must be empty\n");
            return false;
        }

        std::printf("received InventoryRequest rid=%u\n", req.request_id);

        std::string body = common::EncodeInventoryReplyBody(info);

        common::MsgHeader resp{};
        resp.magic = common::kMagic;
        resp.version = common::kVersion;
        resp.msg_type = static_cast<std::uint16_t>(common::MsgType::InventoryReply);
        resp.request_id = req.request_id;
        resp.body_len = static_cast<std::uint32_t>(body.size());

        bool ok = common::SendMessage(fd, resp, body);
        if (ok) {
            std::printf("sent InventoryReply rid=%u body_len=%u\n",
                        resp.request_id, resp.body_len);
        }
        return ok;
    }

    if (msg_type == common::MsgType::HeartbeatRequest) {
        if (!req_body.empty()) {
            std::fprintf(stderr, "HeartbeatRequest body must be empty\n");
            return false;
        }

        std::printf("received HeartbeatRequest rid=%u\n", req.request_id);

        common::MsgHeader resp{};
        resp.magic = common::kMagic;
        resp.version = common::kVersion;
        resp.msg_type = static_cast<std::uint16_t>(common::MsgType::HeartbeatReply);
        resp.request_id = req.request_id;
        resp.body_len = 0;

        bool ok = common::SendMessage(fd, resp, std::string());
        if (ok) {
            std::printf("sent HeartbeatReply rid=%u\n", resp.request_id);
        }
        return ok;
    }

    std::fprintf(stderr, "unsupported msg_type: %u\n", req.msg_type);
    return false;
}

bool HandleClientLoop(int fd, const common::NodeInfo& info) {
    while (true) {
        if (!HandleOneRequest(fd, info)) {
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

    int listen_fd = ListenTcp(control_port);
    if (listen_fd < 0) {
        std::fprintf(stderr, "failed to listen on port %d\n", control_port);
        return 1;
    }

    std::printf("control stub listening on port %d\n", control_port);

    while (true) {
        int fd = ::accept(listen_fd, nullptr, nullptr);
        if (fd < 0) {
            continue;
        }

        std::printf("client connected\n");
        bool ok = HandleClientLoop(fd, info);
        if (!ok) {
            std::printf("client disconnected or handler failed\n");
        }
        ::close(fd);
    }

    return 0;
}
