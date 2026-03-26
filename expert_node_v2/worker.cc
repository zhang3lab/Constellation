#include "worker.h"

#include <cstdio>
#include <string>

#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>

#include "common/protocol.h"
#include "common/socket_utils.h"
#include "expert_node_v2/control.h"
#include "expert_node_v2/expert_registry_v2.h"

namespace {

int CreateListenSocket(int port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        std::perror("socket");
        return -1;
    }

    int opt = 1;
    if (::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) != 0) {
        std::perror("setsockopt(SO_REUSEADDR)");
        ::close(fd);
        return -1;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<std::uint16_t>(port));

    if (::bind(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::perror("bind");
        ::close(fd);
        return -1;
    }

    if (::listen(fd, 128) != 0) {
        std::perror("listen");
        ::close(fd);
        return -1;
    }

    return fd;
}

bool SendInferResponse(
    int fd,
    const std::string& node_id,
    std::uint32_t request_id,
    const common::InferResponseMsg& resp_msg) {
    std::string resp_body = common::EncodeInferResponseBody(resp_msg);

    common::MsgHeader resp{};
    resp.magic = common::kMagic;
    resp.version = common::kVersion;
    resp.msg_type = static_cast<std::uint16_t>(common::MsgType::InferResponse);
    resp.request_id = request_id;
    resp.body_len = static_cast<std::uint32_t>(resp_body.size());

    bool ok = common::SendMessage(fd, resp, resp_body);
    if (ok) {
        std::printf("[%s] sent InferResponse rid=%u status=%d output_dtype=%u output_bytes=%zu\n",
                    node_id.c_str(),
                    request_id,
                    resp_msg.status_code,
                    static_cast<unsigned>(resp_msg.output_dtype),
                    resp_msg.output.size());
    }
    return ok;
}

}  // namespace

bool HandleInferRequest(
    int fd,
    GpuWorkerContextV2* ctx,
    const common::MsgHeader& req,
    const std::string& req_body) {
    if (ctx == nullptr || ctx->state == nullptr || !ctx->workspace) {
        std::fprintf(stderr, "[worker] HandleInferRequest: invalid context\n");
        return false;
    }

    common::InferRequestMsg msg;
    if (!common::DecodeInferRequestBody(req_body, &msg)) {
        std::fprintf(stderr, "[%s] failed to decode InferRequest\n",
                     ctx->state->static_info.node_id.c_str());
        return false;
    }

    std::printf("[%s] received InferRequest rid=%u expert=%d batch=%d hidden=%d "
                "input_dtype=%u output_dtype=%u activation_bytes=%zu worker_id=%d\n",
                ctx->state->static_info.node_id.c_str(),
                req.request_id,
                msg.expert_id,
                msg.batch_size,
                msg.hidden_dim,
                static_cast<unsigned>(msg.input_dtype),
                static_cast<unsigned>(msg.output_dtype),
                msg.activation.size(),
                ctx->worker_id);

    common::InferResponseMsg resp_msg;
    resp_msg.status_code = 4;
    resp_msg.batch_size = msg.batch_size;
    resp_msg.hidden_dim = msg.hidden_dim;
    resp_msg.output_dtype = msg.output_dtype;
    resp_msg.output.clear();

    const ExpertDeviceStorageV2* storage =
        ctx->state->registry.FindDeviceStorage(msg.expert_id, ctx->worker_id);
    if (storage == nullptr) {
        const expert_node_v2::ExpertEntryV2* entry =
            ctx->state->registry.FindEntry(msg.expert_id);
        resp_msg.status_code = (entry == nullptr) ? 1 : 2;
        return SendInferResponse(
            fd,
            ctx->state->static_info.node_id,
            req.request_id,
            resp_msg);
    }

    if (!ctx->workspace->RunExpertRequest(*storage, msg, &resp_msg)) {
        resp_msg.status_code = 4;
        resp_msg.output.clear();
        resp_msg.output_dtype = msg.output_dtype;
        return SendInferResponse(
            fd,
            ctx->state->static_info.node_id,
            req.request_id,
            resp_msg);
    }

    return SendInferResponse(
        fd,
        ctx->state->static_info.node_id,
        req.request_id,
        resp_msg);
}

void RunGpuWorkerLoopV2(GpuWorkerContextV2* ctx) {
    if (ctx == nullptr || ctx->state == nullptr) {
        std::fprintf(stderr, "[worker] invalid ctx\n");
        return;
    }

    if (ctx->worker_id < 0 ||
        static_cast<std::size_t>(ctx->worker_id) >= ctx->state->static_info.gpus.size()) {
        std::fprintf(stderr,
                     "[%s] invalid worker_id=%d gpus.size()=%zu\n",
                     ctx->state->static_info.node_id.c_str(),
                     ctx->worker_id,
                     ctx->state->static_info.gpus.size());
        return;
    }

    const common::StaticGpuInfo& gpu =
        ctx->state->static_info.gpus[static_cast<std::size_t>(ctx->worker_id)];
    const common::GpuVendor vendor = gpu.gpu_vendor;
    const common::VendorWorkerSpan& vendor_span =
        ctx->state->static_info.vendor_spans[static_cast<std::size_t>(vendor)];

    ctx->workspace = expert_node_v2::CreateBackendWorkspaceV2(
        vendor_span,
        ctx->worker_id);
    if (!ctx->workspace) {
        std::fprintf(stderr,
                     "[%s] worker init failed for worker_id=%d vendor=%u port=%d\n",
                     ctx->state->static_info.node_id.c_str(),
                     ctx->worker_id,
                     static_cast<unsigned>(vendor),
                     ctx->worker_port);
        return;
    }

    int listen_fd = CreateListenSocket(ctx->worker_port);
    if (listen_fd < 0) {
        std::fprintf(stderr,
                     "[%s] failed to listen on worker port=%d worker_id=%d vendor=%u\n",
                     ctx->state->static_info.node_id.c_str(),
                     ctx->worker_port,
                     ctx->worker_id,
                     static_cast<unsigned>(vendor));
        return;
    }

    std::printf("[%s] gpu worker started worker_id=%d vendor=%u port=%d\n",
                ctx->state->static_info.node_id.c_str(),
                ctx->worker_id,
                static_cast<unsigned>(vendor),
                ctx->worker_port);

    for (;;) {
        int fd = ::accept(listen_fd, nullptr, nullptr);
        if (fd < 0) {
            std::perror("accept");
            continue;
        }

        for (;;) {
            std::uint8_t hdr_buf[16];
            if (!common::RecvAll(fd, hdr_buf, sizeof(hdr_buf))) {
                break;
            }

            common::MsgHeader req{};
            if (!common::DecodeHeader(hdr_buf, sizeof(hdr_buf), &req)) {
                std::fprintf(stderr,
                             "[%s] failed to decode request header on worker_id=%d\n",
                             ctx->state->static_info.node_id.c_str(),
                             ctx->worker_id);
                break;
            }

            if (req.magic != common::kMagic) {
                std::fprintf(stderr,
                             "[%s] bad magic on worker_id=%d: 0x%x\n",
                             ctx->state->static_info.node_id.c_str(),
                             ctx->worker_id,
                             req.magic);
                break;
            }

            if (req.version != common::kVersion) {
                std::fprintf(stderr,
                             "[%s] bad version on worker_id=%d: %u\n",
                             ctx->state->static_info.node_id.c_str(),
                             ctx->worker_id,
                             req.version);
                break;
            }

            std::string req_body;
            req_body.resize(req.body_len);
            if (req.body_len > 0) {
                if (!common::RecvAll(fd, req_body.data(), req.body_len)) {
                    std::fprintf(stderr,
                                 "[%s] failed to read request body (%u bytes) on worker_id=%d\n",
                                 ctx->state->static_info.node_id.c_str(),
                                 req.body_len,
                                 ctx->worker_id);
                    break;
                }
            }

            const auto msg_type = static_cast<common::MsgType>(req.msg_type);
            if (msg_type != common::MsgType::InferRequest) {
                std::fprintf(stderr,
                             "[%s] worker received unsupported msg_type=%u on worker_id=%d\n",
                             ctx->state->static_info.node_id.c_str(),
                             static_cast<unsigned>(req.msg_type),
                             ctx->worker_id);
                break;
            }

            if (!HandleInferRequest(fd, ctx, req, req_body)) {
                break;
            }
        }

        ::close(fd);
    }

    ::close(listen_fd);
}
