#include "expert.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace expert {
namespace {

constexpr int kDefaultMaxTokens = 256;

enum ResponseStatus : uint32_t {
    STATUS_OK = 0,
    STATUS_BAD_MAGIC = 1,
    STATUS_EXPERT_NOT_LOADED = 2,
    STATUS_BAD_DTYPE = 3,
    STATUS_BAD_HIDDEN_DIM = 4,
    STATUS_BAD_NUM_TOKENS = 5,
    STATUS_BUFFER_TOO_SMALL = 6,
    STATUS_BUFFER_SHAPE_MISMATCH = 7,
    STATUS_IO_ERROR = 8,
    STATUS_CUDA_ERROR = 9,
};

struct RuntimeState {
    int device_id = 0;
    bool initialized = false;
    std::unordered_map<int, ExpertWeights> experts;

    int common_hidden_dim = 0;
    int common_inter_dim = 0;
    bool common_shape_valid = false;
};

RuntimeState g_runtime;

bool read_full(int fd, void* buf, size_t bytes) {
    char* p = static_cast<char*>(buf);
    size_t done = 0;
    while (done < bytes) {
        const ssize_t n = ::recv(fd, p + done, bytes - done, 0);
        if (n == 0) return false;
        if (n < 0) {
            if (errno == EINTR) continue;
            std::fprintf(stderr, "recv failed: %s\n", std::strerror(errno));
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

bool write_full(int fd, const void* buf, size_t bytes) {
    const char* p = static_cast<const char*>(buf);
    size_t done = 0;
    while (done < bytes) {
        const ssize_t n = ::send(fd, p + done, bytes - done, 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            std::fprintf(stderr, "send failed: %s\n", std::strerror(errno));
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

bool send_error_response(
    int client_fd,
    const RequestHeader* req,
    uint32_t status) {
    ResponseHeader resp{};
    resp.magic = kResponseMagic;
    resp.model_expert_id = (req != nullptr) ? req->model_expert_id : 0;
    resp.num_tokens = (req != nullptr) ? req->num_tokens : 0;
    resp.hidden_dim = (req != nullptr) ? req->hidden_dim : 0;
    resp.dtype = (req != nullptr) ? req->dtype : DTYPE_FP16;
    resp.status = status;
    resp.request_id = (req != nullptr) ? req->request_id : 0;
    return write_full(client_fd, &resp, sizeof(resp));
}

bool alloc_quant_matrix_device(const HostQuantMatrix& src, QuantMatrix* dst) {
    if (dst == nullptr) return false;
    if (src.data == nullptr) {
        std::fprintf(stderr, "HostQuantMatrix.data is null\n");
        return false;
    }
    if (src.rows <= 0 || src.cols <= 0 || src.group_size <= 0) {
        std::fprintf(stderr,
                     "invalid HostQuantMatrix shape rows=%d cols=%d group_size=%d\n",
                     src.rows, src.cols, src.group_size);
        return false;
    }

    const size_t data_elems =
        static_cast<size_t>(src.rows) * static_cast<size_t>(src.cols);
    const size_t data_bytes = data_elems * sizeof(uint8_t);

    uint8_t* d_data = nullptr;
    cudaError_t err = cudaMalloc(&d_data, data_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc(matrix.data) failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    err = cudaMemcpy(d_data, src.data, data_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy(matrix.data) failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return false;
    }

    float* d_scales = nullptr;
    if (src.scales != nullptr) {
        const int groups_per_row = (src.cols + src.group_size - 1) / src.group_size;
        const size_t num_scales =
            static_cast<size_t>(src.rows) * static_cast<size_t>(groups_per_row);
        const size_t scale_bytes = num_scales * sizeof(float);

        err = cudaMalloc(&d_scales, scale_bytes);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "cudaMalloc(matrix.scales) failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_data);
            return false;
        }

        err = cudaMemcpy(d_scales, src.scales, scale_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "cudaMemcpy(matrix.scales) failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_scales);
            cudaFree(d_data);
            return false;
        }
    }

    dst->data = d_data;
    dst->scales = d_scales;
    dst->rows = src.rows;
    dst->cols = src.cols;
    dst->group_size = src.group_size;
    dst->fp8_format = src.fp8_format;
    return true;
}

void free_quant_matrix_device(QuantMatrix* m) {
    if (m == nullptr) return;
    if (m->data != nullptr) {
        cudaFree(m->data);
        m->data = nullptr;
    }
    if (m->scales != nullptr) {
        cudaFree(m->scales);
        m->scales = nullptr;
    }
    m->rows = 0;
    m->cols = 0;
    m->group_size = 64;
}

bool check_host_weights(const HostExpertWeights& w) {
    if (w.hidden_dim <= 0 || w.inter_dim <= 0) {
        std::fprintf(stderr, "invalid host expert dims hidden=%d inter=%d\n",
                     w.hidden_dim, w.inter_dim);
        return false;
    }

    const bool ok_up =
        w.w_up.data != nullptr &&
        w.w_up.rows == w.inter_dim &&
        w.w_up.cols == w.hidden_dim &&
        w.w_up.group_size > 0;

    const bool ok_gate =
        w.w_gate.data != nullptr &&
        w.w_gate.rows == w.inter_dim &&
        w.w_gate.cols == w.hidden_dim &&
        w.w_gate.group_size > 0;

    const bool ok_down =
        w.w_down.data != nullptr &&
        w.w_down.rows == w.hidden_dim &&
        w.w_down.cols == w.inter_dim &&
        w.w_down.group_size > 0;

    if (!ok_up) {
        std::fprintf(stderr, "invalid w_up shape/data\n");
        return false;
    }
    if (!ok_gate) {
        std::fprintf(stderr, "invalid w_gate shape/data\n");
        return false;
    }
    if (!ok_down) {
        std::fprintf(stderr, "invalid w_down shape/data\n");
        return false;
    }

    return true;
}

bool update_common_shape_from_weights(const HostExpertWeights& w) {
    if (!g_runtime.common_shape_valid) {
        g_runtime.common_hidden_dim = w.hidden_dim;
        g_runtime.common_inter_dim = w.inter_dim;
        g_runtime.common_shape_valid = true;
        return true;
    }

    if (g_runtime.common_hidden_dim != w.hidden_dim ||
        g_runtime.common_inter_dim != w.inter_dim) {
        std::fprintf(stderr,
                     "expert shape mismatch: existing hidden=%d inter=%d, new hidden=%d inter=%d\n",
                     g_runtime.common_hidden_dim,
                     g_runtime.common_inter_dim,
                     w.hidden_dim,
                     w.inter_dim);
        return false;
    }

    return true;
}

bool ensure_weights_match_request(
    const ExpertWeights& w,
    const RequestHeader& req,
    uint32_t* status) {
    if (req.dtype != DTYPE_FP16) {
        std::fprintf(stderr, "unsupported dtype=%u\n", req.dtype);
        *status = STATUS_BAD_DTYPE;
        return false;
    }
    if (static_cast<int>(req.hidden_dim) != w.hidden_dim) {
        std::fprintf(stderr, "request hidden_dim=%u mismatches weight hidden_dim=%d\n",
                     req.hidden_dim, w.hidden_dim);
        *status = STATUS_BAD_HIDDEN_DIM;
        return false;
    }
    *status = STATUS_OK;
    return true;
}

bool ensure_buffer_shape(
    DeviceBuffers* buffers,
    int hidden_dim,
    int inter_dim) {
    if (buffers->d_input == nullptr ||
        buffers->d_output == nullptr ||
        buffers->d_fused == nullptr) {
        return false;
    }
    return buffers->hidden_dim == hidden_dim && buffers->inter_dim == inter_dim;
}

bool handle_one_request(
    int client_fd,
    uint32_t server_id,
    DeviceBuffers* buffers,
    HostBuffers* host_buffers,
    cudaStream_t stream) {
    RequestHeader req{};
    if (!read_full(client_fd, &req, sizeof(req))) {
        return false;
    }

    if (req.magic != kRequestMagic) {
        std::fprintf(stderr, "[server %u] bad request magic: 0x%x\n",
                     server_id, req.magic);
        return false;
    }

    const ExpertWeights* weights = get_expert_weights(static_cast<int>(req.model_expert_id));
    if (weights == nullptr) {
        std::fprintf(stderr, "[server %u] model_expert_id=%u not loaded\n",
                     server_id, req.model_expert_id);
        (void)send_error_response(client_fd, &req, STATUS_EXPERT_NOT_LOADED);
        return false;
    }

    uint32_t status = STATUS_OK;
    if (!ensure_weights_match_request(*weights, req, &status)) {
        (void)send_error_response(client_fd, &req, status);
        return false;
    }

    if (req.num_tokens == 0) {
        std::fprintf(stderr, "[server %u] num_tokens=0\n", server_id);
        (void)send_error_response(client_fd, &req, STATUS_BAD_NUM_TOKENS);
        return false;
    }

    if (static_cast<int>(req.num_tokens) > buffers->max_tokens) {
        std::fprintf(stderr,
                     "[server %u] num_tokens=%u exceeds max_tokens=%d\n",
                     server_id, req.num_tokens, buffers->max_tokens);
        (void)send_error_response(client_fd, &req, STATUS_BUFFER_TOO_SMALL);
        return false;
    }

    if (!ensure_buffer_shape(buffers, weights->hidden_dim, weights->inter_dim)) {
        std::fprintf(stderr,
                     "[server %u] device buffer dims mismatch buffers(hidden=%d inter=%d) "
                     "weights(hidden=%d inter=%d)\n",
                     server_id,
                     buffers->hidden_dim,
                     buffers->inter_dim,
                     weights->hidden_dim,
                     weights->inter_dim);
        (void)send_error_response(client_fd, &req, STATUS_BUFFER_SHAPE_MISMATCH);
        return false;
    }

    if (host_buffers == nullptr ||
        host_buffers->h_input == nullptr ||
        host_buffers->h_output == nullptr ||
        static_cast<int>(req.num_tokens) > host_buffers->max_tokens ||
        static_cast<int>(req.hidden_dim) != host_buffers->hidden_dim) {
        std::fprintf(stderr,
                     "[server %u] host buffer mismatch host(max_tokens=%d hidden_dim=%d) "
                     "req(num_tokens=%u hidden_dim=%u)\n",
                     server_id,
                     host_buffers ? host_buffers->max_tokens : -1,
                     host_buffers ? host_buffers->hidden_dim : -1,
                     req.num_tokens,
                     req.hidden_dim);
        (void)send_error_response(client_fd, &req, STATUS_BUFFER_TOO_SMALL);
        return false;
    }

    const size_t io_elems =
        static_cast<size_t>(req.num_tokens) * static_cast<size_t>(req.hidden_dim);
    const size_t io_bytes = io_elems * sizeof(half);

    if (!read_full(client_fd, host_buffers->h_input, io_bytes)) {
        return false;
    }

    cudaError_t err = cudaMemcpyAsync(
        buffers->d_input,
        host_buffers->h_input,
        io_bytes,
        cudaMemcpyHostToDevice,
        stream);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpyAsync H2D input failed: %s\n", cudaGetErrorString(err));
        (void)send_error_response(client_fd, &req, STATUS_CUDA_ERROR);
        return false;
    }

    if (!launch_expert_mlp(
            *weights,
            buffers->d_input,
            buffers->d_output,
            buffers->d_fused,
            static_cast<int>(req.num_tokens),
            stream)) {
        std::fprintf(stderr, "launch_expert_mlp failed\n");
        (void)send_error_response(client_fd, &req, STATUS_CUDA_ERROR);
        return false;
    }

    err = cudaMemcpyAsync(
        host_buffers->h_output,
        buffers->d_output,
        io_bytes,
        cudaMemcpyDeviceToHost,
        stream);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpyAsync D2H output failed: %s\n", cudaGetErrorString(err));
        (void)send_error_response(client_fd, &req, STATUS_CUDA_ERROR);
        return false;
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaStreamSynchronize failed: %s\n", cudaGetErrorString(err));
        (void)send_error_response(client_fd, &req, STATUS_CUDA_ERROR);
        return false;
    }

    ResponseHeader resp{};
    resp.magic = kResponseMagic;
    resp.model_expert_id = req.model_expert_id;
    resp.num_tokens = req.num_tokens;
    resp.hidden_dim = req.hidden_dim;
    resp.dtype = req.dtype;
    resp.status = STATUS_OK;
    resp.request_id = req.request_id;

    if (!write_full(client_fd, &resp, sizeof(resp))) {
        return false;
    }
    if (!write_full(client_fd, host_buffers->h_output, io_bytes)) {
        return false;
    }

    return true;
}

}  // namespace

bool allocate_host_buffers(
    HostBuffers* buffers,
    int max_tokens,
    int hidden_dim) {
    if (buffers == nullptr) return false;
    if (max_tokens <= 0 || hidden_dim <= 0) {
        std::fprintf(stderr, "invalid host buffer dims max_tokens=%d hidden_dim=%d\n",
                     max_tokens, hidden_dim);
        return false;
    }

    free_host_buffers(buffers);

    const size_t io_elems =
        static_cast<size_t>(max_tokens) * static_cast<size_t>(hidden_dim);
    const size_t io_bytes = io_elems * sizeof(half);

    cudaError_t err = cudaMallocHost(&buffers->h_input, io_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMallocHost(h_input) failed: %s\n", cudaGetErrorString(err));
        free_host_buffers(buffers);
        return false;
    }

    err = cudaMallocHost(&buffers->h_output, io_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMallocHost(h_output) failed: %s\n", cudaGetErrorString(err));
        free_host_buffers(buffers);
        return false;
    }

    buffers->max_tokens = max_tokens;
    buffers->hidden_dim = hidden_dim;
    return true;
}

void free_host_buffers(HostBuffers* buffers) {
    if (buffers == nullptr) return;

    if (buffers->h_input != nullptr) {
        cudaFreeHost(buffers->h_input);
        buffers->h_input = nullptr;
    }
    if (buffers->h_output != nullptr) {
        cudaFreeHost(buffers->h_output);
        buffers->h_output = nullptr;
    }

    buffers->max_tokens = 0;
    buffers->hidden_dim = 0;
}

bool init_expert_runtime(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaSetDevice(%d) failed: %s\n",
                     device_id, cudaGetErrorString(err));
        return false;
    }

    g_runtime.device_id = device_id;
    g_runtime.initialized = true;
    return true;
}

bool allocate_device_buffers(
    DeviceBuffers* buffers,
    int max_tokens,
    int hidden_dim,
    int inter_dim) {
    if (buffers == nullptr) return false;
    if (max_tokens <= 0 || hidden_dim <= 0 || inter_dim <= 0) {
        std::fprintf(stderr, "invalid buffer dims max_tokens=%d hidden_dim=%d inter_dim=%d\n",
                     max_tokens, hidden_dim, inter_dim);
        return false;
    }

    free_device_buffers(buffers);

    const size_t io_elems =
        static_cast<size_t>(max_tokens) * static_cast<size_t>(hidden_dim);
    const size_t io_bytes = io_elems * sizeof(half);

    const size_t fused_elems =
        static_cast<size_t>(max_tokens) * static_cast<size_t>(inter_dim);
    const size_t fused_bytes = fused_elems * sizeof(float);

    cudaError_t err = cudaMalloc(&buffers->d_input, io_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc(d_input) failed: %s\n", cudaGetErrorString(err));
        free_device_buffers(buffers);
        return false;
    }

    err = cudaMalloc(&buffers->d_output, io_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc(d_output) failed: %s\n", cudaGetErrorString(err));
        free_device_buffers(buffers);
        return false;
    }

    err = cudaMalloc(&buffers->d_fused, fused_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc(d_fused) failed: %s\n", cudaGetErrorString(err));
        free_device_buffers(buffers);
        return false;
    }

    buffers->max_tokens = max_tokens;
    buffers->hidden_dim = hidden_dim;
    buffers->inter_dim = inter_dim;
    return true;
}

void free_device_buffers(DeviceBuffers* buffers) {
    if (buffers == nullptr) return;

    if (buffers->d_input != nullptr) {
        cudaFree(buffers->d_input);
        buffers->d_input = nullptr;
    }
    if (buffers->d_output != nullptr) {
        cudaFree(buffers->d_output);
        buffers->d_output = nullptr;
    }
    if (buffers->d_fused != nullptr) {
        cudaFree(buffers->d_fused);
        buffers->d_fused = nullptr;
    }

    buffers->max_tokens = 0;
    buffers->hidden_dim = 0;
    buffers->inter_dim = 0;
}

bool load_expert_weights(
    int model_expert_id,
    const HostExpertWeights& host_weights) {
    if (!g_runtime.initialized) {
        std::fprintf(stderr, "runtime not initialized\n");
        return false;
    }

    if (!check_host_weights(host_weights)) {
        return false;
    }

    if (!update_common_shape_from_weights(host_weights)) {
        return false;
    }

    ExpertWeights dev{};
    dev.hidden_dim = host_weights.hidden_dim;
    dev.inter_dim = host_weights.inter_dim;

    if (!alloc_quant_matrix_device(host_weights.w_up, &dev.w_up)) {
        return false;
    }
    if (!alloc_quant_matrix_device(host_weights.w_gate, &dev.w_gate)) {
        free_quant_matrix_device(&dev.w_up);
        return false;
    }
    if (!alloc_quant_matrix_device(host_weights.w_down, &dev.w_down)) {
        free_quant_matrix_device(&dev.w_gate);
        free_quant_matrix_device(&dev.w_up);
        return false;
    }

    auto it = g_runtime.experts.find(model_expert_id);
    if (it != g_runtime.experts.end()) {
        free_quant_matrix_device(&it->second.w_up);
        free_quant_matrix_device(&it->second.w_gate);
        free_quant_matrix_device(&it->second.w_down);
        it->second = dev;
    } else {
        g_runtime.experts.emplace(model_expert_id, dev);
    }

    return true;
}

const ExpertWeights* get_expert_weights(int model_expert_id) {
    auto it = g_runtime.experts.find(model_expert_id);
    if (it == g_runtime.experts.end()) return nullptr;
    return &it->second;
}

bool serve_forever(uint32_t server_id, const char* host, int port) {
    if (!g_runtime.initialized) {
        std::fprintf(stderr, "runtime not initialized\n");
        return false;
    }
    if (host == nullptr) {
        std::fprintf(stderr, "host is null\n");
        return false;
    }
    if (g_runtime.experts.empty()) {
        std::fprintf(stderr, "[server %u] no experts loaded\n", server_id);
        return false;
    }
    if (!g_runtime.common_shape_valid) {
        std::fprintf(stderr, "[server %u] common expert shape not set\n", server_id);
        return false;
    }

    DeviceBuffers device_buffers{};
    if (!allocate_device_buffers(
            &device_buffers,
            kDefaultMaxTokens,
            g_runtime.common_hidden_dim,
            g_runtime.common_inter_dim)) {
        std::fprintf(stderr, "[server %u] allocate_device_buffers failed\n", server_id);
        return false;
    }

    HostBuffers host_buffers{};
    if (!allocate_host_buffers(
            &host_buffers,
            kDefaultMaxTokens,
            g_runtime.common_hidden_dim)) {
        std::fprintf(stderr, "[server %u] allocate_host_buffers failed\n", server_id);
        free_device_buffers(&device_buffers);
        return false;
    }

    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
        free_host_buffers(&host_buffers);
        free_device_buffers(&device_buffers);
        return false;
    }

    const int listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        std::fprintf(stderr, "socket failed: %s\n", std::strerror(errno));
        cudaStreamDestroy(stream);
        free_host_buffers(&host_buffers);
        free_device_buffers(&device_buffers);
        return false;
    }

    int opt = 1;
    if (::setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) != 0) {
        std::fprintf(stderr, "setsockopt(SO_REUSEADDR) failed: %s\n", std::strerror(errno));
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));

    if (::inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        std::fprintf(stderr, "inet_pton failed for host=%s\n", host);
        ::close(listen_fd);
        cudaStreamDestroy(stream);
        free_host_buffers(&host_buffers);
        free_device_buffers(&device_buffers);
        return false;
    }

    if (::bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::fprintf(stderr, "bind failed: %s\n", std::strerror(errno));
        ::close(listen_fd);
        cudaStreamDestroy(stream);
        free_host_buffers(&host_buffers);
        free_device_buffers(&device_buffers);
        return false;
    }

    if (::listen(listen_fd, 128) != 0) {
        std::fprintf(stderr, "listen failed: %s\n", std::strerror(errno));
        ::close(listen_fd);
        cudaStreamDestroy(stream);
        free_host_buffers(&host_buffers);
        free_device_buffers(&device_buffers);
        return false;
    }

    std::fprintf(stderr, "[server %u] listening on %s:%d\n", server_id, host, port);

    while (true) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        const int client_fd =
            ::accept(listen_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            std::fprintf(stderr, "accept failed: %s\n", std::strerror(errno));
            break;
        }

        char ipbuf[INET_ADDRSTRLEN] = {0};
        ::inet_ntop(AF_INET, &client_addr.sin_addr, ipbuf, sizeof(ipbuf));
        std::fprintf(stderr, "[server %u] accepted connection from %s:%u\n",
                     server_id, ipbuf, ntohs(client_addr.sin_port));

        while (true) {
            if (!handle_one_request(client_fd, server_id, &device_buffers, &host_buffers, stream)) {
                break;
            }
        }

        ::close(client_fd);
        std::fprintf(stderr, "[server %u] connection closed\n", server_id);
    }

    ::close(listen_fd);
    cudaStreamDestroy(stream);
    free_host_buffers(&host_buffers);
    free_device_buffers(&device_buffers);
    return false;
}

void shutdown_expert_runtime() {
    for (auto& kv : g_runtime.experts) {
        free_quant_matrix_device(&kv.second.w_up);
        free_quant_matrix_device(&kv.second.w_gate);
        free_quant_matrix_device(&kv.second.w_down);
    }
    g_runtime.experts.clear();
    g_runtime.common_hidden_dim = 0;
    g_runtime.common_inter_dim = 0;
    g_runtime.common_shape_valid = false;
    g_runtime.initialized = false;
}

}  // namespace expert
