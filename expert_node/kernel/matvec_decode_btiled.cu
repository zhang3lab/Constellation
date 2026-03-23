// matvec_decode_btiled_clean_migrated.cu
//
// Small-batch decode-specialized row-major FP8 matvec.

#include "expert_node/kernel/expert.h"


#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <limits>


namespace {

__global__ void cast_float_to_half_kernel_local(const float* in, __half* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __float2half(in[idx]);
}

bool launch_cast_float_to_half_local(const float* d_in,
                                     __half* d_out,
                                     int n,
                                     cudaStream_t stream) {
    if (!d_in || !d_out || n < 0) return false;
    if (n == 0) return true;
    const int threads = 256;
    const int blocks = ceil_div_int(n, threads);
    cast_float_to_half_kernel_local<<<blocks, threads, 0, stream>>>(d_in, d_out, n);
    return cudaGetLastError() == cudaSuccess;
}

constexpr int WARP_SIZE = 32;
float g_lut_e4m3_host[256];
float g_lut_e5m2_host[256];
bool g_luts_built = false;

float* g_lut_e4m3_dev = nullptr;
float* g_lut_e5m2_dev = nullptr;
bool g_luts_uploaded = false;

inline bool valid_fp8_format(Fp8Format fmt) {
    return fmt == Fp8Format::E4M3 || fmt == Fp8Format::E5M2;
}

float decode_fp8_e4m3_host(uint8_t x) {
    const int sign = (x >> 7) & 0x1;
    const int exp  = (x >> 3) & 0xF;
    const int mant = x & 0x7;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        const float frac = static_cast<float>(mant) / 8.0f;
        const float val = std::ldexp(frac, -6);
        return sign ? -val : val;
    }

    if (exp == 0xF) {
        const float inf = std::numeric_limits<float>::infinity();
        const float nan = std::numeric_limits<float>::quiet_NaN();
        if (mant == 0) return sign ? -inf : inf;
        return nan;
    }

    const float frac = 1.0f + static_cast<float>(mant) / 8.0f;
    const float val = std::ldexp(frac, exp - 7);
    return sign ? -val : val;
}

float decode_fp8_e5m2_host(uint8_t x) {
    const int sign = (x >> 7) & 0x1;
    const int exp  = (x >> 2) & 0x1F;
    const int mant = x & 0x3;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        const float frac = static_cast<float>(mant) / 4.0f;
        const float val = std::ldexp(frac, -14);
        return sign ? -val : val;
    }

    if (exp == 0x1F) {
        const float inf = std::numeric_limits<float>::infinity();
        const float nan = std::numeric_limits<float>::quiet_NaN();
        if (mant == 0) return sign ? -inf : inf;
        return nan;
    }

    const float frac = 1.0f + static_cast<float>(mant) / 4.0f;
    const float val = std::ldexp(frac, exp - 15);
    return sign ? -val : val;
}

void build_host_luts() {
    if (g_luts_built) return;
    for (int i = 0; i < 256; ++i) {
        g_lut_e4m3_host[i] = decode_fp8_e4m3_host(static_cast<uint8_t>(i));
        g_lut_e5m2_host[i] = decode_fp8_e5m2_host(static_cast<uint8_t>(i));
    }
    g_luts_built = true;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ float load_x_value(const __half* x, int idx) {
    return __half2float(x[idx]);
}

__device__ __forceinline__ float load_x_value(const float* x, int idx) {
    return x[idx];
}

template<typename XType, int BATCH_TILE, int KCHUNK, int ROWS_PER_CTA>
__global__ void matvec_decode_btiled_kernel(
    const uint8_t* __restrict__ weights, // [M, K]
    const float* __restrict__ scales,    // [M, ceil_div(K, KCHUNK)]
    const XType* __restrict__ x,         // [B, K]
    float* __restrict__ y,               // [B, M]
    int M,
    int K,
    int B,
    Fp8Format fp8_format,
    const float* __restrict__ lut_e4m3,
    const float* __restrict__ lut_e5m2) {
    __shared__ float x_sh[BATCH_TILE][KCHUNK];
    __shared__ float lut_sh[256];

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    const int row = blockIdx.x * ROWS_PER_CTA + warp_id;
    const int batch_base = blockIdx.y * BATCH_TILE;
    const bool row_valid = (row < M);

    const float* lut_src = (fp8_format == Fp8Format::E5M2) ? lut_e5m2 : lut_e4m3;
    for (int i = tid; i < 256; i += ROWS_PER_CTA * WARP_SIZE) {
        lut_sh[i] = lut_src[i];
    }
    __syncthreads();

    float acc[BATCH_TILE];
#pragma unroll
    for (int b = 0; b < BATCH_TILE; ++b) acc[b] = 0.0f;

    const int num_k_chunks = ceil_div_int(K, KCHUNK);

    for (int k0 = 0; k0 < K; k0 += KCHUNK) {
        const int kchunk_id = k0 / KCHUNK;
        const float scale_val = row_valid ? scales[(size_t)row * num_k_chunks + kchunk_id] : 1.0f;

        for (int i = tid; i < KCHUNK; i += ROWS_PER_CTA * WARP_SIZE) {
            const int k = k0 + i;
#pragma unroll
            for (int b = 0; b < BATCH_TILE; ++b) {
                const int batch = batch_base + b;
                x_sh[b][i] = (batch < B && k < K) ? load_x_value(x, batch * K + k) : 0.0f;
            }
        }
        __syncthreads();

        const uint8_t* w_row = row_valid ? (weights + (size_t)row * K + k0) : nullptr;
        for (int kk = lane; kk < KCHUNK; kk += WARP_SIZE) {
            const int k = k0 + kk;
            if (row_valid && k < K) {
                const float w = lut_sh[w_row[kk]] * scale_val;
#pragma unroll
                for (int b = 0; b < BATCH_TILE; ++b) {
                    acc[b] += w * x_sh[b][kk];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int b = 0; b < BATCH_TILE; ++b) {
        acc[b] = warp_sum(acc[b]);
    }

    if (row_valid && lane == 0) {
#pragma unroll
        for (int b = 0; b < BATCH_TILE; ++b) {
            const int batch = batch_base + b;
            if (batch < B) {
                y[(size_t)batch * M + row] = acc[b];
            }
        }
    }
}

template<typename XType>
bool launch_dispatch_impl(
    const PackedRowMajorMatrix& W,
    const XType* d_x,
    float* d_y,
    const MlpShape& shape,
    cudaStream_t stream,
    const float* d_lut_e4m3,
    const float* d_lut_e5m2) {
    if (!W.weights || !W.scales || !d_x || !d_y) {
        std::fprintf(stderr, "launch_matvec_decode: null pointer\n");
        return false;
    }
    if (W.rows <= 0 || W.cols <= 0 || shape.num_tokens <= 0) {
        std::fprintf(stderr, "launch_matvec_decode: invalid dims\n");
        return false;
    }
    if (!valid_fp8_format(W.fp8_format)) {
        std::fprintf(stderr, "launch_matvec_decode: bad fp8_format=%d\n", fp8_format_to_int(W.fp8_format));
        return false;
    }
    if (shape.hidden_dim != W.cols) {
        std::fprintf(stderr,
                     "launch_matvec_decode: shape.hidden_dim=%d must equal W.cols=%d\n",
                     shape.hidden_dim, W.cols);
        return false;
    }
    if (shape.k_chunk != W.k_chunk) {
        std::fprintf(stderr,
                     "launch_matvec_decode: shape.k_chunk=%d must equal W.k_chunk=%d\n",
                     shape.k_chunk, W.k_chunk);
        return false;
    }
    if (W.num_k_chunks != 0 && W.num_k_chunks != ceil_div_int(W.cols, W.k_chunk)) {
        std::fprintf(stderr,
                     "launch_matvec_decode: W.num_k_chunks=%d inconsistent with cols=%d k_chunk=%d\n",
                     W.num_k_chunks, W.cols, W.k_chunk);
        return false;
    }

    const int rpc = shape.rows_per_cta;
    const int kch = shape.k_chunk;
    const int num_tokens = shape.num_tokens;

    if (!(rpc == 4 || rpc == 8 || rpc == 16)) {
        std::fprintf(stderr,
                     "launch_matvec_decode: unsupported rows_per_cta=%d (supported: 4,8,16)\n",
                     rpc);
        return false;
    }
    if (!(kch == 256 || kch == 512 || kch == 1024)) {
        std::fprintf(stderr,
                     "launch_matvec_decode: unsupported k_chunk=%d (supported: 256,512,1024)\n",
                     kch);
        return false;
    }

    const int batch_tile = (num_tokens >= 8) ? 8 : ((num_tokens >= 4) ? 4 : ((num_tokens >= 2) ? 2 : 1));
    dim3 block(rpc * WARP_SIZE, 1, 1);
    dim3 grid(ceil_div_int(W.rows, rpc), ceil_div_int(num_tokens, batch_tile), 1);

    cudaError_t err = cudaSuccess;

#define LAUNCH_CASE(BT, KC, RPC)                                                                   \
    do {                                                                                           \
        if (batch_tile == (BT) && kch == (KC) && rpc == (RPC)) {                                  \
            matvec_decode_btiled_kernel<XType, (BT), (KC), (RPC)>                                 \
                <<<grid, block, 0, stream>>>(                                                      \
                    W.weights, W.scales, d_x, d_y,                                                 \
                    W.rows, W.cols, num_tokens, W.fp8_format, d_lut_e4m3, d_lut_e5m2);            \
            err = cudaGetLastError();                                                              \
            if (err != cudaSuccess) {                                                              \
                std::fprintf(stderr, "matvec_decode_btiled launch failed: %s\n",                 \
                             cudaGetErrorString(err));                                             \
                return false;                                                                      \
            }                                                                                      \
            return true;                                                                           \
        }                                                                                          \
    } while (0)

    LAUNCH_CASE(1, 256, 4);  LAUNCH_CASE(1, 256, 8);  LAUNCH_CASE(1, 256, 16);
    LAUNCH_CASE(1, 512, 4);  LAUNCH_CASE(1, 512, 8);  LAUNCH_CASE(1, 512, 16);
    LAUNCH_CASE(1, 1024, 4); LAUNCH_CASE(1, 1024, 8); LAUNCH_CASE(1, 1024, 16);

    LAUNCH_CASE(2, 256, 4);  LAUNCH_CASE(2, 256, 8);  LAUNCH_CASE(2, 256, 16);
    LAUNCH_CASE(2, 512, 4);  LAUNCH_CASE(2, 512, 8);  LAUNCH_CASE(2, 512, 16);
    LAUNCH_CASE(2, 1024, 4); LAUNCH_CASE(2, 1024, 8); LAUNCH_CASE(2, 1024, 16);

    LAUNCH_CASE(4, 256, 4);  LAUNCH_CASE(4, 256, 8);  LAUNCH_CASE(4, 256, 16);
    LAUNCH_CASE(4, 512, 4);  LAUNCH_CASE(4, 512, 8);  LAUNCH_CASE(4, 512, 16);
    LAUNCH_CASE(4, 1024, 4); LAUNCH_CASE(4, 1024, 8); LAUNCH_CASE(4, 1024, 16);

    LAUNCH_CASE(8, 256, 4);  LAUNCH_CASE(8, 256, 8);  LAUNCH_CASE(8, 256, 16);
    LAUNCH_CASE(8, 512, 4);  LAUNCH_CASE(8, 512, 8);  LAUNCH_CASE(8, 512, 16);
    LAUNCH_CASE(8, 1024, 4); LAUNCH_CASE(8, 1024, 8); LAUNCH_CASE(8, 1024, 16);

#undef LAUNCH_CASE

    std::fprintf(stderr, "launch_matvec_decode_btiled: unreachable dispatch case\n");
    return false;
}

} // namespace

bool initialize(cudaStream_t stream) {
    build_host_luts();

    if (g_luts_uploaded) return true;

    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&g_lut_e4m3_dev), 256 * sizeof(float));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "initialize: cudaMalloc g_lut_e4m3_dev failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&g_lut_e5m2_dev), 256 * sizeof(float));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "initialize: cudaMalloc g_lut_e5m2_dev failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(g_lut_e4m3_dev);
        g_lut_e4m3_dev = nullptr;
        return false;
    }

    err = cudaMemcpyAsync(g_lut_e4m3_dev, g_lut_e4m3_host, 256 * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "initialize: memcpy g_lut_e4m3_dev failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(g_lut_e4m3_dev);
        cudaFree(g_lut_e5m2_dev);
        g_lut_e4m3_dev = nullptr;
        g_lut_e5m2_dev = nullptr;
        return false;
    }

    err = cudaMemcpyAsync(g_lut_e5m2_dev, g_lut_e5m2_host, 256 * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "initialize: memcpy g_lut_e5m2_dev failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(g_lut_e4m3_dev);
        cudaFree(g_lut_e5m2_dev);
        g_lut_e4m3_dev = nullptr;
        g_lut_e5m2_dev = nullptr;
        return false;
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "initialize: stream sync failed: %s\n", cudaGetErrorString(err));
        cudaFree(g_lut_e4m3_dev);
        cudaFree(g_lut_e5m2_dev);
        g_lut_e4m3_dev = nullptr;
        g_lut_e5m2_dev = nullptr;
        return false;
    }

    g_luts_uploaded = true;
    return true;
}

bool launch_matvec_decode_from_float(
    const PackedRowMajorMatrix& W,
    const float* d_x,
    float* d_y,
    const MlpShape& shape,
    cudaStream_t stream) {
    if (!initialize(stream)) return false;
    return launch_dispatch_impl(
        W, d_x, d_y, shape, stream,
        g_lut_e4m3_dev, g_lut_e5m2_dev);
}

bool launch_matvec_decode(
    const PackedRowMajorMatrix& W,
    const __half* d_x,
    __half* d_y,
    const MlpShape& shape,
    cudaStream_t stream) {
    if (!initialize(stream)) return false;
    if (!d_y) {
        std::fprintf(stderr, "launch_matvec_decode: null output pointer\n");
        return false;
    }

    float* d_y_float = nullptr;
    const size_t numel = static_cast<size_t>(shape.num_tokens) * static_cast<size_t>(W.rows);
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_y_float), numel * sizeof(float));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "launch_matvec_decode: cudaMalloc temp failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }

    const bool ok = launch_dispatch_impl(
        W, d_x, d_y_float, shape, stream,
        g_lut_e4m3_dev, g_lut_e5m2_dev);
    if (!ok) {
        cudaFree(d_y_float);
        return false;
    }

    if (!launch_cast_float_to_half_local(d_y_float, d_y, static_cast<int>(numel), stream)) {
        cudaFree(d_y_float);
        return false;
    }

    err = cudaFree(d_y_float);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "launch_matvec_decode: cudaFree temp failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }

    return true;
}
