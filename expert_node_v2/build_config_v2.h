#pragma once

// -----------------------------------------------------------------------------
// Build-time feature switches.
//
// This header centralizes compile-time feature detection and optional toggles
// for expert_node_v2.
// -----------------------------------------------------------------------------

// User / build-system controlled switches.
// These can be injected from the build system via -D...
#ifndef EXPERT_NODE_V2_ENABLE_CUDA
#define EXPERT_NODE_V2_ENABLE_CUDA 1
#endif

#ifndef EXPERT_NODE_V2_ENABLE_ONEAPI
#define EXPERT_NODE_V2_ENABLE_ONEAPI 0
#endif

#ifndef EXPERT_NODE_V2_ENABLE_BF16
#define EXPERT_NODE_V2_ENABLE_BF16 1
#endif

// -----------------------------------------------------------------------------
// CUDA feature detection
// -----------------------------------------------------------------------------
#if defined(__CUDACC__) && (EXPERT_NODE_V2_ENABLE_CUDA == 1)
#define EXPERT_NODE_V2_HAS_CUDA 1
#else
#define EXPERT_NODE_V2_HAS_CUDA 0
#endif

#if (EXPERT_NODE_V2_HAS_CUDA == 1) && defined(CUDA_VERSION) && (CUDA_VERSION >= 11000) && (EXPERT_NODE_V2_ENABLE_BF16 == 1)
#define EXPERT_NODE_V2_HAS_CUDA_BF16 1
#else
#define EXPERT_NODE_V2_HAS_CUDA_BF16 0
#endif

// -----------------------------------------------------------------------------
// oneAPI feature detection
// -----------------------------------------------------------------------------
#if defined(SYCL_LANGUAGE_VERSION) && (EXPERT_NODE_V2_ENABLE_ONEAPI == 1)
#define EXPERT_NODE_V2_HAS_ONEAPI 1
#else
#define EXPERT_NODE_V2_HAS_ONEAPI 0
#endif
