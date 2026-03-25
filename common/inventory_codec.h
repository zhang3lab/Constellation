#pragma once

#include <string>

#include "common/protocol.h"

namespace common {

// Encodes exactly one node inventory into the InventoryReply body format.
//
// Body layout:
//   u32 node_id_len
//   bytes node_id
//
//   u32 node_status
//   u32 num_gpus
//
//   repeat num_gpus times:
//     i32 local_gpu_id
//
//     u32 gpu_name_len
//     bytes gpu_name
//
//     u64 total_mem_bytes
//     u64 free_mem_bytes
//     u32 worker_port
//     u32 gpu_status
//
//     u32 gpu_vendor
//     u32 capability_flags
//
//     u32 arch_name_len
//     bytes arch_name
std::string EncodeInventoryReplyBody(
    const NodeInfo& node,
    NodeStatus node_status);

}  // namespace common
