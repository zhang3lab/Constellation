#pragma once

#include <string>

#include "common/types.h"

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
//     u32 gpu_uid_len
//     bytes gpu_uid
//
//     i32 local_gpu_id
//
//     u32 gpu_name_len
//     bytes gpu_name
//
//     u64 total_mem_bytes
//     u64 free_mem_bytes
//     u32 worker_port
//     u32 gpu_status
std::string EncodeInventoryReplyBody(const NodeInfo& node);

}  // namespace common
