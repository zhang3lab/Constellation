#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common/protocol.h"
#include "common/types.h"

namespace common {

// Encodes one node's resident expert inventory into the ResidentInventoryReply
// body format.
//
// Body layout:
//   u32 num_workers
//
//   repeat num_workers times:
//     i32 worker_id
//     u32 num_experts
//
//     repeat num_experts times:
//       i32 expert_id
bool EncodeResidentInventoryReplyBody(
    const std::vector<ResidentInventoryWorkerInfo>& workers,
    std::string* out);

}  // namespace common
