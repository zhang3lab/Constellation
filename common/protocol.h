#pragma once

#include <cstdint>
#include <string>

namespace common {

// Wire format is little-endian.
static constexpr std::uint32_t kMagic = 0x45585054;  // "EXPT"
static constexpr std::uint16_t kVersion = 1;

enum class MsgType : std::uint16_t {
    InventoryRequest = 1,
    InventoryReply = 2,

    PlacementPlan = 10,

    LoadWeightsBegin = 20,
    LoadWeightsChunk = 21,
    LoadWeightsEnd = 22,
    LoadDone = 23,

    CommitServing = 30,

    InferRequest = 40,
    InferResponse = 41,

    HeartbeatRequest = 50,
    HeartbeatReply = 51,

    ErrorReport = 60,
};

struct MsgHeader {
    std::uint32_t magic;
    std::uint16_t version;
    std::uint16_t msg_type;
    std::uint32_t request_id;
    std::uint32_t body_len;
};

static_assert(sizeof(MsgHeader) == 16, "MsgHeader must be 16 bytes");

// Body-building helpers.
// These helpers must append values in little-endian wire order.
void AppendU16(std::string* out, std::uint16_t x);
void AppendU32(std::string* out, std::uint32_t x);
void AppendI32(std::string* out, std::int32_t x);
void AppendU64(std::string* out, std::uint64_t x);
void AppendString(std::string* out, const std::string& s);

}  // namespace common
