#pragma once

#include <cstdint>

#include "expert_node_v2/expert_format_v2.h"

float DecodeFp8ByteV2(Fp8Format fmt, std::uint8_t v);

const float* GetHostFp8LutV2(Fp8Format fmt);
