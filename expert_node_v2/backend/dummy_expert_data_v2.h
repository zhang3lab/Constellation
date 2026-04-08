#pragma once

#include <cstdint>
#include <vector>

#include "common/protocol.h"
#include "expert_node_v2/expert_format_v2.h"

struct DummyExpertShapeV2 {
    int hidden_dim = 7168;
    int inter_dim = 2048;
    int row_block = 128;
    int col_block = 128;
};

void FillDummyExpertBundleV2(
    ExpertTensorBundleV2* bundle,
    const DummyExpertShapeV2& shape = {});

void FillDummyInputActivationV2(
    int hidden_dim,
    common::ActivationDType dtype,
    std::vector<float>* out_float,
    std::vector<std::uint16_t>* out_encoded);
