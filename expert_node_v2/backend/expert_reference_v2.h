#pragma once

#include <vector>

#include "common/protocol.h"
#include "expert_node_v2/expert_format_v2.h"

bool RunFusedUpGateReferenceV2(
    const MatrixBlockScaleViewV2& w_up,
    const MatrixBlockScaleViewV2& w_gate,
    const void* x,
    common::ActivationDType input_dtype,
    std::vector<float>* out_h);

bool RunDownReferenceV2(
    const MatrixBlockScaleViewV2& w_down,
    const float* h,
    common::ActivationDType output_dtype,
    std::vector<std::uint8_t>* out_y_bytes);

bool RunExpertReferenceV2(
    const ExpertWeightsViewV2& weights,
    const void* x,
    common::ActivationDType input_dtype,
    common::ActivationDType output_dtype,
    std::vector<std::uint8_t>* out_y_bytes,
    std::vector<float>* out_h_debug);
