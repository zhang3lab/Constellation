#pragma once

#include "expert_node_v2/expert_format_v2.h"

bool BuildMatrixBlockScaleViewV2(
    const HostTensorV2& weight_ht,
    const HostTensorV2& scale_ht,
    int rows,
    int cols,
    MatrixBlockScaleViewV2* out);

bool BuildExpertWeightsViewV2(
    const ExpertTensorBundleV2& bundle,
    ExpertWeightsViewV2* out);
