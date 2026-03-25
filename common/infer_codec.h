#pragma once

#include <cstdint>
#include <string>

namespace common {

struct InferRequestMsg {
    std::int32_t expert_id = -1;
    std::int32_t batch_size = 0;
    std::int32_t hidden_dim = 0;
    ActivationDType input_dtype = ActivationDType::Unknown;
    ActivationDType output_dtype = ActivationDType::Unknown;
    std::string activation;
};

struct InferResponseMsg {
    std::int32_t status_code = 0;
    std::int32_t batch_size = 0;
    std::int32_t hidden_dim = 0;
    ActivationDType output_dtype = ActivationDType::Unknown;
    std::string output;
};

std::string EncodeInferRequestBody(const InferRequestMsg& msg);
bool DecodeInferRequestBody(const std::string& body, InferRequestMsg* out);

std::string EncodeInferResponseBody(const InferResponseMsg& msg);
bool DecodeInferResponseBody(const std::string& body, InferResponseMsg* out);

}  // namespace common
