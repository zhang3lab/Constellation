#include "common/infer_codec.h"

#include "common/protocol.h"

namespace common {

std::string EncodeInferRequestBody(const InferRequestMsg& msg) {
    std::string body;
    body.reserve(4 + 4 + 4 + 4 + 4 + 4 + msg.activation.size());

    AppendI32(&body, msg.expert_id);
    AppendI32(&body, msg.batch_size);
    AppendI32(&body, msg.hidden_dim);
    AppendU32(&body, static_cast<std::uint32_t>(msg.input_dtype));
    AppendU32(&body, static_cast<std::uint32_t>(msg.output_dtype));
    AppendU32(&body, static_cast<std::uint32_t>(msg.activation.size()));
    body.append(msg.activation);

    return body;
}

bool DecodeInferRequestBody(const std::string& body, InferRequestMsg* out) {
    if (out == nullptr) return false;

    std::size_t offset = 0;
    if (!ReadI32(body, &offset, &out->expert_id)) return false;
    if (!ReadI32(body, &offset, &out->batch_size)) return false;
    if (!ReadI32(body, &offset, &out->hidden_dim)) return false;

    std::uint32_t input_dtype_raw = 0;
    std::uint32_t output_dtype_raw = 0;
    if (!ReadU32(body, &offset, &input_dtype_raw)) return false;
    if (!ReadU32(body, &offset, &output_dtype_raw)) return false;

    out->input_dtype = static_cast<ActivationDType>(input_dtype_raw);
    out->output_dtype = static_cast<ActivationDType>(output_dtype_raw);

    std::uint32_t nbytes = 0;
    if (!ReadU32(body, &offset, &nbytes)) return false;
    if (!ReadBytes(body, &offset, nbytes, &out->activation)) return false;

    return offset == body.size();
}

std::string EncodeInferResponseBody(const InferResponseMsg& msg) {
    std::string body;
    body.reserve(4 + 4 + 4 + 4 + 4 + msg.output.size());

    AppendI32(&body, msg.status_code);
    AppendI32(&body, msg.batch_size);
    AppendI32(&body, msg.hidden_dim);
    AppendU32(&body, static_cast<std::uint32_t>(msg.output_dtype));
    AppendU32(&body, static_cast<std::uint32_t>(msg.output.size()));
    body.append(msg.output);

    return body;
}

bool DecodeInferResponseBody(const std::string& body, InferResponseMsg* out) {
    if (out == nullptr) return false;

    std::size_t offset = 0;
    if (!ReadI32(body, &offset, &out->status_code)) return false;
    if (!ReadI32(body, &offset, &out->batch_size)) return false;
    if (!ReadI32(body, &offset, &out->hidden_dim)) return false;

    std::uint32_t output_dtype_raw = 0;
    if (!ReadU32(body, &offset, &output_dtype_raw)) return false;
    out->output_dtype = static_cast<ActivationDType>(output_dtype_raw);

    std::uint32_t nbytes = 0;
    if (!ReadU32(body, &offset, &nbytes)) return false;
    if (!ReadBytes(body, &offset, nbytes, &out->output)) return false;

    return offset == body.size();
}

}  // namespace common
