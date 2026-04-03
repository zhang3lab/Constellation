from __future__ import annotations

import time
from typing import Any

from .generation_runner import GenerationRunner
from .generation_types import (
    GreedySampling,
    SamplingConfig,
    TemperatureTopPSampling,
)


def map_finish_reason_to_chat_completions(reason: str | None) -> str | None:
    if reason in ("eos_token", "stop_token", "stop_string"):
        return "stop"
    if reason == "max_new_tokens":
        return "length"
    return None


def _normalize_stop_strings(stop: Any) -> list[str]:
    if stop is None:
        return []

    if isinstance(stop, str):
        return [stop]

    if isinstance(stop, list):
        out: list[str] = []
        for i, x in enumerate(stop):
            if not isinstance(x, str):
                raise TypeError(
                    f"request.stop[{i}] expected str, got {type(x).__name__}"
                )
            out.append(x)
        return out

    raise TypeError(
        f"request.stop expected str | list[str] | None, got {type(stop).__name__}"
    )


def _build_sampling_config_from_request(request: dict) -> SamplingConfig:
    if not isinstance(request, dict):
        raise TypeError(f"request expected dict, got {type(request).__name__}")

    if "max_completion_tokens" in request:
        max_new_tokens = int(request["max_completion_tokens"])
    elif "max_tokens" in request:
        max_new_tokens = int(request["max_tokens"])
    else:
        max_new_tokens = 16

    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be > 0, got {max_new_tokens}")

    temperature = float(request.get("temperature", 1.0))
    top_p = float(request.get("top_p", 1.0))
    stop_strings = _normalize_stop_strings(request.get("stop"))

    if temperature == 0.0:
        strategy = GreedySampling()
    else:
        strategy = TemperatureTopPSampling(
            temperature=temperature,
            top_p=top_p,
        )

    return SamplingConfig(
        strategy=strategy,
        max_new_tokens=max_new_tokens,
        stop_strings=stop_strings,
    )


def chat_completions_request_to_generation_inputs(
    session,
    request: dict,
) -> tuple[list[int], SamplingConfig]:
    if not isinstance(request, dict):
        raise TypeError(f"request expected dict, got {type(request).__name__}")

    stream = bool(request.get("stream", False))
    if stream:
        raise NotImplementedError("stream=True is not supported yet")

    n = int(request.get("n", 1))
    if n != 1:
        raise NotImplementedError("only n=1 is supported")

    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("request.messages must be a non-empty list")

    executor = session.full_model_executor
    if executor is None:
        raise RuntimeError("session.full_model_executor is not initialized")

    tokenizer = session.get_deepseek_model_loader().load_tokenizer()
    templated = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )

    if "input_ids" not in templated:
        raise RuntimeError("apply_chat_template(...) did not return input_ids")

    token_ids_obj = templated["input_ids"]
    if token_ids_obj and isinstance(token_ids_obj[0], list):
        input_ids = [int(x) for x in token_ids_obj[0]]
    else:
        input_ids = [int(x) for x in token_ids_obj]

    if not input_ids:
        raise RuntimeError("chat template produced empty input_ids")

    sampling_config = _build_sampling_config_from_request(request)
    return input_ids, sampling_config


def generation_result_to_chat_completions_response(
    result,
) -> dict:
    return {
        "id": f"chatcmpl_{result.request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result.model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.output_text,
                },
                "finish_reason": map_finish_reason_to_chat_completions(
                    result.finish_reason
                ),
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
        },
    }


def run_chat_completions(
    session,
    request: dict,
) -> dict:
    runner = GenerationRunner(session)

    input_ids, sampling_config = chat_completions_request_to_generation_inputs(
        session,
        request,
    )

    result = runner.generate(
        input_ids=input_ids,
        sampling_config=sampling_config,
    )

    return generation_result_to_chat_completions_response(result)
