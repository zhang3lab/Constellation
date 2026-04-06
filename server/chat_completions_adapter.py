from __future__ import annotations

import time
import uuid
from typing import Any

from server.generation_runtime import run_generation_from_input_ids
from server.generation_types import (
    GenerationResult,
    GenerationState,
    SamplingConfig,
    TemperatureTopPSampling,
)


def _require_dict(x: Any, name: str) -> dict:
    if not isinstance(x, dict):
        raise TypeError(f"{name} expected dict, got {type(x).__name__}")
    return x


def _require_list(x: Any, name: str) -> list:
    if not isinstance(x, list):
        raise TypeError(f"{name} expected list, got {type(x).__name__}")
    return x


def _get_model_loader(session):
    if not hasattr(session, "get_deepseek_model_loader"):
        raise RuntimeError("session.get_deepseek_model_loader is not available")
    loader = session.get_deepseek_model_loader()
    if loader is None:
        raise RuntimeError("session.get_deepseek_model_loader() returned None")
    return loader


def _get_tokenizer(session):
    model_loader = _get_model_loader(session)
    tokenizer = model_loader.load_tokenizer()
    if tokenizer is None:
        raise RuntimeError("model_loader.load_tokenizer() returned None")
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("tokenizer does not support apply_chat_template")
    return tokenizer


def _get_kv_cache_cfg(session) -> dict:
    cfg = getattr(session, "cfg", None)
    if not isinstance(cfg, dict):
        raise RuntimeError("session.cfg must be initialized")
    kv_cache_cfg = cfg.get("kv_cache")
    if not isinstance(kv_cache_cfg, dict):
        raise RuntimeError("session.cfg['kv_cache'] must be a dict")
    return kv_cache_cfg


def _get_run_cfg(session) -> dict:
    cfg = getattr(session, "cfg", None)
    if not isinstance(cfg, dict):
        raise RuntimeError("session.cfg must be initialized")
    run_cfg = cfg.get("run", {})
    if not isinstance(run_cfg, dict):
        raise RuntimeError("session.cfg['run'] must be a dict")
    return run_cfg


def _get_model_name(session, request: dict) -> str:
    model = request.get("model")
    if model is not None and not isinstance(model, str):
        raise TypeError(f"request['model'] expected str, got {type(model).__name__}")
    if isinstance(model, str) and model:
        return model

    executor = getattr(session, "full_model_executor", None)
    model_name = getattr(executor, "model_name", None) if executor is not None else None
    if isinstance(model_name, str) and model_name:
        return model_name

    return "deepseek-v3"


def _reject_unsupported_request_fields(request: dict) -> None:
    unsupported = [
        "stream",
        "n",
        "tools",
        "tool_choice",
        "response_format",
        "parallel_tool_calls",
        "audio",
        "modalities",
        "prediction",
        "reasoning_effort",
        "logprobs",
        "top_logprobs",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "user",
    ]
    for key in unsupported:
        if key in request:
            raise RuntimeError(f"unsupported request field: {key!r}")


def _normalize_messages(messages: Any) -> list[dict]:
    messages = _require_list(messages, "request['messages']")
    if not messages:
        raise RuntimeError("request['messages'] must be non-empty")

    allowed_roles = {"system", "user", "assistant"}

    out: list[dict] = []
    for i, msg in enumerate(messages):
        msg = _require_dict(msg, f"request['messages'][{i}]")

        allowed_keys = {"role", "content"}
        extra_keys = sorted(set(msg.keys()) - allowed_keys)
        if extra_keys:
            raise RuntimeError(
                f"request['messages'][{i}] has unsupported keys: {extra_keys}"
            )

        role = msg.get("role")
        if not isinstance(role, str) or not role:
            raise RuntimeError(f"request['messages'][{i}]['role'] must be a non-empty str")
        if role not in allowed_roles:
            raise RuntimeError(
                f"request['messages'][{i}]['role'] must be one of {sorted(allowed_roles)}, got {role!r}"
            )

        content = msg.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for j, part in enumerate(content):
                part = _require_dict(part, f"request['messages'][{i}]['content'][{j}]")

                allowed_part_keys = {"type", "text"}
                extra_part_keys = sorted(set(part.keys()) - allowed_part_keys)
                if extra_part_keys:
                    raise RuntimeError(
                        f"request['messages'][{i}]['content'][{j}] has unsupported keys: {extra_part_keys}"
                    )

                part_type = part.get("type")
                if part_type != "text":
                    raise RuntimeError(
                        f"request['messages'][{i}]['content'][{j}] only supports type='text', got {part_type!r}"
                    )

                text_part = part.get("text", "")
                if not isinstance(text_part, str):
                    raise RuntimeError(
                        f"request['messages'][{i}]['content'][{j}]['text'] must be str"
                    )
                parts.append(text_part)
            text = "".join(parts)
        else:
            raise RuntimeError(
                f"request['messages'][{i}]['content'] must be str or list, got {type(content).__name__}"
            )

        out.append(
            {
                "role": role,
                "content": text,
            }
        )
    return out


def _render_chat_messages_to_input_ids_and_prompt(
    session,
    messages: list[dict],
) -> tuple[list[int], str]:
    tokenizer = _get_tokenizer(session)

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(prompt_text, str):
        raise RuntimeError(
            f"tokenizer.apply_chat_template(..., tokenize=False) expected str, got {type(prompt_text).__name__}"
        )

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if not isinstance(input_ids, list) or not input_ids:
        raise RuntimeError(
            "tokenizer.apply_chat_template(..., tokenize=True) must return non-empty list[int]"
        )
    if not all(isinstance(x, int) for x in input_ids):
        raise RuntimeError(
            "tokenizer.apply_chat_template(..., tokenize=True) must return list[int]"
        )

    return [int(x) for x in input_ids], prompt_text


def _normalize_stop_strings(stop: Any) -> list[str]:
    if stop is None:
        return []

    if isinstance(stop, str):
        if stop == "":
            raise RuntimeError("request['stop'] must not be empty")
        return [stop]

    if isinstance(stop, list):
        out: list[str] = []
        for i, item in enumerate(stop):
            if not isinstance(item, str):
                raise TypeError(
                    f"request['stop'][{i}] expected str, got {type(item).__name__}"
                )
            if item == "":
                raise RuntimeError("request['stop'] must not contain empty strings")
            out.append(item)
        return out

    raise TypeError(
        f"request['stop'] expected str | list[str] | None, got {type(stop).__name__}"
    )


def _normalize_stop_to_token_sequences(session, stop: Any) -> list[list[int]]:
    tokenizer = _get_tokenizer(session)
    stop_strings = _normalize_stop_strings(stop)

    out: list[list[int]] = []
    for s in stop_strings:
        # NOTE:
        # We intentionally encode stop strings independently with the same tokenizer,
        # rather than rendering them through the full chat template context. This is
        # a simplified first-pass behavior and may differ from context-dependent
        # tokenization at string boundaries.
        token_ids = tokenizer.encode(s, add_special_tokens=False)
        if not isinstance(token_ids, list) or not token_ids:
            raise RuntimeError(f"stop string encoded to empty token sequence: {s!r}")
        if not all(isinstance(x, int) for x in token_ids):
            raise RuntimeError(
                "tokenizer.encode(stop_str, add_special_tokens=False) must return list[int]"
            )
        out.append([int(x) for x in token_ids])

    return out


def _resolve_max_new_tokens(request: dict) -> int:
    max_completion_tokens = request.get("max_completion_tokens")
    max_tokens = request.get("max_tokens")

    value = max_completion_tokens if max_completion_tokens is not None else max_tokens
    if value is None:
        return 16

    if not isinstance(value, int):
        raise TypeError(
            f"request max tokens field expected int, got {type(value).__name__}"
        )
    if value < 0:
        raise RuntimeError(f"max_new_tokens must be >= 0, got {value}")
    return int(value)


def _build_sampling_config(session, request: dict) -> SamplingConfig:
    max_new_tokens = _resolve_max_new_tokens(request)

    temperature = request.get("temperature", 1.0)
    top_p = request.get("top_p", 1.0)

    if temperature is None:
        temperature = 1.0
    if top_p is None:
        top_p = 1.0

    if not isinstance(temperature, (int, float)):
        raise TypeError(
            f"request['temperature'] expected int|float, got {type(temperature).__name__}"
        )
    if not isinstance(top_p, (int, float)):
        raise TypeError(f"request['top_p'] expected int|float, got {type(top_p).__name__}")

    temperature = float(temperature)
    top_p = float(top_p)

    if temperature <= 0.0:
        raise RuntimeError(f"temperature must be > 0, got {temperature}")
    if top_p <= 0.0 or top_p > 1.0:
        raise RuntimeError(f"top_p must be in (0, 1], got {top_p}")

    stop_token_sequences = _normalize_stop_to_token_sequences(session, request.get("stop"))

    return SamplingConfig(
        strategy=TemperatureTopPSampling(
            temperature=temperature,
            top_p=top_p,
        ),
        max_new_tokens=max_new_tokens,
        stop_token_sequences=stop_token_sequences,
    )


def _map_finish_reason(x: str | None) -> str | None:
    if x in ("eos_token", "stop_token"):
        return "stop"
    if x == "max_new_tokens":
        return "length"
    return x


def _build_choice(result) -> dict:
    return {
        "index": 0,
        "message": {
            "role": "assistant",
            "content": result.output_text,
        },
        "finish_reason": _map_finish_reason(result.finish_reason),
    }


def _build_usage(result) -> dict:
    return {
        "prompt_tokens": int(result.prompt_tokens),
        "completion_tokens": int(result.completion_tokens),
        "total_tokens": int(result.total_tokens),
    }


def run_chat_completions(
    session,
    *,
    request: dict,
    return_aux: bool = False,
) -> dict:
    request = _require_dict(request, "request")
    if not session.is_chat_runtime_ready():
        raise RuntimeError("chat runtime is not ready")

    _reject_unsupported_request_fields(request)

    messages = _normalize_messages(request.get("messages"))
    input_ids, prompt_text = _render_chat_messages_to_input_ids_and_prompt(session, messages)
    model_name = _get_model_name(session, request)
    sampling_config = _build_sampling_config(session, request)

    kv_cache_cfg = _get_kv_cache_cfg(session)
    session.reset_full_model_kv_cache(kv_cache_cfg=kv_cache_cfg)

    kv_cache = getattr(session, "page_attention_cache_managers", None)
    if kv_cache is None:
        raise RuntimeError("session.page_attention_cache_managers is not initialized")
    if not isinstance(kv_cache, dict):
        raise RuntimeError("session.page_attention_cache_managers must be a dict")

    run_cfg = _get_run_cfg(session)
    start_layer = int(run_cfg.get("start_layer", 0))
    end_layer = int(run_cfg.get("end_layer", 60))

    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    state = GenerationState(
        request_id=request_id,
        model_name=model_name,
        sampling_config=sampling_config,
    )

    result = run_generation_from_input_ids(
        session,
        state=state,
        input_ids=input_ids,
        start_layer=start_layer,
        end_layer=end_layer,
        kv_cache=kv_cache,
    )
    if not isinstance(result, GenerationResult):
        raise TypeError(
            f"run_generation_from_input_ids(...) expected GenerationResult, got {type(result).__name__}"
        )

    payload = {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            _build_choice(result),
        ],
        "usage": _build_usage(result),
    }

    if return_aux:
        return {
            "result": payload,
            "aux": {
                "prompt_text": prompt_text,
                "input_ids": [int(x) for x in input_ids],
            },
        }

    return {
        "result": payload,
    }
