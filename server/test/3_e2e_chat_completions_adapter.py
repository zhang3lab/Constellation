from __future__ import annotations

import argparse
import json

import torch

from server.chat_completions_adapter import run_chat_completions
from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.deepseek_full_model_executor import DeepseekFullModelExecutor
from server.inference_session import InferenceSession


def _require_dict(x, name: str) -> dict:
    if not isinstance(x, dict):
        raise TypeError(f"{name} expected dict, got {type(x).__name__}")
    return x


def _validate_result_payload(result_payload: dict) -> None:
    result_payload = _require_dict(result_payload, "result")

    if result_payload.get("object") != "chat.completion":
        raise RuntimeError(
            f"result['object'] expected 'chat.completion', got {result_payload.get('object')!r}"
        )

    result_id = result_payload.get("id")
    if not isinstance(result_id, str) or not result_id.startswith("chatcmpl-"):
        raise RuntimeError(f"result['id'] has invalid format: {result_id!r}")

    created = result_payload.get("created")
    if not isinstance(created, int):
        raise RuntimeError(f"result['created'] expected int, got {type(created).__name__}")

    model = result_payload.get("model")
    if not isinstance(model, str) or not model:
        raise RuntimeError(f"result['model'] expected non-empty str, got {model!r}")

    choices = result_payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("result['choices'] must be a list of length 1")

    choice0 = _require_dict(choices[0], "result['choices'][0]")
    if choice0.get("index") != 0:
        raise RuntimeError(f"result['choices'][0]['index'] expected 0, got {choice0.get('index')!r}")

    message = _require_dict(choice0.get("message"), "result['choices'][0]['message']")
    if message.get("role") != "assistant":
        raise RuntimeError(
            f"result['choices'][0]['message']['role'] expected 'assistant', got {message.get('role')!r}"
        )
    if not isinstance(message.get("content"), str):
        raise RuntimeError("result['choices'][0]['message']['content'] must be str")

    finish_reason = choice0.get("finish_reason")
    if finish_reason not in ("stop", "length", None):
        raise RuntimeError(f"unexpected finish_reason: {finish_reason!r}")

    usage = _require_dict(result_payload.get("usage"), "result['usage']")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        val = usage.get(key)
        if not isinstance(val, int):
            raise RuntimeError(f"result['usage']['{key}'] expected int, got {type(val).__name__}")

    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        raise RuntimeError(
            "usage total_tokens mismatch: "
            f"{usage['total_tokens']} != {usage['prompt_tokens']} + {usage['completion_tokens']}"
        )


def _validate_aux_payload(aux_payload: dict) -> None:
    aux_payload = _require_dict(aux_payload, "aux")

    prompt_text = aux_payload.get("prompt_text")
    if not isinstance(prompt_text, str) or not prompt_text:
        raise RuntimeError("aux['prompt_text'] must be a non-empty str")

    input_ids = aux_payload.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise RuntimeError("aux['input_ids'] must be torch.Tensor")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise RuntimeError(
            f"aux['input_ids'] expected shape [1, T], got {tuple(input_ids.shape)}"
        )
    if input_ids.numel() <= 0:
        raise RuntimeError("aux['input_ids'] must be non-empty")


def _jsonable(x):
    if isinstance(x, torch.Tensor):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/test/config.json")
    ap.add_argument("--message", type=str, default="Hello world")
    args = ap.parse_args()

    cfg = load_config(args.config)
    coord = Coordinator(cfg["nodes"], log_level=cfg["log_level"])
    setup_control_plane(coord, cfg)

    kv_cache_cfg = cfg["kv_cache"]

    with InferenceSession(coord, cfg) as session:
        session.full_model_executor = DeepseekFullModelExecutor(session)
        session.initialize_full_model_runtime(
            tensor_cache_dir="tmp/non_moe_backbone_cache",
            split_layer=30,
            backbone_dtype=torch.bfloat16,
            kv_cache_cfg=kv_cache_cfg,
        )

        if not session.is_chat_runtime_ready():
            raise RuntimeError("chat runtime is not ready")

        base_messages = [
            {
                "role": "user",
                "content": args.message,
            }
        ]

        request0 = {
            "model": "deepseek-v3",
            "messages": base_messages,
            "max_completion_tokens": 0,
            "temperature": 1.0,
            "top_p": 1.0,
        }
        response0 = run_chat_completions(
            session,
            request=request0,
            return_aux=False,
        )
        response0 = _require_dict(response0, "response0")
        if set(response0.keys()) != {"result"}:
            raise RuntimeError(f"response0 expected keys {{'result'}}, got {sorted(response0.keys())}")
        _validate_result_payload(response0["result"])

        result0 = response0["result"]
        choice0 = result0["choices"][0]
        usage0 = result0["usage"]

        if choice0["finish_reason"] != "length":
            raise RuntimeError(
                f"case0 expected finish_reason='length', got {choice0['finish_reason']!r}"
            )
        if usage0["completion_tokens"] != 0:
            raise RuntimeError(
                f"case0 expected completion_tokens=0, got {usage0['completion_tokens']}"
            )
        if choice0["message"]["content"] != "":
            raise RuntimeError(
                f"case0 expected empty assistant content, got {choice0['message']['content']!r}"
            )

        request1 = {
            "model": "deepseek-v3",
            "messages": base_messages,
            "max_completion_tokens": 1,
            "temperature": 1.0,
            "top_p": 1.0,
            "stop": ["[[STOP_A]]", "[[STOP_B]]"],
        }
        response1 = run_chat_completions(
            session,
            request=request1,
            return_aux=True,
        )
        response1 = _require_dict(response1, "response1")
        if set(response1.keys()) != {"result", "aux"}:
            raise RuntimeError(
                f"response1 expected keys {{'result', 'aux'}}, got {sorted(response1.keys())}"
            )

        _validate_result_payload(response1["result"])
        _validate_aux_payload(response1["aux"])

        result1 = response1["result"]
        choice1 = result1["choices"][0]
        usage1 = result1["usage"]
        aux1 = response1["aux"]

        if usage1["completion_tokens"] > 1:
            raise RuntimeError(
                f"case1 expected completion_tokens <= 1, got {usage1['completion_tokens']}"
            )

        if not isinstance(choice1["message"]["content"], str):
            raise RuntimeError("case1 assistant content must be str")

        if "Assistant" not in aux1["prompt_text"] and "assistant" not in aux1["prompt_text"]:
            print("[warn] aux.prompt_text does not visibly contain assistant tag; template may differ")

        request2 = {
            "model": "deepseek-v3",
            "messages": [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": [{"type": "text", "text": args.message}]},
            ],
            "max_tokens": 1,
            "temperature": 0.8,
            "top_p": 0.95,
            "stop": "[[SINGLE_STOP]]",
        }
        response2 = run_chat_completions(
            session,
            request=request2,
            return_aux=True,
        )
        response2 = _require_dict(response2, "response2")
        _validate_result_payload(response2["result"])
        _validate_aux_payload(response2["aux"])

        result2 = response2["result"]
        usage2 = result2["usage"]
        if usage2["completion_tokens"] > 1:
            raise RuntimeError(
                f"case2 expected completion_tokens <= 1, got {usage2['completion_tokens']}"
            )

        print("PASS: chat completions adapter e2e")
        print(f"message={args.message!r}")

        print("--- case max_completion_tokens=0, return_aux=False ---")
        print(json.dumps(_jsonable(response0), ensure_ascii=False, indent=2))

        print("--- case max_completion_tokens=1, stop=list[str], return_aux=True ---")
        print(json.dumps(_jsonable(response1), ensure_ascii=False, indent=2))

        print("--- case max_tokens fallback, content=list[text], stop=str, return_aux=True ---")
        print(json.dumps(_jsonable(response2), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
