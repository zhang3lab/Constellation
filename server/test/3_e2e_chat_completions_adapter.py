from __future__ import annotations

import argparse
import json

import torch

from server.chat_completions_adapter import (
    chat_completions_request_to_generation_inputs,
    generation_result_to_chat_completions_response,
)
from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.deepseek_full_model_executor import DeepseekFullModelExecutor
from server.generation_runner import GenerationRunner
from server.inference_session import InferenceSession


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/test/config.json")
    ap.add_argument("--system", type=str, default="You are a helpful assistant.")
    ap.add_argument("--prompt", type=str, default="Hello world")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    coord = Coordinator(cfg["nodes"])
    setup_control_plane(coord, cfg)

    kv_cache_cfg = cfg["kv_cache"]

    request = {
        "model": cfg["model"]["name"],
        "messages": [
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.prompt},
        ],
        "max_tokens": int(args.max_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "stream": False,
    }

    with InferenceSession(coord, cfg) as session:
        session.full_model_executor = DeepseekFullModelExecutor(session)

        session.ensure_full_model_runtime(
            tensor_cache_dir="tmp/non_moe_backbone_cache",
            split_layer=30,
            backbone_dtype=torch.bfloat16,
            kv_cache_cfg=kv_cache_cfg,
        )
        session.reset_full_model_kv_cache(kv_cache_cfg=kv_cache_cfg)

        tokenizer = session.get_deepseek_model_loader().load_tokenizer()
        runner = GenerationRunner(session)

        input_ids, sampling_config = chat_completions_request_to_generation_inputs(
            session,
            request,
        )

        result = runner.generate(
            input_ids=input_ids,
            sampling_config=sampling_config,
        )
        response = generation_result_to_chat_completions_response(result)

        print("[chat] request =")
        print(json.dumps(request, ensure_ascii=False, indent=2))
        print()

        print("[chat] templated input_ids =")
        print(input_ids)
        print()

        print("[chat] templated decoded prompt =")
        print(repr(tokenizer.decode(input_ids)))
        print(tokenizer.decode(input_ids))
        print()

        print("[chat] output_token_ids =")
        print(result.output_token_ids)
        print()

        if hasattr(tokenizer, "convert_ids_to_tokens"):
            print("[chat] output_tokens =")
            print(tokenizer.convert_ids_to_tokens(result.output_token_ids))
            print()

        print("[chat] output_text =")
        print(repr(result.output_text))
        print(result.output_text)
        print()

        print("[chat] response =")
        print(json.dumps(response, ensure_ascii=False, indent=2))
        print()

        if not isinstance(response, dict):
            raise TypeError(f"response expected dict, got {type(response).__name__}")

        if response.get("object") != "chat.completion":
            raise RuntimeError(
                f"response.object expected 'chat.completion', got {response.get('object')!r}"
            )

        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise RuntimeError("response.choices must be a single-element list")

        choice0 = choices[0]
        if not isinstance(choice0, dict):
            raise RuntimeError("response.choices[0] must be a dict")

        message = choice0.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("response.choices[0].message must be a dict")

        if message.get("role") != "assistant":
            raise RuntimeError(
                f"assistant role expected, got {message.get('role')!r}"
            )

        if not isinstance(message.get("content"), str):
            raise RuntimeError("response assistant content must be a string")

        usage = response.get("usage")
        if not isinstance(usage, dict):
            raise RuntimeError("response.usage must be a dict")

        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if key not in usage:
                raise RuntimeError(f"response.usage missing key {key!r}")
            if not isinstance(usage[key], int):
                raise RuntimeError(f"response.usage[{key!r}] must be int")

        print("[chat] chat_completions_adapter passed")


if __name__ == "__main__":
    main()
