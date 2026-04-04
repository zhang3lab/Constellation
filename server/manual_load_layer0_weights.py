from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM


def load_index(model_dir: str) -> dict:
    with open(f"{model_dir}/model.safetensors.index.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_raw_tensor(model_dir: str, tensor_name: str, index: dict | None = None) -> torch.Tensor:
    if index is None:
        index = load_index(model_dir)
    weight_map = index["weight_map"]
    if tensor_name not in weight_map:
        raise KeyError(f"{tensor_name} not found in weight_map for {model_dir}")
    shard = weight_map[tensor_name]
    shard_path = f"{model_dir}/{shard}"
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        return f.get_tensor(tensor_name)


def get_attr_by_dotted_name(obj, dotted_name: str):
    cur = obj
    for part in dotted_name.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def copy_named_tensor_into_model(
    model,
    *,
    model_dir: str,
    tensor_name: str,
    index: dict | None = None,
) -> None:
    target = get_attr_by_dotted_name(model, tensor_name)
    if not isinstance(target, (torch.Tensor, torch.nn.Parameter)):
        raise TypeError(f"{tensor_name} is not tensor-like, got {type(target).__name__}")

    raw = load_raw_tensor(model_dir, tensor_name, index=index)

    with torch.no_grad():
        target.copy_(raw.to(device=target.device, dtype=target.dtype))


def load_model_emptyish(model_dir: str, device: str):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map=device,
    )
    model.eval()
    return model


def load_named_tensors_into_model(
    model,
    *,
    model_dir: str,
    tensor_names: Iterable[str],
) -> None:
    index = load_index(model_dir)
    for name in tensor_names:
        print(f"[manual-load] copy {name}")
        copy_named_tensor_into_model(
            model,
            model_dir=model_dir,
            tensor_name=name,
            index=index,
        )


def default_layer0_names() -> list[str]:
    return [
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.q_a_layernorm.weight",
        "model.layers.0.self_attn.q_b_proj.weight",
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.0.self_attn.kv_a_layernorm.weight",
        "model.layers.0.self_attn.kv_b_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.post_attention_layernorm.weight",
    ]


def compare_loaded_vs_raw(
    model,
    *,
    model_dir: str,
    tensor_name: str,
    index: dict | None = None,
) -> tuple[float, float]:
    target = get_attr_by_dotted_name(model, tensor_name)
    raw = load_raw_tensor(model_dir, tensor_name, index=index)
    loaded = target.detach().float().cpu()
    diff = (loaded - raw.float().cpu()).abs()
    return float(diff.max().item()), float(diff.mean().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--names-json", type=str, default=None)
    args = ap.parse_args()

    if args.names_json:
        with open(args.names_json, "r", encoding="utf-8") as f:
            tensor_names = json.load(f)
        if not isinstance(tensor_names, list) or not all(isinstance(x, str) for x in tensor_names):
            raise ValueError("--names-json must contain a JSON list of strings")
    else:
        tensor_names = default_layer0_names()

    model = load_model_emptyish(args.model_dir, args.device)
    index = load_index(args.model_dir)

    print("[manual-load] before copy checks")
    for name in tensor_names[:3]:
        mx, mn = compare_loaded_vs_raw(model, model_dir=args.model_dir, tensor_name=name, index=index)
        print(f"[manual-load] before {name}: max_abs={mx} mean_abs={mn}")

    if not args.check_only:
        load_named_tensors_into_model(
            model,
            model_dir=args.model_dir,
            tensor_names=tensor_names,
        )

        print("[manual-load] after copy checks")
        for name in tensor_names:
            mx, mn = compare_loaded_vs_raw(model, model_dir=args.model_dir, tensor_name=name, index=index)
            print(f"[manual-load] after {name}: max_abs={mx} mean_abs={mn}")

    print("[manual-load] done")


if __name__ == "__main__":
    main()
