from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().view(-1)
    b = b.float().view(-1)
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def load_model(model_dir: str, device: str):
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


def compare_param(a: torch.Tensor, b: torch.Tensor, name: str) -> dict:
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()

    if a.shape != b.shape:
        return {
            "name": name,
            "shape_match": False,
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
        }

    diff = (a - b).abs()
    return {
        "name": name,
        "shape_match": True,
        "shape": list(a.shape),
        "cosine": cosine_similarity(a, b),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "a_mean": float(a.mean().item()),
        "b_mean": float(b.mean().item()),
        "a_std": float(a.std().item()),
        "b_std": float(b.std().item()),
    }


def get_param(model, dotted_name: str) -> torch.Tensor:
    obj = model
    for part in dotted_name.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    if not isinstance(obj, torch.Tensor) and not isinstance(obj, torch.nn.Parameter):
        raise TypeError(f"{dotted_name} is not a tensor/parameter, got {type(obj).__name__}")
    return obj


def read_param_from_model_dir(model_dir: str, device: str, dotted_name: str) -> torch.Tensor:
    model = load_model(model_dir, device)
    try:
        tensor = get_param(model, dotted_name).detach().cpu().clone()
    finally:
        del model
        torch.cuda.empty_cache()
    return tensor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eager-model-dir", type=str, required=True)
    ap.add_argument("--absorbed-model-dir", type=str, required=True)
    ap.add_argument("--output-json", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    param_names = [
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
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.shared_experts.gate_proj.weight",
        "model.layers.0.mlp.shared_experts.up_proj.weight",
        "model.layers.0.mlp.shared_experts.down_proj.weight",
    ]

    report = {
        "eager_model_dir": args.eager_model_dir,
        "absorbed_model_dir": args.absorbed_model_dir,
        "comparisons": {},
    }

    for name in param_names:
        try:
            print(f"[param-compare] reading eager {name} ...")
            a = read_param_from_model_dir(args.eager_model_dir, args.device, name)

            print(f"[param-compare] reading absorbed {name} ...")
            b = read_param_from_model_dir(args.absorbed_model_dir, args.device, name)

            report["comparisons"][name] = compare_param(a, b, name)
            print(
                f"[param-compare] {name}: "
                f"cos={report['comparisons'][name].get('cosine')} "
                f"max_abs={report['comparisons'][name].get('max_abs')}"
            )
        except Exception as exc:
            report["comparisons"][name] = {
                "name": name,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[param-compare] {name}: ERROR {exc}")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"[param-compare] wrote {output_path}")


if __name__ == "__main__":
    main()
