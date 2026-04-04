from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().view(-1)
    b = b.float().view(-1)
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def summarize_tensor(name: str, x: torch.Tensor) -> dict:
    x = x.detach().float().cpu()
    return {
        "name": name,
        "shape": list(x.shape),
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def compare_optional(a: dict, b: dict, key: str) -> dict | None:
    if key not in a or key not in b:
        return None
    xa = a[key].float()
    xb = b[key].float()
    if xa.shape != xb.shape:
        return {
            "key": key,
            "shape_a": list(xa.shape),
            "shape_b": list(xb.shape),
            "shape_match": False,
        }
    diff = (xa - xb).abs()
    return {
        "key": key,
        "shape_match": True,
        "shape": list(xa.shape),
        "cosine": cosine_similarity(xa, xb),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eager-model-dir", type=str, required=True)
    ap.add_argument("--absorbed-model-dir", type=str, required=True)
    ap.add_argument("--input-json", type=str, required=True)
    ap.add_argument("--output-json", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        inp = json.load(f)
    input_ids = inp["input_ids"]

    tokenizer = AutoTokenizer.from_pretrained(args.eager_model_dir, trust_remote_code=True)
    print("[debug] decoded =", repr(tokenizer.decode(input_ids)))

    eager_model = load_model(args.eager_model_dir, args.device)
    absorbed_model = load_model(args.absorbed_model_dir, args.device)

    eager_device = next(eager_model.parameters()).device
    absorbed_device = next(absorbed_model.parameters()).device

    eager_ids = torch.tensor([input_ids], dtype=torch.long, device=eager_device)
    absorbed_ids = torch.tensor([input_ids], dtype=torch.long, device=absorbed_device)

    with torch.no_grad():
        eager_out = eager_model(input_ids=eager_ids, use_cache=False, return_dict=True)
        absorbed_out = absorbed_model(input_ids=absorbed_ids, use_cache=False, return_dict=True)

    eager_layer0 = eager_model.model.layers[0].self_attn
    absorbed_layer0 = absorbed_model.model.layers[0].self_attn

    eager_dbg = dict(getattr(eager_layer0, "last_debug", {}))
    absorbed_dbg = dict(getattr(absorbed_layer0, "last_debug", {}))

    keys = sorted(set(eager_dbg.keys()) | set(absorbed_dbg.keys()))
    print("[debug] eager keys =", sorted(eager_dbg.keys()))
    print("[debug] absorbed keys =", sorted(absorbed_dbg.keys()))

    report = {
        "input_ids": input_ids,
        "decoded_input": tokenizer.decode(input_ids),
        "eager_debug_keys": sorted(eager_dbg.keys()),
        "absorbed_debug_keys": sorted(absorbed_dbg.keys()),
        "tensor_summaries": {},
        "comparisons": {},
        "eager_logits_argmax": int(torch.argmax(eager_out.logits[0, -1]).item()),
        "absorbed_logits_argmax": int(torch.argmax(absorbed_out.logits[0, -1]).item()),
    }

    for key, val in eager_dbg.items():
        if isinstance(val, torch.Tensor):
            report["tensor_summaries"][f"eager::{key}"] = summarize_tensor(f"eager::{key}", val)
    for key, val in absorbed_dbg.items():
        if isinstance(val, torch.Tensor):
            report["tensor_summaries"][f"absorbed::{key}"] = summarize_tensor(f"absorbed::{key}", val)

    compare_keys = [
        "hidden_states",
        "q_nope",
        "q_pe",
        "k_nope",
        "cache_latent",
        "cache_k_rope",
        "q_nope_absorb",
        "attn_output_pre_o_proj",
        "attn_output_final",
    ]

    for key in compare_keys:
        cmp_result = compare_optional(eager_dbg, absorbed_dbg, key)
        if cmp_result is not None:
            report["comparisons"][key] = cmp_result

    # also compare final logits directly
    eager_logits = eager_out.logits[0, -1].float().cpu()
    absorbed_logits = absorbed_out.logits[0, -1].float().cpu()
    logits_diff = (eager_logits - absorbed_logits).abs()
    report["comparisons"]["final_logits"] = {
        "shape": list(eager_logits.shape),
        "cosine": cosine_similarity(eager_logits, absorbed_logits),
        "max_abs": float(logits_diff.max().item()),
        "mean_abs": float(logits_diff.mean().item()),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(report["comparisons"], ensure_ascii=False, indent=2))
    print(f"[debug] wrote {output_path}")


if __name__ == "__main__":
    main()
