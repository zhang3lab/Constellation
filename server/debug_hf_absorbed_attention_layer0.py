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
    if not isinstance(a[key], torch.Tensor) or not isinstance(b[key], torch.Tensor):
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


def run_and_collect(
    *,
    model_dir: str,
    device: str,
    input_ids: list[int],
) -> tuple[dict, torch.Tensor]:
    model = load_model(model_dir, device)
    model_device = next(model.parameters()).device
    ids = torch.tensor([input_ids], dtype=torch.long, device=model_device)

    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False, return_dict=True)

    layer0 = model.model.layers[0].self_attn
    dbg = dict(getattr(layer0, "last_debug", {}))
    logits = out.logits[0, -1].detach().float().cpu()

    # Move debug tensors to CPU copies explicitly
    dbg_cpu = {}
    for k, v in dbg.items():
        if isinstance(v, torch.Tensor):
            dbg_cpu[k] = v.detach().cpu()
        else:
            dbg_cpu[k] = v

    del out
    del model
    torch.cuda.empty_cache()

    return dbg_cpu, logits


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

    print("[debug] loading eager...")
    eager_dbg, eager_logits = run_and_collect(
        model_dir=args.eager_model_dir,
        device=args.device,
        input_ids=input_ids,
    )
    print("[debug] eager done")

    print("[debug] loading absorbed...")
    absorbed_dbg, absorbed_logits = run_and_collect(
        model_dir=args.absorbed_model_dir,
        device=args.device,
        input_ids=input_ids,
    )
    print("[debug] absorbed done")

    print("[debug] eager keys =", sorted(eager_dbg.keys()))
    print("[debug] absorbed keys =", sorted(absorbed_dbg.keys()))

    report = {
        "input_ids": input_ids,
        "decoded_input": tokenizer.decode(input_ids),
        "eager_debug_keys": sorted(eager_dbg.keys()),
        "absorbed_debug_keys": sorted(absorbed_dbg.keys()),
        "tensor_summaries": {},
        "comparisons": {},
        "eager_logits_argmax": int(torch.argmax(eager_logits).item()),
        "absorbed_logits_argmax": int(torch.argmax(absorbed_logits).item()),
    }

    for key, val in eager_dbg.items():
        if isinstance(val, torch.Tensor):
            report["tensor_summaries"][f"eager::{key}"] = summarize_tensor(f"eager::{key}", val)
    for key, val in absorbed_dbg.items():
        if isinstance(val, torch.Tensor):
            report["tensor_summaries"][f"absorbed::{key}"] = summarize_tensor(f"absorbed::{key}", val)

    compare_keys = [
        "hidden_states",
        "q_a",
        "q_nope",
        "q_pe",
        "compressed_kv",
        "k_nope",
        "value_states",
        "cache_latent",
        "cache_k_rope",
        "q_nope_absorb",
        "attn_output_pre_o_proj",
        "attn_output_final",
        "last_latent_out",
        "last_value_heads",
    ]

    for key in compare_keys:
        cmp_result = compare_optional(eager_dbg, absorbed_dbg, key)
        if cmp_result is not None:
            report["comparisons"][key] = cmp_result

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
