from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from safetensors import safe_open
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


def load_index(model_dir: str) -> dict:
    with open(f"{model_dir}/model.safetensors.index.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_raw_tensor(model_dir: str, tensor_name: str, index: dict | None = None) -> torch.Tensor:
    if index is None:
        index = load_index(model_dir)
    shard = index["weight_map"][tensor_name]
    with safe_open(f"{model_dir}/{shard}", framework="pt", device="cpu") as f:
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
    raw = load_raw_tensor(model_dir, tensor_name, index=index)
    with torch.no_grad():
        target.copy_(raw.to(device=target.device, dtype=target.dtype))


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


def layer0_manual_names() -> list[str]:
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


def run_and_collect(
    *,
    model_dir: str,
    device: str,
    input_ids: list[int],
    tensor_names: Iterable[str],
) -> tuple[dict, torch.Tensor]:
    model = load_model(model_dir, device)
    try:
        index = load_index(model_dir)
        for name in tensor_names:
            print(f"[manual-debug] copy {name}")
            copy_named_tensor_into_model(
                model,
                model_dir=model_dir,
                tensor_name=name,
                index=index,
            )

        model_device = next(model.parameters()).device
        ids = torch.tensor([input_ids], dtype=torch.long, device=model_device)

        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False, return_dict=True)

        layer0 = model.model.layers[0].self_attn
        dbg = dict(getattr(layer0, "last_debug", {}))
        dbg_cpu = {}
        for k, v in dbg.items():
            if isinstance(v, torch.Tensor):
                dbg_cpu[k] = v.detach().cpu()
            else:
                dbg_cpu[k] = v

        logits = out.logits[0, -1].detach().float().cpu()
        return dbg_cpu, logits
    finally:
        del model
        torch.cuda.empty_cache()


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
    print("[manual-debug] decoded =", repr(tokenizer.decode(input_ids)))

    names = layer0_manual_names()

    print("[manual-debug] eager ...")
    eager_dbg, eager_logits = run_and_collect(
        model_dir=args.eager_model_dir,
        device=args.device,
        input_ids=input_ids,
        tensor_names=names,
    )
    print("[manual-debug] eager done")

    print("[manual-debug] absorbed ...")
    absorbed_dbg, absorbed_logits = run_and_collect(
        model_dir=args.absorbed_model_dir,
        device=args.device,
        input_ids=input_ids,
        tensor_names=names,
    )
    print("[manual-debug] absorbed done")

    print("[manual-debug] eager keys =", sorted(eager_dbg.keys()))
    print("[manual-debug] absorbed keys =", sorted(absorbed_dbg.keys()))

    # compare q layout after permuting absorbed to eager layout
    absorbed_dbg_cmp = dict(absorbed_dbg)
    if "q_nope" in absorbed_dbg_cmp and isinstance(absorbed_dbg_cmp["q_nope"], torch.Tensor):
        if absorbed_dbg_cmp["q_nope"].ndim == 4:
            absorbed_dbg_cmp["q_nope_bhtd"] = absorbed_dbg_cmp["q_nope"].permute(0, 2, 1, 3).contiguous()
    if "q_pe" in absorbed_dbg_cmp and isinstance(absorbed_dbg_cmp["q_pe"], torch.Tensor):
        if absorbed_dbg_cmp["q_pe"].ndim == 4:
            absorbed_dbg_cmp["q_pe_bhtd"] = absorbed_dbg_cmp["q_pe"].permute(0, 2, 1, 3).contiguous()

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
    for key, val in absorbed_dbg_cmp.items():
        if isinstance(val, torch.Tensor):
            report["tensor_summaries"][f"absorbed::{key}"] = summarize_tensor(f"absorbed::{key}", val)

    compare_pairs = [
        ("hidden_states", "hidden_states"),
        ("q_nope", "q_nope_bhtd"),
        ("q_pe", "q_pe_bhtd"),
        ("q_nope", "q_nope"),
        ("q_pe", "q_pe"),
        ("attn_output_pre_o_proj", "attn_output_final"),
        ("final_logits", "final_logits"),
    ]

    for eager_key, absorbed_key in compare_pairs:
        if eager_key == "final_logits":
            xa = eager_logits
            xb = absorbed_logits
            diff = (xa - xb).abs()
            report["comparisons"]["final_logits"] = {
                "shape": list(xa.shape),
                "cosine": cosine_similarity(xa, xb),
                "max_abs": float(diff.max().item()),
                "mean_abs": float(diff.mean().item()),
            }
            continue

        a = eager_dbg.get(eager_key)
        b = absorbed_dbg_cmp.get(absorbed_key)
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            cmp = compare_optional({"x": a}, {"x": b}, "x")
            if cmp is not None:
                report["comparisons"][f"{eager_key}__vs__{absorbed_key}"] = cmp

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(report["comparisons"], ensure_ascii=False, indent=2))
    print(f"[manual-debug] wrote {output_path}")


if __name__ == "__main__":
    main()
