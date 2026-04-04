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
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def compare_tensors(a: torch.Tensor, b: torch.Tensor, key: str) -> dict:
    xa = a.float()
    xb = b.float()
    if xa.shape != xb.shape:
        return {
            "key": key,
            "shape_match": False,
            "shape_a": list(xa.shape),
            "shape_b": list(xb.shape),
        }
    diff = (xa - xb).abs()
    denom = torch.dot(xa.reshape(-1), xa.reshape(-1)).item()
    alpha = float(torch.dot(xa.reshape(-1), xb.reshape(-1)).item() / denom) if denom != 0 else float("nan")
    return {
        "key": key,
        "shape_match": True,
        "shape": list(xa.shape),
        "cosine": cosine_similarity(xa, xb),
        "alpha": alpha,
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


def copy_named_tensor_into_model(model, *, model_dir: str, tensor_name: str, index: dict | None = None) -> None:
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


def dense_layer_names(layer_id: int) -> list[str]:
    prefix = f"model.layers.{layer_id}"
    return [
        f"{prefix}.input_layernorm.weight",
        f"{prefix}.self_attn.q_a_proj.weight",
        f"{prefix}.self_attn.q_a_layernorm.weight",
        f"{prefix}.self_attn.q_b_proj.weight",
        f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
        f"{prefix}.self_attn.kv_a_layernorm.weight",
        f"{prefix}.self_attn.kv_b_proj.weight",
        f"{prefix}.self_attn.o_proj.weight",
        f"{prefix}.post_attention_layernorm.weight",
        f"{prefix}.mlp.gate_proj.weight",
        f"{prefix}.mlp.up_proj.weight",
        f"{prefix}.mlp.down_proj.weight",
    ]


def prefix_names() -> list[str]:
    names = ["model.embed_tokens.weight"]
    for layer_id in [0, 1, 2]:
        names.extend(dense_layer_names(layer_id))
    return names


def run_and_collect(model_dir: str, device: str, input_ids: list[int], tensor_names: Iterable[str]) -> tuple[dict, torch.Tensor]:
    model = load_model(model_dir, device)
    try:
        index = load_index(model_dir)
        for name in tensor_names:
            print(f"[prefix-debug] copy {name}")
            copy_named_tensor_into_model(model, model_dir=model_dir, tensor_name=name, index=index)

        ids = torch.tensor([input_ids], dtype=torch.long, device=next(model.parameters()).device)

        layer_hidden = {}

        def make_hook(name: str):
            def _hook(_module, _inp, out):
                x = out[0] if isinstance(out, tuple) else out
                layer_hidden[name] = x.detach().cpu()
            return _hook

        hooks = []
        for i in [0, 1, 2]:
            hooks.append(model.model.layers[i].register_forward_hook(make_hook(f"layer_{i}_output")))

        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False, return_dict=True)

        for h in hooks:
            h.remove()

        logits = out.logits[0, -1].detach().float().cpu()
        return layer_hidden, logits
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
    print("[prefix-debug] decoded =", repr(tokenizer.decode(input_ids)))

    names = prefix_names()

    print("[prefix-debug] eager ...")
    eager_hidden, eager_logits = run_and_collect(args.eager_model_dir, args.device, input_ids, names)
    print("[prefix-debug] eager done")

    print("[prefix-debug] absorbed ...")
    absorbed_hidden, absorbed_logits = run_and_collect(args.absorbed_model_dir, args.device, input_ids, names)
    print("[prefix-debug] absorbed done")

    report = {
        "input_ids": input_ids,
        "decoded_input": tokenizer.decode(input_ids),
        "comparisons": {},
    }

    for k in sorted(set(eager_hidden.keys()) | set(absorbed_hidden.keys())):
        if k in eager_hidden and k in absorbed_hidden:
            report["comparisons"][k] = compare_tensors(eager_hidden[k], absorbed_hidden[k], k)

    report["comparisons"]["final_logits"] = compare_tensors(eager_logits, absorbed_logits, "final_logits")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(report["comparisons"], ensure_ascii=False, indent=2))
    print(f"[prefix-debug] wrote {output_path}")


if __name__ == "__main__":
    main()
