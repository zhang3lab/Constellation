from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def topk_summary(logits: torch.Tensor, k: int) -> tuple[list[int], list[float]]:
    vals, ids = torch.topk(logits, k=k)
    return (
        [int(x) for x in ids.tolist()],
        [float(x) for x in vals.tolist()],
    )


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().view(-1)
    b = b.float().view(-1)
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def run_one(
    model_dir: str,
    *,
    attn_impl: str,
    input_ids: list[int],
    device: str,
    topk: int,
) -> dict:
    config = AutoConfig.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )
    config._attn_implementation = attn_impl

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map=device,
    )
    model.eval()

    model_device = next(model.parameters()).device
    ids = torch.tensor([input_ids], dtype=torch.long, device=model_device)

    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False, return_dict=True)

    logits = out.logits[0, -1].detach().float().cpu()
    topk_ids, topk_vals = topk_summary(logits, topk)

    return {
        "attn_impl": attn_impl,
        "argmax": int(torch.argmax(logits).item()),
        "topk_ids": topk_ids,
        "topk_vals": topk_vals,
        "logits": logits,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--input-json", type=str, required=True)
    ap.add_argument("--output-json", type=str, required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save-eager-logits", type=str, default=None)
    ap.add_argument("--save-absorbed-logits", type=str, default=None)
    args = ap.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        inp = json.load(f)

    input_ids = inp["input_ids"]
    if not isinstance(input_ids, list) or not input_ids:
        raise ValueError("input_json must contain non-empty list input_ids")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
    )
    print("[ablate] tokenizer ok")
    print(f"[ablate] num_input_ids = {len(input_ids)}")
    print(f"[ablate] decoded = {tokenizer.decode(input_ids)!r}")

    eager = run_one(
        args.model_dir,
        attn_impl="eager",
        input_ids=input_ids,
        device=args.device,
        topk=args.topk,
    )
    print("[ablate] eager ok")

    absorbed = run_one(
        args.model_dir,
        attn_impl="absorbed",
        input_ids=input_ids,
        device=args.device,
        topk=args.topk,
    )
    print("[ablate] absorbed ok")

    eager_logits = eager.pop("logits")
    absorbed_logits = absorbed.pop("logits")

    abs_diff = (eager_logits - absorbed_logits).abs()
    overlap = sorted(set(eager["topk_ids"]) & set(absorbed["topk_ids"]))

    report = {
        "input_ids": input_ids,
        "decoded_input": tokenizer.decode(input_ids),
        "eager": eager,
        "absorbed": absorbed,
        "argmax_same": eager["argmax"] == absorbed["argmax"],
        "topk_overlap_ids": overlap,
        "topk_overlap_count": len(overlap),
        "cosine": cosine_similarity(eager_logits, absorbed_logits),
        "max_abs": float(abs_diff.max().item()),
        "mean_abs": float(abs_diff.mean().item()),
    }

    if args.save_eager_logits:
        path = Path(args.save_eager_logits)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(eager_logits, path)
        report["eager_logits_path"] = str(path)

    if args.save_absorbed_logits:
        path = Path(args.save_absorbed_logits)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(absorbed_logits, path)
        report["absorbed_logits_path"] = str(path)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[ablate] wrote {output_path}")


if __name__ == "__main__":
    main()
