from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def topk_summary(logits: torch.Tensor, k: int) -> tuple[list[int], list[float]]:
    vals, ids = torch.topk(logits, k=k)
    return (
        [int(x) for x in ids.tolist()],
        [float(x) for x in vals.tolist()],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--input-json", type=str, required=True)
    ap.add_argument("--output-json", type=str, required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--save-logits", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
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
    print("[hf] tokenizer ok")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map=args.device,
    )
    model.eval()
    print("[hf] model ok")

    model_device = next(model.parameters()).device
    ids = torch.tensor([input_ids], dtype=torch.long, device=model_device)

    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False, return_dict=True)

    logits = out.logits[0, -1].detach().float().cpu()
    topk_ids, topk_vals = topk_summary(logits, args.topk)

    report = {
        "backend": "hf",
        "model_dir": args.model_dir,
        "input_ids": input_ids,
        "argmax": int(torch.argmax(logits).item()),
        "topk_ids": topk_ids,
        "topk_vals": topk_vals,
        "logits_shape": list(logits.shape),
        "dtype": str(logits.dtype),
        "device": str(model_device),
    }

    if args.save_logits:
        logits_path = Path(args.save_logits)
        logits_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(logits, logits_path)
        report["logits_path"] = str(logits_path)
        print(f"[hf] saved logits to {logits_path}")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"[hf] wrote {output_path}")
    print(f"[hf] argmax = {report['argmax']}")
    print(f"[hf] topk_ids = {topk_ids}")
    print(f"[hf] topk_vals = {topk_vals}")


if __name__ == "__main__":
    main()
