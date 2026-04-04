from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return obj


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().view(-1)
    b = b.float().view(-1)
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-json", type=str, required=True)
    ap.add_argument("--runtime-json", type=str, required=True)
    ap.add_argument("--hf-logits", type=str, default=None)
    ap.add_argument("--runtime-logits", type=str, default=None)
    ap.add_argument("--output-json", type=str, default=None)
    args = ap.parse_args()

    hf = load_json(args.hf_json)
    rt = load_json(args.runtime_json)

    if hf["input_ids"] != rt["input_ids"]:
        raise RuntimeError(
            f"input_ids mismatch:\n"
            f"hf={hf['input_ids']}\n"
            f"runtime={rt['input_ids']}"
        )

    hf_topk_ids = list(hf["topk_ids"])
    rt_topk_ids = list(rt["topk_ids"])
    overlap = sorted(set(hf_topk_ids) & set(rt_topk_ids))

    report = {
        "input_ids": hf["input_ids"],
        "hf_argmax": int(hf["argmax"]),
        "runtime_argmax": int(rt["argmax"]),
        "argmax_same": int(hf["argmax"]) == int(rt["argmax"]),
        "hf_topk_ids": hf_topk_ids,
        "runtime_topk_ids": rt_topk_ids,
        "topk_overlap_ids": overlap,
        "topk_overlap_count": len(overlap),
        "hf_topk_vals": list(hf["topk_vals"]),
        "runtime_topk_vals": list(rt["topk_vals"]),
    }

    hf_logits_path = args.hf_logits or hf.get("logits_path")
    rt_logits_path = args.runtime_logits or rt.get("logits_path")

    if hf_logits_path and rt_logits_path:
        hf_logits = torch.load(hf_logits_path, map_location="cpu").float()
        rt_logits = torch.load(rt_logits_path, map_location="cpu").float()

        if tuple(hf_logits.shape) != tuple(rt_logits.shape):
            raise RuntimeError(
                f"logits shape mismatch: hf={tuple(hf_logits.shape)} runtime={tuple(rt_logits.shape)}"
            )

        abs_diff = (hf_logits - rt_logits).abs()
        report["cosine"] = cosine_similarity(hf_logits, rt_logits)
        report["max_abs"] = float(abs_diff.max().item())
        report["mean_abs"] = float(abs_diff.mean().item())
        report["hf_logits_shape"] = list(hf_logits.shape)
        report["runtime_logits_shape"] = list(rt_logits.shape)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"[compare] wrote {output_path}")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("[compare] first-step compare done")


if __name__ == "__main__":
    main()
