from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


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


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", type=str, required=True)
    ap.add_argument("--runtime-dir", type=str, required=True)
    ap.add_argument("--output-json", type=str, required=True)
    args = ap.parse_args()

    hf_meta = load_json(str(Path(args.hf_dir) / "hf_layer0_attention.json"))
    rt_meta = load_json(str(Path(args.runtime_dir) / "runtime_layer0_attention.json"))

    if hf_meta["input_ids"] != rt_meta["input_ids"]:
        raise RuntimeError("input_ids mismatch")

    hf_tensor = torch.load(Path(args.hf_dir) / "layer0_attn_output.pt", map_location="cpu")
    rt_tensor = torch.load(Path(args.runtime_dir) / "layer0_attn_output.pt", map_location="cpu")

    report = {
        "input_ids": hf_meta["input_ids"],
        "comparison": compare_tensors(hf_tensor, rt_tensor, "layer0_attn_output"),
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(report["comparison"], ensure_ascii=False, indent=2))
    print(f"[compare-attn0] wrote {out}")


if __name__ == "__main__":
    main()
