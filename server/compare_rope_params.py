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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", type=str, required=True)
    ap.add_argument("--runtime-dir", type=str, required=True)
    ap.add_argument("--output-json", type=str, required=True)
    args = ap.parse_args()

    hf_dir = Path(args.hf_dir)
    rt_dir = Path(args.runtime_dir)

    freq_cis = torch.load(rt_dir / "freq_cis.pt", map_location="cpu").float()
    rope_cos = torch.load(hf_dir / "rope_cos.pt", map_location="cpu").float()
    rope_sin = torch.load(hf_dir / "rope_sin.pt", map_location="cpu").float()

    # runtime freq_cis: [L, D/2, 2]
    freq_cos = freq_cis[..., 0]
    freq_sin = freq_cis[..., 1]

    # Normalize HF cos/sin to [L, D/2]
    # Common cases:
    #   [1, 1, L, D/2] -> squeeze -> [L, D/2]
    #   [1, L, 1, D/2] -> squeeze -> [L, D/2]
    #   [L, D/2] -> unchanged
    rope_cos_n = rope_cos.squeeze()
    rope_sin_n = rope_sin.squeeze()

    out = {
        "comparisons": {
            "rope_cos": compare_tensors(freq_cos, rope_cos_n, "rope_cos"),
            "rope_sin": compare_tensors(freq_sin, rope_sin_n, "rope_sin"),
        }
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(out["comparisons"], ensure_ascii=False, indent=2))
    print(f"[compare-rope-params] wrote {output_path}")


if __name__ == "__main__":
    main()
