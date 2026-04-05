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
    ap.add_argument("--token-idx", type=int, default=11)
    ap.add_argument("--output-json", type=str, required=True)
    args = ap.parse_args()

    hf_dir = Path(args.hf_dir)
    rt_dir = Path(args.runtime_dir)
    tok = int(args.token_idx)

    hf_ids = torch.load(hf_dir / "layer_3_moe_topk_ids.pt", map_location="cpu")[tok]
    hf_restored = torch.load(hf_dir / "layer_3_moe_restored_by_token.pt", map_location="cpu")[tok]
    hf_weighted = torch.load(hf_dir / "layer_3_moe_weighted_by_token.pt", map_location="cpu")[tok]

    out = {
        "token_idx": tok,
        "hf_local_ids": [int(x) for x in hf_ids.tolist()],
        "hf_global_ids": [768 + int(x) for x in hf_ids.tolist()],
        "comparisons": {},
    }

    for i in range(hf_restored.shape[0]):
        rt_out = torch.load(rt_dir / f"layer_3_tok{tok}_expert{i}_output.pt", map_location="cpu")
        rt_weighted = torch.load(rt_dir / f"layer_3_tok{tok}_expert{i}_weighted_output.pt", map_location="cpu")

        out["comparisons"][f"expert{i}_output"] = compare_tensors(
            hf_restored[i], rt_out, f"expert{i}_output"
        )
        out["comparisons"][f"expert{i}_weighted_output"] = compare_tensors(
            hf_weighted[i], rt_weighted, f"expert{i}_weighted_output"
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[compare-layer3-experts] wrote {output_path}")


if __name__ == "__main__":
    main()
