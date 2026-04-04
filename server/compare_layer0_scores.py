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


def load_pt(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", type=str, required=True)
    ap.add_argument("--runtime-dir", type=str, required=True)
    ap.add_argument("--output-json", type=str, required=True)
    args = ap.parse_args()

    hf_dir = Path(args.hf_dir)
    rt_dir = Path(args.runtime_dir)

    out = {"comparisons": {}}

    # HF [1,H,T], runtime [1,L,H,T] -> take last query and permute
    hf = load_pt(hf_dir / "scores_nope.pt")
    rt = load_pt(rt_dir / "scores_pre_softmax.pt")
    if rt.ndim == 4:
        rt = rt[:, -1].permute(0, 2, 1).contiguous()  # [1,H,T]
    out["comparisons"]["scores_nope_vs_runtime_pre"] = compare_tensors(
        hf, rt, "scores_nope_vs_runtime_pre"
    )

    hf = load_pt(hf_dir / "scores_rope.pt")
    # no direct runtime rope-only score unless you export it later
    out["comparisons"]["scores_rope_hf_only"] = {
        "key": "scores_rope_hf_only",
        "shape": list(hf.shape),
    }

    hf = load_pt(hf_dir / "scores_pre_softmax.pt")
    rt = load_pt(rt_dir / "scores_pre_softmax.pt")
    if rt.ndim == 4:
        rt = rt[:, -1].permute(0, 2, 1).contiguous()  # [1,H,T]
    out["comparisons"]["scores_pre_softmax"] = compare_tensors(
        hf, rt, "scores_pre_softmax"
    )

    hf = load_pt(hf_dir / "scores_post_softmax.pt")
    rt = load_pt(rt_dir / "scores_post_softmax.pt")
    if rt.ndim == 4:
        rt = rt[:, -1].permute(0, 2, 1).contiguous()  # [1,H,T]
    out["comparisons"]["scores_post_softmax"] = compare_tensors(
        hf, rt, "scores_post_softmax"
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(out["comparisons"], ensure_ascii=False, indent=2))
    print(f"[compare-scores] wrote {output_path}")


if __name__ == "__main__":
    main()
