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


def align_hf_to_runtime(hf_tensor: torch.Tensor, rt_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if hf_tensor.ndim == 3 and hf_tensor.shape[0] == 1 and rt_tensor.ndim == 2:
        hf_tensor = hf_tensor.squeeze(0).contiguous()
    return hf_tensor, rt_tensor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", type=str, required=True)
    ap.add_argument("--runtime-dir", type=str, required=True)
    ap.add_argument("--output-json", type=str, required=True)
    args = ap.parse_args()

    hf_dir = Path(args.hf_dir)
    rt_dir = Path(args.runtime_dir)

    names = [
        "layer_4_attention_output",
        "layer_4_post_attention_hidden",
        "layer_4_ffn_hidden",
        "layer_4_shared_expert_output",
        "layer_4_routed_output",
        "layer_4_ffn_total",
        "layer_4_output",
    ]

    out = {"comparisons": {}}
    for name in names:
        hf_tensor = load_pt(hf_dir / f"{name}.pt")
        rt_tensor = load_pt(rt_dir / f"{name}.pt")
        hf_tensor, rt_tensor = align_hf_to_runtime(hf_tensor, rt_tensor)
        out["comparisons"][name] = compare_tensors(hf_tensor, rt_tensor, name)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(out["comparisons"], ensure_ascii=False, indent=2))
    print(f"[compare-layer4] wrote {output_path}")


if __name__ == "__main__":
    main()
