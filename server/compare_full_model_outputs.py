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
    # HF often stores full-sequence tensors: [B, T, H]
    # Runtime manual full-model path often stores last-token decode tensors: [H] or [1, H]
    if hf_tensor.ndim == 3 and rt_tensor.ndim == 1:
        hf_tensor = hf_tensor[:, -1, :].contiguous().squeeze(0)
    elif hf_tensor.ndim == 3 and rt_tensor.ndim == 2 and rt_tensor.shape[0] == 1:
        hf_tensor = hf_tensor[:, -1, :].contiguous()

    return hf_tensor, rt_tensor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", type=str, required=True)
    ap.add_argument("--runtime-dir", type=str, required=True)
    ap.add_argument("--output-json", type=str, required=True)
    ap.add_argument("--start-layer", type=int, default=0)
    ap.add_argument("--end-layer", type=int, default=60)
    args = ap.parse_args()

    hf_dir = Path(args.hf_dir)
    rt_dir = Path(args.runtime_dir)

    out = {"comparisons": {}}

    for layer_id in range(args.start_layer, args.end_layer + 1):
        name = f"layer_{layer_id}_output"
        hf_tensor = load_pt(hf_dir / f"{name}.pt")
        rt_tensor = load_pt(rt_dir / f"{name}.pt")
        hf_tensor, rt_tensor = align_hf_to_runtime(hf_tensor, rt_tensor)
        out["comparisons"][name] = compare_tensors(hf_tensor, rt_tensor, name)

    hf_logits = load_pt(hf_dir / "logits.pt")
    rt_logits = load_pt(rt_dir / "logits.pt")
    hf_logits, rt_logits = align_hf_to_runtime(hf_logits, rt_logits)
    out["comparisons"]["logits"] = compare_tensors(hf_logits, rt_logits, "logits")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(out["comparisons"], ensure_ascii=False, indent=2))
    print(f"[compare-full] wrote {output_path}")


if __name__ == "__main__":
    main()
