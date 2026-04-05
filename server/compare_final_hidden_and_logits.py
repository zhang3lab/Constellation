from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load_pt(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu").float()


def align_tensor(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if a.ndim == 3 and a.shape[0] == 1 and b.ndim == 2:
        a = a.squeeze(0).contiguous()
    if b.ndim == 3 and b.shape[0] == 1 and a.ndim == 2:
        b = b.squeeze(0).contiguous()
    return a, b


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def compare_tensors(a: torch.Tensor, b: torch.Tensor, key: str) -> dict:
    a, b = align_tensor(a, b)

    if a.shape != b.shape:
        return {
            "key": key,
            "shape_match": False,
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
        }

    diff = (a - b).abs()
    denom = torch.dot(a.reshape(-1), a.reshape(-1)).item()
    alpha = (
        float(torch.dot(a.reshape(-1), b.reshape(-1)).item() / denom)
        if denom != 0
        else float("nan")
    )

    return {
        "key": key,
        "shape_match": True,
        "shape": list(a.shape),
        "cosine": cosine_similarity(a, b),
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

    out = {"comparisons": {}}

    if (hf_dir / "final_hidden.pt").exists() and (rt_dir / "final_hidden.pt").exists():
        hf = load_pt(hf_dir / "final_hidden.pt")
        rt = load_pt(rt_dir / "final_hidden.pt")
        out["comparisons"]["final_hidden"] = compare_tensors(hf, rt, "final_hidden")

    hf = load_pt(hf_dir / "logits.pt")
    rt = load_pt(rt_dir / "logits.pt")
    out["comparisons"]["logits"] = compare_tensors(hf, rt, "logits")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[compare-final-hidden-logits] wrote {output_path}")


if __name__ == "__main__":
    main()
