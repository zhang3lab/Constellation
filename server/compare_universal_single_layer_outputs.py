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


def maybe_compare(out: dict, hf_dir: Path, rt_dir: Path, name: str) -> None:
    hf_path = hf_dir / f"{name}.pt"
    rt_path = rt_dir / f"{name}.pt"
    if hf_path.exists() and rt_path.exists():
        hf = load_pt(hf_path)
        rt = load_pt(rt_path)
        out[name] = compare_tensors(hf, rt, name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", type=str, required=True)
    ap.add_argument("--runtime-dir", type=str, required=True)
    ap.add_argument("--layer-id", type=int, required=True)
    ap.add_argument("--output-json", type=str, required=True)
    args = ap.parse_args()

    hf_dir = Path(args.hf_dir)
    rt_dir = Path(args.runtime_dir)
    layer_id = int(args.layer_id)

    out = {}

    base_names = [
        f"layer_{layer_id}_attention_input",
        f"layer_{layer_id}_attention_output",
        f"layer_{layer_id}_post_attention_hidden",
        f"layer_{layer_id}_ffn_hidden",
        f"layer_{layer_id}_dense_ffn_output",
        f"layer_{layer_id}_shared_expert_output",
        f"layer_{layer_id}_routed_output",
        f"layer_{layer_id}_ffn_total",
        f"layer_{layer_id}_output",
    ]
    for name in base_names:
        maybe_compare(out, hf_dir, rt_dir, name)

    tail_names = [
        "final_hidden",
        "logits",
    ]
    for name in tail_names:
        maybe_compare(out, hf_dir, rt_dir, name)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[compare-universal-single-layer] wrote {output_path}")


if __name__ == "__main__":
    main()
