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


def align_hf_router_tensor(hf_tensor: torch.Tensor, rt_tensor: torch.Tensor, token_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    # HF stores full-sequence router debug: [T, ...]
    # runtime stores per-token router debug: [...]
    if hf_tensor.ndim >= 2 and rt_tensor.ndim == hf_tensor.ndim - 1:
        hf_tensor = hf_tensor[token_idx].contiguous()
    return hf_tensor, rt_tensor


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

    pairs = [
        ("layer_3_logits", f"layer_3_tok{tok}_logits"),
        ("layer_3_scores", f"layer_3_tok{tok}_scores"),
        ("layer_3_scores_for_choice", f"layer_3_tok{tok}_scores_for_choice"),
        ("layer_3_group_scores", f"layer_3_tok{tok}_group_scores"),
        ("layer_3_selected_group_idx", f"layer_3_tok{tok}_selected_group_idx"),
        ("layer_3_score_mask", f"layer_3_tok{tok}_score_mask"),
        ("layer_3_topk_choice_vals", f"layer_3_tok{tok}_topk_choice_vals"),
        ("layer_3_topk_idx", f"layer_3_tok{tok}_topk_idx"),
        ("layer_3_topk_weight", f"layer_3_tok{tok}_topk_weight"),
        ("layer_3_resident_mask", f"layer_3_tok{tok}_resident_mask"),
    ]

    out = {"token_idx": tok, "comparisons": {}}
    for hf_name, rt_name in pairs:
        hf_tensor = load_pt(hf_dir / f"{hf_name}.pt")
        rt_tensor = load_pt(rt_dir / f"{rt_name}.pt")
        hf_tensor, rt_tensor = align_hf_router_tensor(hf_tensor, rt_tensor, tok)
        out["comparisons"][hf_name] = compare_tensors(hf_tensor, rt_tensor, hf_name)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(out["comparisons"], ensure_ascii=False, indent=2))
    print(f"[compare-layer3-router] wrote {output_path}")


if __name__ == "__main__":
    main()
