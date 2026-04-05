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
    ap.add_argument("--layer-id", type=int, default=3)
    ap.add_argument("--experts-per-layer", type=int, default=256)
    ap.add_argument("--output-json", type=str, required=True)
    args = ap.parse_args()

    hf_dir = Path(args.hf_dir)
    rt_dir = Path(args.runtime_dir)
    tok = int(args.token_idx)
    layer_id = int(args.layer_id)
    base_gid = layer_id * int(args.experts_per_layer)

    hf_ids = torch.load(hf_dir / "layer_3_moe_topk_ids.pt", map_location="cpu")[tok].tolist()
    hf_restored = torch.load(hf_dir / "layer_3_moe_restored_by_token.pt", map_location="cpu")[tok]
    hf_weighted = torch.load(hf_dir / "layer_3_moe_weighted_by_token.pt", map_location="cpu")[tok]

    with open(rt_dir / f"layer_3_tok{tok}_expert_outputs_meta.json", "r", encoding="utf-8") as f:
        rt_meta = json.load(f)

    rt_by_gid_output = {}
    rt_by_gid_weighted = {}
    for i, meta in enumerate(rt_meta):
        gid = int(meta["expert_id"])
        rt_by_gid_output[gid] = torch.load(
            rt_dir / f"layer_3_tok{tok}_expert{i}_output.pt",
            map_location="cpu",
        )
        rt_by_gid_weighted[gid] = torch.load(
            rt_dir / f"layer_3_tok{tok}_expert{i}_weighted_output.pt",
            map_location="cpu",
        )

    out = {
        "token_idx": tok,
        "layer_id": layer_id,
        "comparisons": {},
    }

    for slot, local_id_f in enumerate(hf_ids):
        local_id = int(local_id_f)
        gid = base_gid + local_id

        hf_out = hf_restored[slot]
        hf_wout = hf_weighted[slot]
        rt_out = rt_by_gid_output[gid]
        rt_wout = rt_by_gid_weighted[gid]

        out["comparisons"][f"gid_{gid}_output"] = compare_tensors(
            hf_out, rt_out, f"gid_{gid}_output"
        )
        out["comparisons"][f"gid_{gid}_weighted_output"] = compare_tensors(
            hf_wout, rt_wout, f"gid_{gid}_weighted_output"
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[compare-layer3-experts-by-id] wrote {output_path}")


if __name__ == "__main__":
    main()
