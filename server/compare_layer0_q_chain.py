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

    hf_meta = load_json(str(Path(args.hf_dir) / "hf_layer0_q_chain.json"))
    rt_meta = load_json(str(Path(args.runtime_dir) / "runtime_layer0_q_chain.json"))

    if hf_meta["input_ids"] != rt_meta["input_ids"]:
        raise RuntimeError("input_ids mismatch")

    out = {"comparisons": {}}

    for name in [
        "q_latent_pre_norm",
        "q_latent_post_norm",
        "q_pre_split",
    ]:
        hf_tensor = torch.load(Path(args.hf_dir) / f"{name}.pt", map_location="cpu")
        rt_tensor = torch.load(Path(args.runtime_dir) / f"{name}.pt", map_location="cpu")
        out["comparisons"][name] = compare_tensors(hf_tensor, rt_tensor, name)

    hf_q_rope_pre = torch.load(Path(args.hf_dir) / "q_rope_pre_rotary.pt", map_location="cpu")
    rt_q_rope_pre = torch.load(Path(args.runtime_dir) / "q_rope_pre_rotary.pt", map_location="cpu")

    # Normalize both to [B, T, H, D]
    if hf_q_rope_pre.ndim == 4 and hf_q_rope_pre.shape[1] == 128:
        hf_q_rope_pre = hf_q_rope_pre.permute(0, 2, 1, 3).contiguous()
    if rt_q_rope_pre.ndim == 4 and rt_q_rope_pre.shape[1] == 128:
        rt_q_rope_pre = rt_q_rope_pre.permute(0, 2, 1, 3).contiguous()

    out["comparisons"]["q_rope_pre_rotary"] = compare_tensors(
        hf_q_rope_pre,
        rt_q_rope_pre,
        "q_rope_pre_rotary",
    )

    hf_q_rope_post = torch.load(Path(args.hf_dir) / "q_rope_post_rotary.pt", map_location="cpu")
    rt_q_rope_post = torch.load(Path(args.runtime_dir) / "q_rope_post_rotary.pt", map_location="cpu")

    # Normalize both to [B, T, H, D]
    if hf_q_rope_post.ndim == 4 and hf_q_rope_post.shape[1] == 128:
        hf_q_rope_post = hf_q_rope_post.permute(0, 2, 1, 3).contiguous()
    if rt_q_rope_post.ndim == 4 and rt_q_rope_post.shape[1] == 128:
        rt_q_rope_post = rt_q_rope_post.permute(0, 2, 1, 3).contiguous()

    out["comparisons"]["q_rope_post_rotary"] = compare_tensors(
        hf_q_rope_post,
        rt_q_rope_post,
        "q_rope_post_rotary",
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(out["comparisons"], ensure_ascii=False, indent=2))
    print(f"[compare-qchain] wrote {output_path}")


if __name__ == "__main__":
    main()
