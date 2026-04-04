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


def maybe_compare(report: dict, hf_dir: Path, rt_dir: Path, name: str) -> None:
    hf_path = hf_dir / f"{name}.pt"
    rt_path = rt_dir / f"{name}.pt"

    if not hf_path.exists():
        report[name] = {"key": name, "error": f"missing HF tensor: {hf_path}"}
        return
    if not rt_path.exists():
        report[name] = {"key": name, "error": f"missing runtime tensor: {rt_path}"}
        return

    hf_tensor = torch.load(hf_path, map_location="cpu")
    rt_tensor = torch.load(rt_path, map_location="cpu")
    report[name] = compare_tensors(hf_tensor, rt_tensor, name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", type=str, required=True)
    ap.add_argument("--runtime-dir", type=str, required=True)
    ap.add_argument("--output-json", type=str, required=True)
    args = ap.parse_args()

    hf_dir = Path(args.hf_dir)
    rt_dir = Path(args.runtime_dir)

    hf_meta = load_json(str(hf_dir / "hf_layer0_attention_aux.json"))
    rt_meta = load_json(str(rt_dir / "runtime_layer0_attention_aux.json"))

    if hf_meta["input_ids"] != rt_meta["input_ids"]:
        raise RuntimeError("input_ids mismatch")

    comparisons = {}
    maybe_compare(comparisons, hf_dir, rt_dir, "layer0_attn_output")
    maybe_compare(comparisons, hf_dir, rt_dir, "q_flash")
    maybe_compare(comparisons, hf_dir, rt_dir, "blocked_k_token")

    report = {
        "input_ids": hf_meta["input_ids"],
        "comparisons": comparisons,
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(report["comparisons"], ensure_ascii=False, indent=2))
    print(f"[compare-attn0-aux] wrote {out}")


if __name__ == "__main__":
    main()
