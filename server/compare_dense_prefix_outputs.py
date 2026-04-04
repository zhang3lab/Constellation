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
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", type=str, required=True)
    ap.add_argument("--runtime-dir", type=str, required=True)
    ap.add_argument("--output-json", type=str, required=True)
    args = ap.parse_args()

    hf_meta = load_json(str(Path(args.hf_dir) / "hf_prefix.json"))
    rt_meta = load_json(str(Path(args.runtime_dir) / "runtime_prefix.json"))

    if hf_meta["input_ids"] != rt_meta["input_ids"]:
        raise RuntimeError(
            f"input_ids mismatch:\n"
            f"hf={hf_meta['input_ids']}\n"
            f"runtime={rt_meta['input_ids']}"
        )

    report = {
        "input_ids": hf_meta["input_ids"],
        "comparisons": {},
    }

    for key in ["layer_0_output", "layer_1_output", "layer_2_output"]:
        hf_path = Path(args.hf_dir) / f"{key}.pt"
        rt_path = Path(args.runtime_dir) / f"{key}.pt"

        if not hf_path.exists():
            report["comparisons"][key] = {"key": key, "error": f"missing HF tensor: {hf_path}"}
            continue
        if not rt_path.exists():
            report["comparisons"][key] = {"key": key, "error": f"missing runtime tensor: {rt_path}"}
            continue

        hf_tensor = torch.load(hf_path, map_location="cpu")
        rt_tensor = torch.load(rt_path, map_location="cpu")

        if hf_tensor.ndim == 3 and rt_tensor.ndim == 2 and hf_tensor.shape[0] == 1:
            rt_tensor = rt_tensor.unsqueeze(0)
        elif hf_tensor.ndim == 2 and rt_tensor.ndim == 3 and rt_tensor.shape[0] == 1:
            hf_tensor = hf_tensor.unsqueeze(0)

        report["comparisons"][key] = compare_tensors(hf_tensor, rt_tensor, key)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(report["comparisons"], ensure_ascii=False, indent=2))
    print(f"[compare-prefix] wrote {output_path}")


if __name__ == "__main__":
    main()
