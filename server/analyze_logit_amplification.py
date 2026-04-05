from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_pt(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu").float()


def vec_stats(x: torch.Tensor) -> dict:
    flat = x.reshape(-1)
    return {
        "shape": list(x.shape),
        "mean": float(flat.mean().item()),
        "abs_mean": float(flat.abs().mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "norm": float(torch.linalg.vector_norm(flat).item()),
        "max_abs": float(flat.abs().max().item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
    }


def rel_norm_diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    da = a.reshape(-1)
    db = b.reshape(-1)
    diff = da - db
    na = torch.linalg.vector_norm(da).item()
    nb = torch.linalg.vector_norm(db).item()
    nd = torch.linalg.vector_norm(diff).item()
    return {
        "norm_a": float(na),
        "norm_b": float(nb),
        "norm_diff": float(nd),
        "rel_diff_vs_a": float(nd / na) if na != 0 else float("nan"),
        "rel_diff_vs_b": float(nd / nb) if nb != 0 else float("nan"),
    }


def row_norm_stats(w: torch.Tensor) -> dict:
    # w: [vocab, hidden]
    row_norms = torch.linalg.vector_norm(w, dim=1)
    return {
        "shape": list(w.shape),
        "row_norm_mean": float(row_norms.mean().item()),
        "row_norm_std": float(row_norms.std(unbiased=False).item()),
        "row_norm_min": float(row_norms.min().item()),
        "row_norm_p50": float(row_norms.median().item()),
        "row_norm_p90": float(torch.quantile(row_norms, 0.9).item()),
        "row_norm_p99": float(torch.quantile(row_norms, 0.99).item()),
        "row_norm_max": float(row_norms.max().item()),
        "fro_norm": float(torch.linalg.vector_norm(w).item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", type=str, required=True)
    ap.add_argument("--runtime-dir", type=str, required=True)
    ap.add_argument("--output-json", type=str, required=True)
    args = ap.parse_args()

    hf_dir = Path(args.hf_dir)
    rt_dir = Path(args.runtime_dir)

    hf_hidden = load_pt(hf_dir / "final_hidden.pt")
    rt_hidden = load_pt(rt_dir / "final_hidden.pt")
    if hf_hidden.ndim == 3 and hf_hidden.shape[0] == 1:
        hf_hidden = hf_hidden.squeeze(0).contiguous()
    if rt_hidden.ndim == 3 and rt_hidden.shape[0] == 1:
        rt_hidden = rt_hidden.squeeze(0).contiguous()

    hf_logits = load_pt(hf_dir / "logits.pt")
    rt_logits = load_pt(rt_dir / "logits.pt")
    if hf_logits.ndim == 3 and hf_logits.shape[0] == 1:
        hf_logits = hf_logits.squeeze(0).contiguous()
    if rt_logits.ndim == 3 and rt_logits.shape[0] == 1:
        rt_logits = rt_logits.squeeze(0).contiguous()

    w = load_pt(hf_dir / "lm_head_weight.pt")

    # Same weight, recompute logits from both final_hidden tensors
    hf_logits_recomputed = torch.matmul(hf_hidden, w.t())
    rt_logits_recomputed = torch.matmul(rt_hidden, w.t())

    out = {
        "final_hidden": {
            "hf": vec_stats(hf_hidden),
            "runtime": vec_stats(rt_hidden),
            "diff": rel_norm_diff(hf_hidden, rt_hidden),
        },
        "logits_saved": {
            "hf": vec_stats(hf_logits),
            "runtime": vec_stats(rt_logits),
            "diff": rel_norm_diff(hf_logits, rt_logits),
        },
        "logits_recomputed_same_weight": {
            "hf_from_hf_hidden": vec_stats(hf_logits_recomputed),
            "rt_from_rt_hidden": vec_stats(rt_logits_recomputed),
            "diff": rel_norm_diff(hf_logits_recomputed, rt_logits_recomputed),
        },
        "lm_head_weight": row_norm_stats(w),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[analyze-logit-amplification] wrote {output_path}")


if __name__ == "__main__":
    main()
