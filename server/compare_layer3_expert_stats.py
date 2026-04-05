from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_pt(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu").float()


def stats(x: torch.Tensor) -> dict:
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

    hf_restored = load_pt(hf_dir / "layer_3_moe_restored_by_token.pt")[tok]
    hf_weighted = load_pt(hf_dir / "layer_3_moe_weighted_by_token.pt")[tok]

    out = {
        "token_idx": tok,
        "experts": [],
    }

    k = int(hf_restored.shape[0])
    for i in range(k):
        rt_out = load_pt(rt_dir / f"layer_3_tok{tok}_expert{i}_output.pt")
        rt_weighted = load_pt(rt_dir / f"layer_3_tok{tok}_expert{i}_weighted_output.pt")

        out["experts"].append(
            {
                "expert_slot": i,
                "hf_output": stats(hf_restored[i]),
                "runtime_output": stats(rt_out),
                "hf_weighted_output": stats(hf_weighted[i]),
                "runtime_weighted_output": stats(rt_weighted),
            }
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[compare-layer3-expert-stats] wrote {output_path}")


if __name__ == "__main__":
    main()
