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


def interleaved_to_half_split(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"last dim must be even, got {x.shape[-1]}")
    d_half = x.shape[-1] // 2
    y = x.reshape(*x.shape[:-1], d_half, 2)
    y = y.transpose(-1, -2).reshape(*x.shape[:-1], x.shape[-1])
    return y.contiguous()


def maybe_relayout_ref_for_compare(
    name: str,
    ref_x: torch.Tensor,
    rt_x: torch.Tensor,
    hf_dir: Path,
    rt_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    ref_x, rt_x = align_tensor(ref_x, rt_x)

    if name.endswith("_q_flash"):
        rope_name = name.replace("_q_flash", "_q_rope_post_rotary")
        rt_rope_path = rt_dir / f"{rope_name}.pt"
        if rt_rope_path.exists():
            rt_rope = load_pt(rt_rope_path)
            rt_rope, _ = align_tensor(rt_rope, rt_rope)
            rope_dim = int(rt_rope.shape[-1])
            prefix_dim = int(rt_x.shape[-1]) - rope_dim
            if prefix_dim < 0:
                raise RuntimeError(
                    f"{name}: invalid dims total={rt_x.shape[-1]} rope={rope_dim}"
                )
            ref_prefix = ref_x[..., :prefix_dim]
            ref_rope = ref_x[..., prefix_dim:]
            ref_rope = interleaved_to_half_split(ref_rope)
            ref_x = torch.cat([ref_prefix, ref_rope], dim=-1).contiguous()

    elif name.endswith("_blocked_k_token"):
        latent_name = name.replace("_blocked_k_token", "_cache_latent")
        rt_latent_path = rt_dir / f"{latent_name}.pt"
        if rt_latent_path.exists():
            rt_latent = load_pt(rt_latent_path)
            rt_latent, _ = align_tensor(rt_latent, rt_latent)
            latent_dim = int(rt_latent.shape[-1])
            ref_latent = ref_x[..., :latent_dim]
            ref_rope = ref_x[..., latent_dim:]
            ref_rope = interleaved_to_half_split(ref_rope)
            ref_x = torch.cat([ref_latent, ref_rope], dim=-1).contiguous()

    return ref_x, rt_x


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
        hf, rt = maybe_relayout_ref_for_compare(name, hf, rt, hf_dir, rt_dir)
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
        f"layer_{layer_id}_q_flash",
        f"layer_{layer_id}_blocked_k_token",
    ]
    for name in base_names:
        maybe_compare(out, hf_dir, rt_dir, name)

    tail_names = [
        "final_hidden",
        "logits",
    ]
    for name in tail_names:
        maybe_compare(out, hf_dir, rt_dir, name)

    for i in range(layer_id + 1):
        maybe_compare(out, hf_dir, rt_dir, f"layer_{i}_output")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[compare-universal-single-layer] wrote {output_path}")


if __name__ == "__main__":
    main()
