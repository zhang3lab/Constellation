from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from server.shallow_mla_wrapper import precompute_freqs_cis  # adjust if your project uses another helper
from server.mla_runtime import fused_apply_rotary_emb


def apply_rotary_emb_torch_ref(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    x: [B, L, H, D]
    freqs_cis: [L, D//2, 2]
    """
    dtype = x.dtype
    x_ri = torch.view_as_real(torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2)))
    x_real = x_ri[..., 0]
    x_imag = x_ri[..., 1]

    freqs_cos = freqs_cis[:, :, 0].unsqueeze(0).unsqueeze(2).to(device=x.device, dtype=torch.float32)
    freqs_sin = freqs_cis[:, :, 1].unsqueeze(0).unsqueeze(2).to(device=x.device, dtype=torch.float32)

    out_real = x_real * freqs_cos - x_imag * freqs_sin
    out_imag = x_real * freqs_sin + x_imag * freqs_cos
    out = torch.stack([out_real, out_imag], dim=-1).flatten(3)
    return out.to(dtype=dtype)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def make_freqs(
    *,
    seq_len: int,
    rope_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # DeepSeek / ShallowMLA-style defaults used elsewhere in your project.
    freqs = precompute_freqs_cis(
        dim=rope_dim,
        end=seq_len,
        theta=10000.0,
        rope_factor=40.0,
        beta_fast=32.0,
        beta_slow=1.0,
        original_seq_len=4096,
    )
    if not isinstance(freqs, torch.Tensor):
        freqs = torch.as_tensor(freqs)
    return freqs.to(device=device, dtype=dtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=12)
    ap.add_argument("--num-heads", type=int, default=128)
    ap.add_argument("--rope-dim", type=int, default=64)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-dir", type=str, default="")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    device = torch.device("cuda")

    torch.manual_seed(args.seed)

    x = torch.randn(
        args.batch_size,
        args.seq_len,
        args.num_heads,
        args.rope_dim,
        device=device,
        dtype=dtype,
    )
    freqs_cis = make_freqs(
        seq_len=args.seq_len,
        rope_dim=args.rope_dim,
        device=device,
        dtype=dtype,
    )

    y_ref = apply_rotary_emb_torch_ref(x, freqs_cis)
    y_fused = fused_apply_rotary_emb(x, freqs_cis)

    diff = (y_ref.float() - y_fused.float()).abs()

    report = {
        "x_shape": list(x.shape),
        "freqs_shape": list(freqs_cis.shape),
        "dtype": str(dtype),
        "cosine": cosine_similarity(y_ref, y_fused),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "ref_norm": float(y_ref.float().norm().item()),
        "fused_norm": float(y_fused.float().norm().item()),
    }

    print(json.dumps(report, indent=2))

    if args.save_dir:
        outdir = Path(args.save_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        torch.save(x.detach().cpu(), outdir / "x.pt")
        torch.save(freqs_cis.detach().cpu(), outdir / "freqs_cis.pt")
        torch.save(y_ref.detach().cpu(), outdir / "y_ref.pt")
        torch.save(y_fused.detach().cpu(), outdir / "y_fused.pt")
        with (outdir / "report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        print(f"[saved] {outdir}")


if __name__ == "__main__":
    main()
