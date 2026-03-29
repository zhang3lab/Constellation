import argparse
import numpy as np
import torch

from server.deepseek_model_loader import DeepseekModelLoader
from server.shallowmla_adapter import ShallowMLAAttentionWrapper
from server.test_utils import compare_arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=str, required=True)
    parser.add_argument("--layer-id", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--start-pos", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = "cuda"
    dtype = torch.float32 if args.dtype == "float32" else torch.float16

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    loader = DeepseekModelLoader(args.model_root)
    mla_cfg = loader.mla_config()
    hidden_size = int(mla_cfg["dim"])

    x = torch.randn(
        args.batch_size,
        args.seq_len,
        hidden_size,
        device=device,
        dtype=dtype,
    )

    wrapper_torch = ShallowMLAAttentionWrapper(
        model_loader=loader,
        layer_id=args.layer_id,
        max_batch_size=max(4, args.batch_size),
        dtype=dtype,
        device=device,
        optim_type="torch",
    )
    wrapper_triton = ShallowMLAAttentionWrapper(
        model_loader=loader,
        layer_id=args.layer_id,
        max_batch_size=max(4, args.batch_size),
        dtype=dtype,
        device=device,
        optim_type="triton",
    )

    with torch.no_grad():
        y_torch = wrapper_torch.forward(x, start_pos=args.start_pos, mask=None)
        y_triton = wrapper_triton.forward(x, start_pos=args.start_pos, mask=None)

    print("x shape       :", tuple(x.shape), x.dtype, x.device)
    print("y_torch shape :", tuple(y_torch.shape), y_torch.dtype, y_torch.device)
    print("y_triton shape:", tuple(y_triton.shape), y_triton.dtype, y_triton.device)

    compare_arrays(
        "shallowmla_torch_vs_triton",
        y_torch.detach().float().cpu().numpy(),
        y_triton.detach().float().cpu().numpy(),
    )

    yt = y_torch[0, 0].detach().float().cpu().numpy()
    yr = y_triton[0, 0].detach().float().cpu().numpy()
    print("[torch ] output[:8] =", yt[:8])
    print("[triton] output[:8] =", yr[:8])


if __name__ == "__main__":
    main()
