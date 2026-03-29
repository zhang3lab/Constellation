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
    parser.add_argument("--prefill-len", type=int, default=1024)
    parser.add_argument("--decode-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    if args.batch_size != 1:
        raise RuntimeError("this regression currently assumes batch_size=1")

    device = "cuda"
    dtype = torch.float32 if args.dtype == "float32" else torch.float16

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    loader = DeepseekModelLoader(args.model_root)
    mla_cfg = loader.mla_config()
    hidden_size = int(mla_cfg["dim"])

    total_len = int(args.prefill_len) + int(args.decode_steps)
    x_all = torch.randn(
        args.batch_size,
        total_len,
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

    prefill_x = x_all[:, : args.prefill_len, :]
    decode_xs = [
        x_all[:, args.prefill_len + i : args.prefill_len + i + 1, :]
        for i in range(args.decode_steps)
    ]

    with torch.no_grad():
        # prefill
        y_prefill_torch = wrapper_torch.forward(prefill_x, start_pos=0, mask=None)
        y_prefill_triton = wrapper_triton.forward(prefill_x, start_pos=0, mask=None)

        compare_arrays(
            "prefill_output_torch_vs_triton",
            y_prefill_torch.detach().float().cpu().numpy(),
            y_prefill_triton.detach().float().cpu().numpy(),
        )

        # decode
        decode_out_torch = []
        decode_out_triton = []

        for i, x_step in enumerate(decode_xs):
            start_pos = args.prefill_len + i

            y_step_torch = wrapper_torch.forward(x_step, start_pos=start_pos, mask=None)
            y_step_triton = wrapper_triton.forward(x_step, start_pos=start_pos, mask=None)

            decode_out_torch.append(y_step_torch)
            decode_out_triton.append(y_step_triton)

            compare_arrays(
                f"decode_step_{i}_torch_vs_triton",
                y_step_torch.detach().float().cpu().numpy(),
                y_step_triton.detach().float().cpu().numpy(),
            )

        y_decode_torch = torch.cat(decode_out_torch, dim=1)
        y_decode_triton = torch.cat(decode_out_triton, dim=1)

        compare_arrays(
            "decode_all_steps_torch_vs_triton",
            y_decode_torch.detach().float().cpu().numpy(),
            y_decode_triton.detach().float().cpu().numpy(),
        )

    print("prefill_x shape      :", tuple(prefill_x.shape), prefill_x.dtype, prefill_x.device)
    print("y_prefill_torch shape:", tuple(y_prefill_torch.shape))
    print("y_decode_torch shape :", tuple(y_decode_torch.shape))
    print("ALL CACHE REGRESSION CASES PASSED")


if __name__ == "__main__":
    main()
