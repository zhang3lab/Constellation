import torch


def dequant_fp8_weight_blockwise(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    if weight_fp8.ndim != 2:
        raise RuntimeError(f"expected 2D fp8 weight, got shape={tuple(weight_fp8.shape)}")
    if scale_inv.ndim != 2:
        raise RuntimeError(f"expected 2D scale_inv, got shape={tuple(scale_inv.shape)}")

    rows, cols = weight_fp8.shape
    br, bc = scale_inv.shape

    expected_br = (rows + block_size - 1) // block_size
    expected_bc = (cols + block_size - 1) // block_size
    if br != expected_br or bc != expected_bc:
        raise RuntimeError(
            f"shape mismatch: weight={tuple(weight_fp8.shape)} "
            f"scale_inv={tuple(scale_inv.shape)} "
            f"expected_scale_inv=({expected_br}, {expected_bc}) "
            f"block_size={block_size}"
        )

    w = weight_fp8.float()
    s = scale_inv.float()
    out = torch.empty((rows, cols), dtype=torch.float32)

    for bi in range(br):
        r0 = bi * block_size
        r1 = min(r0 + block_size, rows)
        for bj in range(bc):
            c0 = bj * block_size
            c1 = min(c0 + block_size, cols)
            out[r0:r1, c0:c1] = w[r0:r1, c0:c1] * s[bi, bj]

    return out.contiguous()
