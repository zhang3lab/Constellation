from __future__ import annotations

import argparse
import json

import torch
from safetensors import safe_open


def load_index(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tensor(model_dir: str, tensor_name: str) -> torch.Tensor:
    index = load_index(f"{model_dir}/model.safetensors.index.json")
    shard = index["weight_map"][tensor_name]
    shard_path = f"{model_dir}/{shard}"
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        return f.get_tensor(tensor_name).float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-a", type=str, required=True)
    ap.add_argument("--dir-b", type=str, required=True)
    ap.add_argument("--tensor", type=str, required=True)
    args = ap.parse_args()

    a = load_tensor(args.dir_a, args.tensor)
    b = load_tensor(args.dir_b, args.tensor)

    print("shape_a =", tuple(a.shape))
    print("shape_b =", tuple(b.shape))

    if a.shape != b.shape:
        print("shape mismatch")
        return

    diff = (a - b).abs()
    cos = torch.nn.functional.cosine_similarity(
        a.view(1, -1), b.view(1, -1)
    ).item()

    print("cosine =", cos)
    print("max_abs =", diff.max().item())
    print("mean_abs =", diff.mean().item())


if __name__ == "__main__":
    main()
