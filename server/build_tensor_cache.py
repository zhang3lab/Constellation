import argparse

from server.config import load_config
from server.deepseek_model_loader import DeepseekModelLoader
from server.tensor_cache import (
    TensorCacheBuilder,
    collect_non_moe_backbone_tensor_names_deepseek,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-root",
        type=str,
        default="/model/ModelScope/deepseek-ai/DeepSeek-V3.1",
    )
    ap.add_argument(
        "--cache-dir",
        type=str,
        default="tmp/non_moe_backbone_cache",
    )
    ap.add_argument(
        "--config",
        type=str,
        default="server/test/config.json",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    kv_cache_cfg = cfg.get("kv_cache")
    if not isinstance(kv_cache_cfg, dict):
        raise RuntimeError("config missing kv_cache dict")

    max_seq_len = int(kv_cache_cfg["max_seq_len"])

    loader = DeepseekModelLoader(args.model_root)
    names = collect_non_moe_backbone_tensor_names_deepseek()

    print(f"[tensor-cache] model_root={args.model_root}")
    print(f"[tensor-cache] cache_dir={args.cache_dir}")
    print(f"[tensor-cache] config={args.config}")
    print(f"[tensor-cache] max_seq_len={max_seq_len}")
    print(f"[tensor-cache] num_tensors={len(names)}")

    builder = TensorCacheBuilder(args.cache_dir)
    builder.build_from_names(
        loader,
        names,
        max_seq_len=max_seq_len,
        overwrite=args.overwrite,
        num_workers=8,
        prefetch=8,
    )

    print("[tensor-cache] done")


if __name__ == "__main__":
    main()
