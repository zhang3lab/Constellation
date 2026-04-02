from __future__ import annotations

import argparse
import numpy as np
import torch

from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.deepseek_full_model_executor import DeepseekFullModelExecutor
from server.inference_session import InferenceSession
from server.test.utils import compare_arrays, print_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/test/config.json")
    ap.add_argument("--prompt", type=str, default="Hello world")
    args = ap.parse_args()

    cfg = load_config(args.config)
    coord = Coordinator(cfg["nodes"])
    setup_control_plane(coord, cfg)

    kv_cache_cfg = cfg["kv_cache"]

    with InferenceSession(coord, cfg) as session:
        session.full_model_executor = DeepseekFullModelExecutor(session)

        session.ensure_full_model_runtime(
            tensor_cache_dir="tmp/non_moe_backbone_cache",
            split_layer=30,
            backbone_dtype=torch.bfloat16,
            kv_cache_cfg=kv_cache_cfg,
        )

        executor = session.full_model_executor
        prepared = executor.prepare_prompt_hidden_input(args.prompt)

        hidden_in = np.asarray(prepared["hidden_in"], dtype=np.float32)
        input_ids = [int(x) for x in prepared["input_ids"]]
        last_token_id = int(input_ids[-1])

        embed_hidden = executor.embed_token_id(last_token_id)

        print(f"[semantics] prompt={args.prompt!r}")
        print(f"[semantics] input_ids={input_ids}")
        print(f"[semantics] last_token_id={last_token_id}")

        print_stats("semantics.prepared_hidden_in", hidden_in)
        print_stats("semantics.embed_last_token", embed_hidden)
        compare_arrays("prepared_hidden_vs_embed_last_token", hidden_in, embed_hidden)

        if not np.isfinite(hidden_in).all():
            raise RuntimeError("prepared hidden_in contains non-finite values")
        if not np.isfinite(embed_hidden).all():
            raise RuntimeError("embed hidden contains non-finite values")

        print("[semantics] prompt hidden semantics check finished")


if __name__ == "__main__":
    main()
