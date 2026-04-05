from __future__ import annotations

import argparse

import numpy as np
import torch

from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.deepseek_full_model_executor import DeepseekFullModelExecutor
from server.full_model_runtime import run_full_model
from server.inference_session import InferenceSession
from server.test.utils import print_stats, to_numpy_f32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/test/config.json")
    ap.add_argument("--prompt", type=str, default="Hello world")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    cfg = load_config(args.config)
    coord = Coordinator(cfg["nodes"])
    setup_control_plane(coord, cfg)

    run_cfg = cfg["run"]
    kv_cache_cfg = cfg["kv_cache"]

    start_layer = int(run_cfg["start_layer"])
    end_layer = int(run_cfg["end_layer"])
    collect_per_layer = bool(run_cfg["collect_per_layer"])

    with InferenceSession(coord, cfg) as session:
        session.full_model_executor = DeepseekFullModelExecutor(session)

        session.ensure_full_model_runtime(
            tensor_cache_dir="tmp/non_moe_backbone_cache",
            split_layer=30,
            backbone_dtype=torch.bfloat16,
            kv_cache_cfg=kv_cache_cfg,
        )
        session.reset_full_model_kv_cache(kv_cache_cfg=kv_cache_cfg)

        prepared = session.full_model_executor.prepare_prompt_hidden_input(args.prompt)
        hidden = prepared["hidden_in"]

        hidden_size = int(session.get_router_config()["hidden_size"])
        if tuple(hidden.shape) != (hidden_size,):
            raise RuntimeError(
                f"[e2e] hidden shape mismatch: got={tuple(hidden.shape)} expected={(hidden_size,)}"
            )

        print(f"[e2e] prompt={args.prompt!r}")
        print(f"[e2e] input_ids={prepared['input_ids']}")
        print(
            f"[e2e] start_layer={start_layer} "
            f"end_layer={end_layer} hidden_size={hidden_size}"
        )

        result = run_full_model(
            session,
            hidden,
            start_layer=start_layer,
            end_layer=end_layer,
            position_ids=prepared.get("position_ids"),
            attention_mask=prepared.get("attention_mask"),
            kv_cache=session.page_attention_cache_managers,
            collect_per_layer=collect_per_layer,
        )

        out_t = result["output"]
        out = to_numpy_f32(out_t)
        print_stats("e2e.final_hidden", out)

        if not np.isfinite(out).all():
            raise RuntimeError("[e2e] final hidden contains non-finite values")

        logits_result = session.full_model_executor.run_final_norm_and_lm_head(
            out_t,
            return_aux=False,
        )
        logits = to_numpy_f32(logits_result.output)
        print_stats("e2e.logits", logits)

        if not np.isfinite(logits).all():
            raise RuntimeError("[e2e] logits contain non-finite values")

        topk = int(args.topk)
        top_ids = np.argsort(logits)[-topk:][::-1]
        top_vals = logits[top_ids]
        top_tokens = [
            session.full_model_executor.decode([int(tok_id)])
            for tok_id in top_ids.tolist()
        ]

        print("[e2e] final_hidden[:8] =", out[:8])
        print("[e2e] logits_top_ids =", top_ids.tolist())
        print("[e2e] logits_top_vals =", top_vals.tolist())
        print("[e2e] logits_top_tokens =", top_tokens)

        if collect_per_layer:
            per_layer = result.get("per_layer", [])
            print(f"[e2e] collected_layers={len(per_layer)}")

        assert top_ids.shape == (topk,)
        assert len(top_tokens) == topk

        print("[e2e] full_model_debug passed")


if __name__ == "__main__":
    main()
