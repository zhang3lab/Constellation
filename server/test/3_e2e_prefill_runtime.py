from __future__ import annotations

import argparse

import torch

from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.inference_session import InferenceSession
from server.prefill_runtime import run_prefill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/test/config.json")
    ap.add_argument("--prompt", type=str, default="Hello world")
    ap.add_argument("--start-layer", type=int, default=None)
    ap.add_argument("--end-layer", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    coord = Coordinator(cfg["nodes"])
    setup_control_plane(coord, cfg)

    run_cfg = cfg["run"]
    kv_cache_cfg = cfg["kv_cache"]

    start_layer = int(run_cfg["start_layer"]) if args.start_layer is None else int(args.start_layer)
    end_layer = int(run_cfg["end_layer"]) if args.end_layer is None else int(args.end_layer)

    with InferenceSession(coord, cfg) as session:
        session.full_model_executor = session.full_model_executor or None

        session.ensure_full_model_runtime(
            tensor_cache_dir="tmp/non_moe_backbone_cache",
            split_layer=30,
            backbone_dtype=torch.bfloat16,
            kv_cache_cfg=kv_cache_cfg,
        )
        session.reset_full_model_kv_cache(kv_cache_cfg=kv_cache_cfg)

        if session.full_model_executor is None:
            from server.deepseek_full_model_executor import DeepseekFullModelExecutor

            session.full_model_executor = DeepseekFullModelExecutor(session)

        result = run_prefill(
            session,
            prompt=args.prompt,
            start_layer=start_layer,
            end_layer=end_layer,
            kv_cache=session.page_attention_cache_managers,
            collect_per_layer=False,
        )

        final_hidden = result["final_hidden"]
        final_hidden_last_token = result["final_hidden_last_token"]

        print("[prefill] ok")
        print(f"[prefill] prompt={args.prompt!r}")
        print(f"[prefill] num_input_tokens={len(result['input_ids'])}")
        print(f"[prefill] final_position={result['final_position']}")
        print(f"[prefill] final_hidden.shape={tuple(final_hidden.shape)}")
        print(f"[prefill] final_hidden_last_token.shape={tuple(final_hidden_last_token.shape)}")
        print(f"[prefill] final_hidden.dtype={final_hidden.dtype}")
        print(f"[prefill] final_hidden.device={final_hidden.device}")


if __name__ == "__main__":
    main()
