from __future__ import annotations

import argparse
import torch

from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.decode_runtime import run_decode
from server.deepseek_full_model_executor import DeepseekFullModelExecutor
from server.inference_session import InferenceSession


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/test/config.json")
    ap.add_argument("--prompt", type=str, default="Hello world")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--strategy", type=str, default="greedy")
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    coord = Coordinator(cfg["nodes"])
    setup_control_plane(coord, cfg)

    run_cfg = cfg["run"]
    kv_cache_cfg = cfg["kv_cache"]

    with InferenceSession(coord, cfg) as session:
        session.full_model_executor = DeepseekFullModelExecutor(session)

        session.ensure_full_model_runtime(
            tensor_cache_dir="tmp/non_moe_backbone_cache",
            split_layer=30,
            backbone_dtype=torch.bfloat16,
            kv_cache_cfg=kv_cache_cfg,
        )
        session.reset_full_model_kv_cache(kv_cache_cfg=kv_cache_cfg)

        result = run_decode(
            session,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            start_layer=run_cfg["start_layer"],
            end_layer=run_cfg["end_layer"],
            kv_cache=session.page_attention_cache_managers,
            strategy=args.strategy,
            topk=args.topk,
            temperature=args.temperature,
            collect_per_step=True,
        )

        print(f"[decode] prompt={result['prompt']!r}")
        print(f"[decode] strategy={result['strategy']}")
        print(f"[decode] prompt_token_ids={result['prompt_token_ids']}")

        for step in result["per_step"]:
            print(
                f"[decode] step={step['step']} pos={step['position']} "
                f"next_token_id={step['next_token_id']} "
                f"piece={step['next_token_text']!r}"
            )
            print(f"[decode] top_ids={step['logits_top_ids']}")
            print(f"[decode] top_vals={step['logits_top_vals']}")
            print(f"[decode] text_so_far={step['text_so_far']!r}")

        print(f"[decode] final_token_ids={result['generated_token_ids']}")
        print(f"[decode] final_text={result['generated_text']!r}")
        print("[decode] decode runtime passed")


if __name__ == "__main__":
    main()
