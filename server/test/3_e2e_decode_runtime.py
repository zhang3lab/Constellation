from __future__ import annotations

import argparse

import torch

from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.decode_runtime import (
    run_decode_step_logits,
    sample_greedy_from_logits,
)
from server.deepseek_full_model_executor import DeepseekFullModelExecutor
from server.inference_session import InferenceSession


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/test/config.json")
    ap.add_argument("--prompt", type=str, default="Hello world")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--topk", type=int, default=5)
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

        executor = session.full_model_executor
        tokenizer = session.get_deepseek_model_loader().load_tokenizer()

        prepared = executor.prepare_prompt_hidden_input(args.prompt)
        current_hidden = prepared["hidden_in"]
        if not isinstance(current_hidden, torch.Tensor):
            raise TypeError(
                f'prepared["hidden_in"] expected torch.Tensor, got {type(current_hidden).__name__}'
            )

        prompt_token_ids = [int(x) for x in prepared["input_ids"]]
        generated_ids = list(prompt_token_ids)
        current_pos = executor.infer_prompt_last_position(prepared)

        per_step = []

        for step in range(int(args.max_new_tokens)):
            out = run_decode_step_logits(
                session,
                current_hidden=current_hidden,
                position_id=int(current_pos),
                start_layer=int(run_cfg["start_layer"]),
                end_layer=int(run_cfg["end_layer"]),
                kv_cache=session.page_attention_cache_managers,
            )

            logits = out["logits"]
            topk_vals, topk_ids = torch.topk(logits, k=int(args.topk))
            next_token_id = sample_greedy_from_logits(logits)

            generated_ids.append(next_token_id)

            per_step.append(
                {
                    "step": step,
                    "position": int(current_pos),
                    "next_token_id": int(next_token_id),
                    "next_token_text": tokenizer.decode([next_token_id]),
                    "logits_top_ids": [int(x) for x in topk_ids.detach().cpu().tolist()],
                    "logits_top_vals": [float(x) for x in topk_vals.detach().cpu().tolist()],
                    "text_so_far": tokenizer.decode(generated_ids),
                }
            )

            current_hidden = executor.embed_token_ids(next_token_id)
            current_pos += 1

        print(f"[decode] prompt={args.prompt!r}")
        print("[decode] strategy='greedy'")
        print(f"[decode] prompt_token_ids={prompt_token_ids}")

        for step in per_step:
            print(
                f"[decode] step={step['step']} pos={step['position']} "
                f"next_token_id={step['next_token_id']} "
                f"piece={step['next_token_text']!r}"
            )
            print(f"[decode] top_ids={step['logits_top_ids']}")
            print(f"[decode] top_vals={step['logits_top_vals']}")
            print(f"[decode] text_so_far={step['text_so_far']!r}")

        print(f"[decode] final_token_ids={generated_ids}")
        print(f"[decode] final_text={tokenizer.decode(generated_ids)!r}")
        print("[decode] decode runtime passed")


if __name__ == "__main__":
    main()
