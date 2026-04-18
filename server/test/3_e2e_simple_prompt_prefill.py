from __future__ import annotations

import argparse

import torch

from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.deepseek_full_model_executor import DeepseekFullModelExecutor
from server.generation_runtime import run_generation_from_input_ids
from server.generation_types import GenerationState, GreedySampling, SamplingConfig
from server.inference_session import InferenceSession
from server.prefill_runtime import run_prefill_from_input_ids


def _decode_token_safe(executor, token_id: int) -> str:
    try:
        s = executor.decode([int(token_id)])
        return repr(s)
    except Exception as e:
        return f"<decode_error:{e}>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/test/config.json")
    ap.add_argument("--prompt", type=str, default="The capital of France is")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    args = ap.parse_args()

    cfg = load_config(args.config)
    coord = Coordinator(cfg["nodes"], log_level=cfg["log_level"])
    setup_control_plane(coord, cfg)

    kv_cache_cfg = cfg["kv_cache"]
    run_cfg = cfg.get("run", {})
    start_layer = int(run_cfg.get("start_layer", 0))
    end_layer = int(run_cfg.get("end_layer", 60))

    with InferenceSession(coord, cfg) as session:
        session.full_model_executor = DeepseekFullModelExecutor(session)
        session.initialize_full_model_runtime(
            tensor_cache_dir="tmp/non_moe_backbone_cache",
            split_layer=30,
            backbone_dtype=torch.bfloat16,
            kv_cache_cfg=kv_cache_cfg,
        )

        if not session.is_chat_runtime_ready():
            raise RuntimeError("chat runtime is not ready")

        tokenizer = session.get_deepseek_model_loader().load_tokenizer()
        encoded = tokenizer(
            args.prompt,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            raise RuntimeError("tokenizer did not return torch.Tensor input_ids")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise RuntimeError(f"input_ids expected shape [1, T], got {tuple(input_ids.shape)}")

        print(f"[prompt] {args.prompt!r}")
        print(f"[input_ids.shape] {tuple(input_ids.shape)}")
        print(f"[input_ids] {input_ids[0].tolist()}")

        session.reset_full_model_kv_cache(kv_cache_cfg=kv_cache_cfg)
        kv_cache = getattr(session, "page_attention_cache_managers", None)
        if kv_cache is None:
            raise RuntimeError("session.page_attention_cache_managers is not initialized")

        prefill_result = run_prefill_from_input_ids(
            session,
            input_ids=input_ids,
            start_layer=start_layer,
            end_layer=end_layer,
            kv_cache=kv_cache,
            collect_per_layer=False,
        )

        logits = prefill_result.next_token_logits
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError(f"next_token_logits expected torch.Tensor, got {type(logits).__name__}")
        if logits.ndim != 1:
            raise RuntimeError(f"next_token_logits expected 1D tensor, got shape {tuple(logits.shape)}")

        topk = torch.topk(logits, k=int(args.topk))
        topk_ids = [int(x) for x in topk.indices.tolist()]
        topk_vals = [float(x) for x in topk.values.tolist()]

        executor = session.full_model_executor
        if executor is None:
            raise RuntimeError("session.full_model_executor is not initialized")

        print("[prefill.topk]")
        for rank, (tok_id, val) in enumerate(zip(topk_ids, topk_vals), start=1):
            print(
                f"  rank={rank:02d} token_id={tok_id} "
                f"logit={val:.6f} text={_decode_token_safe(executor, tok_id)}"
            )

        greedy_first_token_id = int(topk_ids[0])
        greedy_first_token_text = executor.decode([greedy_first_token_id])
        print(f"[prefill.greedy_first_token_id] {greedy_first_token_id}")
        print(f"[prefill.greedy_first_token_text] {greedy_first_token_text!r}")

        session.reset_full_model_kv_cache(kv_cache_cfg=kv_cache_cfg)
        kv_cache = getattr(session, "page_attention_cache_managers", None)
        if kv_cache is None:
            raise RuntimeError("session.page_attention_cache_managers is not initialized")

        state = GenerationState(
            request_id="simple-prompt-test",
            model_name="deepseek-v3",
            sampling_config=SamplingConfig(
                strategy=GreedySampling(),
                max_new_tokens=int(args.max_new_tokens),
                stop_token_sequences=[],
            ),
        )

        result = run_generation_from_input_ids(
            session,
            state=state,
            input_ids=input_ids,
            start_layer=start_layer,
            end_layer=end_layer,
            kv_cache=kv_cache,
        )

        print(f"[generation.finish_reason] {result.finish_reason}")
        print(f"[generation.prompt_tokens] {result.prompt_tokens}")
        print(f"[generation.completion_tokens] {result.completion_tokens}")
        print(f"[generation.output_token_ids] {result.output_token_ids}")
        print(f"[generation.output_text] {result.output_text!r}")


if __name__ == "__main__":
    main()
