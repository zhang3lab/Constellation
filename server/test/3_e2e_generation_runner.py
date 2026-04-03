from __future__ import annotations

import argparse

import torch

from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.deepseek_full_model_executor import DeepseekFullModelExecutor
from server.generation_runner import GenerationRunner
from server.generation_types import (
    GreedySampling,
    SamplingConfig,
)
from server.inference_session import InferenceSession


def _encode_prompt(session: InferenceSession, prompt: str) -> list[int]:
    tokenizer = session.get_deepseek_model_loader().load_tokenizer()
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = [int(x) for x in input_ids]
    if not input_ids:
        raise RuntimeError("prompt encoded to empty input_ids")
    return input_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/test/config.json")
    ap.add_argument("--prompt", type=str, default="Hello world")
    ap.add_argument("--max-new-tokens", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config(args.config)
    coord = Coordinator(cfg["nodes"])
    setup_control_plane(coord, cfg)

    with InferenceSession(coord, cfg) as session:
        session.full_model_executor = DeepseekFullModelExecutor(session)

        runner = GenerationRunner(session)

        sampling_config = SamplingConfig(
            strategy=GreedySampling(),
            max_new_tokens=int(args.max_new_tokens),
        )

        input_ids = _encode_prompt(session, args.prompt)
        print(f"[runner] prompt={args.prompt!r}")
        print(f"[runner] input_ids={input_ids}")

        # ---- create_generation + prefill ----
        gen = runner.create_generation(
            sampling_config=sampling_config,
        )

        prefill_result = runner.prefill(gen, input_ids)

        print("[runner] prefill ok")
        print(f"[runner] model_name={gen.model_name!r}")
        print(f"[runner] request_id={gen.request_id}")
        print(f"[runner] prompt_tokens_count={gen.prompt_tokens_count}")
        print(f"[runner] completion_tokens_count={gen.completion_tokens_count}")
        print(f"[runner] total_tokens_count={gen.total_tokens_count}")
        print(f"[runner] is_prefilled={gen.is_prefilled}")
        print(f"[runner] is_finished={gen.is_finished}")
        print(f"[runner] can_decode={gen.can_decode}")
        print(f"[runner] has_kv_cache={gen.kv_cache is not None}")
        print(f"[runner] has_last_logits={gen.last_logits is not None}")
        print(f"[runner] prefill_prompt_tokens={prefill_result.prompt_tokens}")
        print(f"[runner] prefill_time_ms={prefill_result.prefill_time_ms}")

        if not gen.is_prefilled:
            raise RuntimeError("prefill did not mark gen.is_prefilled")
        if gen.is_finished:
            raise RuntimeError("prefill unexpectedly marked generation finished")
        if gen.kv_cache is None:
            raise RuntimeError("prefill did not initialize gen.kv_cache")
        if gen.last_logits is None:
            raise RuntimeError("prefill did not initialize gen.last_logits")
        if not gen.can_decode:
            raise RuntimeError("gen.can_decode should be True after prefill")

        # ---- one decode step ----
        step = runner.decode_step(gen)

        print("[runner] decode_step ok")
        print(f"[runner] step.token_id={step.token_id}")
        print(f"[runner] step.text={step.text!r}")
        print(f"[runner] step.finish_reason={step.finish_reason!r}")
        print(f"[runner] step.decode_time_ms={step.decode_time_ms}")
        print(f"[runner] last_token_id={gen.last_token_id}")
        print(f"[runner] generated_token_ids={gen.generated_token_ids}")
        print(f"[runner] completion_tokens_count={gen.completion_tokens_count}")
        print(f"[runner] total_tokens_count={gen.total_tokens_count}")
        print(f"[runner] is_finished={gen.is_finished}")
        print(f"[runner] can_decode={gen.can_decode}")
        print(f"[runner] has_last_logits={gen.last_logits is not None}")

        if gen.completion_tokens_count != 1:
            raise RuntimeError(
                f"expected completion_tokens_count == 1, got {gen.completion_tokens_count}"
            )
        if len(gen.generated_token_ids) != 1:
            raise RuntimeError(
                f"expected len(generated_token_ids) == 1, got {len(gen.generated_token_ids)}"
            )
        if gen.last_token_id != step.token_id:
            raise RuntimeError(
                f"last_token_id mismatch: gen={gen.last_token_id} step={step.token_id}"
            )
        if not gen.is_finished and gen.last_logits is None:
            raise RuntimeError("generation can continue but gen.last_logits is None")

        # ---- full generate ----
        result = runner.generate(
            input_ids=input_ids,
            sampling_config=sampling_config,
        )

        print("[runner] generate ok")
        print(f"[runner] result.request_id={result.request_id}")
        print(f"[runner] result.model_name={result.model_name!r}")
        print(f"[runner] result.output_token_ids={result.output_token_ids}")
        print(f"[runner] result.output_text={result.output_text!r}")
        print(f"[runner] result.finish_reason={result.finish_reason!r}")
        print(f"[runner] result.prompt_tokens={result.prompt_tokens}")
        print(f"[runner] result.completion_tokens={result.completion_tokens}")
        print(f"[runner] result.total_tokens={result.total_tokens}")
        print(f"[runner] result.generate_started_at={result.generate_started_at}")
        print(f"[runner] result.generate_finished_at={result.generate_finished_at}")
        print(f"[runner] result.prefill_time_ms={result.prefill_time_ms}")
        print("[runner] generation_runner passed")


if __name__ == "__main__":
    main()
