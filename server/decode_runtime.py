from __future__ import annotations

import numpy as np
import torch

from server.array_utils import ARRCFG_HIDDEN_TORCH, as_array, torch_dtype_name
from server.full_model_runtime import run_full_model


def run_decode(
    session,
    *,
    prompt: str,
    max_new_tokens: int,
    start_layer: int,
    end_layer: int,
    kv_cache,
    strategy: str = "greedy",
    topk: int = 5,
    temperature: float = 1.0,
    collect_per_step: bool = False,
):
    if strategy != "greedy":
        raise RuntimeError(f"unsupported decode strategy: {strategy}")

    if temperature <= 0:
        raise RuntimeError(f"temperature must be > 0, got {temperature}")

    executor = session.full_model_executor
    tokenizer = session.get_deepseek_model_loader().load_tokenizer()

    prepared = executor.prepare_prompt_hidden_input(prompt)
    current_hidden = prepared["hidden_in"]
    if not isinstance(current_hidden, torch.Tensor):
        raise TypeError(
            f'prepared["hidden_in"] expected torch.Tensor, got {type(current_hidden).__name__}'
        )

    prompt_token_ids = [int(x) for x in prepared["input_ids"]]
    generated_ids = list(prompt_token_ids)
    current_pos = executor.infer_prompt_last_position(prepared)

    per_step = []

    for step in range(int(max_new_tokens)):
        result = run_full_model(
            session,
            current_hidden,
            start_layer=int(start_layer),
            end_layer=int(end_layer),
            position_ids=np.asarray([current_pos], dtype=np.int64),
            attention_mask=None,
            kv_cache=kv_cache,
            collect_per_layer=False,
        )

        final_hidden = result["output"]
        if not isinstance(final_hidden, torch.Tensor):
            raise TypeError(
                f'run_full_model(... )["output"] expected torch.Tensor, got {type(final_hidden).__name__}'
            )

        final_hidden = as_array(
            final_hidden,
            f"decode.step{step}.final_hidden",
            ARRCFG_HIDDEN_TORCH(
                torch_dtype_name(final_hidden.dtype),
                str(final_hidden.device),
            ),
        )
        if not torch.all(torch.isfinite(final_hidden)).item():
            raise RuntimeError(f"non-finite final_hidden at decode step {step}")

        logits_result = executor.run_final_norm_and_lm_head(
            final_hidden,
            return_aux=False,
        )
        logits = logits_result.output
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                f"logits_result.output expected torch.Tensor, got {type(logits).__name__}"
            )

        logits = as_array(
            logits,
            f"decode.step{step}.logits",
            ARRCFG_HIDDEN_TORCH(
                torch_dtype_name(logits.dtype),
                str(logits.device),
            ),
        )
        if not torch.all(torch.isfinite(logits)).item():
            raise RuntimeError(f"non-finite logits at decode step {step}")

        if logits.ndim != 1:
            raise RuntimeError(
                f"decode step {step} expected 1D logits for single-token decode, got shape={tuple(logits.shape)}"
            )

        if strategy == "greedy":
            topk_vals, topk_ids = torch.topk(logits, k=int(topk))
            next_token_id = int(topk_ids[0].item())
        else:
            raise RuntimeError(f"unsupported decode strategy: {strategy}")

        generated_ids.append(next_token_id)

        if collect_per_step:
            per_step.append(
                {
                    "step": step,
                    "position": current_pos,
                    "final_hidden": final_hidden.detach().clone(),
                    "logits_top_ids": [int(x) for x in topk_ids.detach().cpu().tolist()],
                    "logits_top_vals": [float(x) for x in topk_vals.detach().cpu().tolist()],
                    "next_token_id": next_token_id,
                    "next_token_text": tokenizer.decode([next_token_id]),
                    "text_so_far": tokenizer.decode(generated_ids),
                }
            )

        current_hidden = executor.embed_token_id(next_token_id)
        current_pos += 1

    return {
        "prompt": prompt,
        "prompt_token_ids": prompt_token_ids,
        "generated_token_ids": generated_ids,
        "generated_text": tokenizer.decode(generated_ids),
        "final_position": current_pos,
        "strategy": strategy,
        "per_step": per_step,
    }
