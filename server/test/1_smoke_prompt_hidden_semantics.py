from __future__ import annotations

import argparse
import numpy as np
import torch

from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.deepseek_full_model_executor import DeepseekFullModelExecutor
from server.generation_types import PrefillResult
from server.inference_session import InferenceSession
from server.prefill_runtime import run_prefill
from server.test.utils import compare_arrays, print_stats, to_numpy_f32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/test/config.json")
    ap.add_argument("--prompt", type=str, default="Hello world")
    args = ap.parse_args()

    cfg = load_config(args.config)
    coord = Coordinator(cfg["nodes"], log_level=cfg["log_level"])
    setup_control_plane(coord, cfg)

    run_cfg = cfg["run"]
    kv_cache_cfg = cfg["kv_cache"]

    start_layer = int(run_cfg["start_layer"])
    end_layer = int(run_cfg["end_layer"])

    with InferenceSession(coord, cfg) as session:
        session.full_model_executor = DeepseekFullModelExecutor(session)

        session.initialize_full_model_runtime(
            tensor_cache_dir="tmp/non_moe_backbone_cache",
            split_layer=30,
            backbone_dtype=torch.bfloat16,
            kv_cache_cfg=kv_cache_cfg,
        )

        if session.page_attention_cache_managers is None:
            raise RuntimeError("session.page_attention_cache_managers is not initialized")
        if not isinstance(session.page_attention_cache_managers, dict):
            raise RuntimeError("session.page_attention_cache_managers must be a dict")

        prefill_result = run_prefill(
            session,
            prompt=args.prompt,
            start_layer=start_layer,
            end_layer=end_layer,
            kv_cache=session.page_attention_cache_managers,
            collect_per_layer=False,
        )

        if not isinstance(prefill_result, PrefillResult):
            raise TypeError(
                f"run_prefill(...) expected PrefillResult, got {type(prefill_result).__name__}"
            )

        aux = prefill_result.aux
        if aux is None:
            raise RuntimeError("prefill_result.aux must not be None")

        input_ids = aux.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            raise RuntimeError("prefill_result.aux['input_ids'] must be torch.Tensor")
        if input_ids.ndim == 2:
            if input_ids.shape[0] != 1:
                raise RuntimeError(
                    f"prefill_result.aux['input_ids'] expected shape [T] or [1, T], got {tuple(input_ids.shape)}"
                )
            input_ids = input_ids[0]
        elif input_ids.ndim != 1:
            raise RuntimeError(
                f"prefill_result.aux['input_ids'] expected shape [T] or [1, T], got {tuple(input_ids.shape)}"
            )
        if input_ids.numel() <= 0:
            raise RuntimeError("prefill_result.aux['input_ids'] must be non-empty")

        last_hidden_t = aux.get("last_hidden")
        if not isinstance(last_hidden_t, torch.Tensor):
            raise TypeError(
                f"prefill_result.aux['last_hidden'] expected torch.Tensor, got {type(last_hidden_t).__name__}"
            )

        if last_hidden_t.ndim != 1:
            raise RuntimeError(
                f"prefill_result.aux['last_hidden'] expected shape [H], got {tuple(last_hidden_t.shape)}"
            )

        last_token_id = int(input_ids[-1].item())
        embed_hidden_t = session.full_model_executor.embed_token_ids(
            torch.tensor([last_token_id], dtype=torch.long)
        )
        if not isinstance(embed_hidden_t, torch.Tensor):
            raise TypeError(
                "executor.embed_token_ids(torch.tensor([last_token_id], dtype=torch.long)) "
                f"expected torch.Tensor, got {type(embed_hidden_t).__name__}"
            )

        if embed_hidden_t.ndim == 2:
            if embed_hidden_t.shape[0] != 1:
                raise RuntimeError(
                    f"embed hidden expected shape [1, H] or [H], got {tuple(embed_hidden_t.shape)}"
                )
            embed_hidden_t = embed_hidden_t[0]
        elif embed_hidden_t.ndim != 1:
            raise RuntimeError(
                f"embed hidden expected shape [1, H] or [H], got {tuple(embed_hidden_t.shape)}"
            )

        last_hidden = to_numpy_f32(last_hidden_t)
        embed_hidden = to_numpy_f32(embed_hidden_t)

        print(f"[semantics] prompt={args.prompt!r}")
        print(f"[semantics] input_ids={input_ids.tolist()}")
        print(f"[semantics] last_token_id={last_token_id}")
        print(f"[semantics] prompt_tokens={prefill_result.prompt_tokens}")
        print(f"[semantics] next_position={prefill_result.next_position}")

        print_stats("semantics.prefill_last_hidden", last_hidden)
        print_stats("semantics.embed_last_token", embed_hidden)
        compare_arrays("prefill_last_hidden_vs_embed_last_token", last_hidden, embed_hidden)

        if not np.isfinite(last_hidden).all():
            raise RuntimeError("prefill last_hidden contains non-finite values")
        if not np.isfinite(embed_hidden).all():
            raise RuntimeError("embed hidden contains non-finite values")

        if last_hidden.shape != embed_hidden.shape:
            raise RuntimeError(
                f"shape mismatch: last_hidden={last_hidden.shape} embed_hidden={embed_hidden.shape}"
            )

        if np.allclose(last_hidden, embed_hidden):
            raise RuntimeError(
                "unexpected semantics: prefill last_hidden should not collapse to raw token embedding"
            )

        print("[semantics] prompt hidden semantics check finished")


if __name__ == "__main__":
    main()
