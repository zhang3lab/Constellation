from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch

from server.absorbed_latent_ref import run_attention_block_ref
from server.backbone_store import BackboneLoadPlan, preload_non_moe_backbone
from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.deepseek_full_model_executor import DeepseekFullModelExecutor
from server.full_model_types import ModelExecResult
from server.inference_session import InferenceSession
from server.tensor_cache import MappedTensorStore
from server.test.utils import compare_arrays, prenorm_hidden_for_attention, print_stats, to_numpy_f32


class DeepseekFullModelReference(DeepseekFullModelExecutor):
    def run_attention_block(
        self,
        hidden_in: np.ndarray,
        layer_id: int,
        *,
        position_ids=None,
        attention_mask=None,
        kv_cache=None,
        return_aux: bool = False,
    ) -> ModelExecResult:
        return run_attention_block_ref(
            self.session,
            hidden_in,
            int(layer_id),
            position_ids=position_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            return_aux=return_aux,
        )


def _align_semantic_shape(a: Any, b: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    a_np = to_numpy_f32(a)
    b_np = to_numpy_f32(b)

    a_sq = np.squeeze(a_np)
    b_sq = np.squeeze(b_np)

    if a_sq.shape != b_sq.shape:
        raise RuntimeError(
            f"{name}: semantic shape mismatch after squeeze: "
            f"{tuple(a_np.shape)} -> {tuple(a_sq.shape)} vs "
            f"{tuple(b_np.shape)} -> {tuple(b_sq.shape)}"
        )

    return a_sq, b_sq


def _compare_aligned(name: str, ref_v: Any, rt_v: Any) -> None:
    ref_aligned, rt_aligned = _align_semantic_shape(ref_v, rt_v, name)

    print_stats(f"ref.{name}", ref_aligned)
    print_stats(f"runtime.{name}", rt_aligned)

    if ref_aligned.shape != rt_aligned.shape:
        raise RuntimeError(
            f"{name}: shape mismatch before compare: "
            f"{tuple(ref_aligned.shape)} vs {tuple(rt_aligned.shape)}"
        )

    compare_arrays(name, ref_aligned, rt_aligned)


def run_runtime_attention(
    session: InferenceSession,
    *,
    prompt: str,
    layer_id: int,
    position_id: int,
) -> dict[str, Any]:
    session.full_model_executor = DeepseekFullModelExecutor(session)

    kv_cache_cfg = session.cfg["kv_cache"]
    session.ensure_full_model_runtime(
        tensor_cache_dir="tmp/non_moe_backbone_cache",
        split_layer=30,
        backbone_dtype=torch.bfloat16,
        kv_cache_cfg=kv_cache_cfg,
    )
    session.reset_full_model_kv_cache(kv_cache_cfg=kv_cache_cfg)

    prepared = session.full_model_executor.prepare_prompt_hidden_input(prompt)
    hidden = prepared["hidden_in"]
    hidden_prenorm = prenorm_hidden_for_attention(session, hidden, layer_id)

    pos = np.asarray([position_id], dtype=np.int64)

    out = session.full_model_executor.run_attention_block(
        hidden_prenorm,
        layer_id,
        position_ids=pos,
        attention_mask=None,
        kv_cache=session.page_attention_cache_managers,
        return_aux=True,
    )

    return {
        "hidden_in": hidden,
        "hidden_prenorm": hidden_prenorm,
        "output": to_numpy_f32(out.output),
        "aux": out.aux or {},
    }


def run_reference_attention(
    session: InferenceSession,
    *,
    prompt: str,
    layer_id: int,
    position_id: int,
) -> dict[str, Any]:
    session.full_model_executor = DeepseekFullModelReference(session)

    kv_cache_cfg = session.cfg["kv_cache"]

    if session.mapped_tensor_store is None:
        session.mapped_tensor_store = MappedTensorStore("tmp/non_moe_backbone_cache")

    if session.backbone_store is None:
        session.backbone_store = preload_non_moe_backbone(
            session,
            mapped_store=session.mapped_tensor_store,
            plan=BackboneLoadPlan.attention_only(
                attention_dtype=torch.float32,
                embed_dtype=torch.float32,
                layer_ids={int(layer_id)},
            ),
        )

    session.ensure_freq_cis_by_device(
        max_seq_len=int(kv_cache_cfg["max_seq_len"]),
    )
    session.reset_full_model_kv_cache(kv_cache_cfg=kv_cache_cfg)

    prepared = session.full_model_executor.prepare_prompt_hidden_input(prompt)
    hidden = prepared["hidden_in"]
    hidden_prenorm = prenorm_hidden_for_attention(session, hidden, layer_id)

    pos = np.asarray([position_id], dtype=np.int64)

    out = session.full_model_executor.run_attention_block(
        hidden_prenorm,
        layer_id,
        position_ids=pos,
        attention_mask=None,
        kv_cache=session.page_attention_cache_managers,
        return_aux=True,
    )

    return {
        "hidden_in": hidden,
        "hidden_prenorm": hidden_prenorm,
        "output": to_numpy_f32(out.output),
        "aux": out.aux or {},
    }


def maybe_compare_aux_tensor(
    name: str,
    ref_aux: dict[str, Any],
    rt_aux: dict[str, Any],
    key: str,
) -> None:
    if key not in ref_aux:
        print(f"[skip] ref aux missing {key}")
        return
    if key not in rt_aux:
        print(f"[skip] runtime aux missing {key}")
        return

    _compare_aligned(f"{name}.{key}", ref_aux[key], rt_aux[key])


def _try_compare_variants(name: str, ref_v: Any, rt_v: Any) -> None:
    ref_x, rt_x = _align_semantic_shape(ref_v, rt_v, name)

    print_stats(f"ref.{name}", ref_x)
    print_stats(f"runtime.{name}", rt_x)

    compare_arrays(name, ref_x, rt_x)

    if ref_x.ndim == 2 and rt_x.ndim == 2:
        if ref_x.T.shape == rt_x.shape:
            compare_arrays(f"{name}.ref_T", ref_x.T, rt_x)
        if rt_x.T.shape == ref_x.shape:
            compare_arrays(f"{name}.rt_T", ref_x, rt_x.T)
        if ref_x[::-1].shape == rt_x.shape:
            compare_arrays(f"{name}.ref_rev0", ref_x[::-1], rt_x)
        if ref_x[:, ::-1].shape == rt_x.shape:
            compare_arrays(f"{name}.ref_rev1", ref_x[:, ::-1], rt_x)
        if rt_x[::-1].shape == ref_x.shape:
            compare_arrays(f"{name}.rt_rev0", ref_x, rt_x[::-1])
        if rt_x[:, ::-1].shape == ref_x.shape:
            compare_arrays(f"{name}.rt_rev1", ref_x, rt_x[:, ::-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/config.json")
    ap.add_argument("--prompt", type=str, default="Hello world")
    ap.add_argument("--layer-id", type=int, default=3)
    ap.add_argument("--position-id", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    coord = Coordinator(cfg["nodes"])
    setup_control_plane(coord, cfg)

    with InferenceSession(coord, cfg) as runtime_sess:
        runtime_out = run_runtime_attention(
            runtime_sess,
            prompt=args.prompt,
            layer_id=args.layer_id,
            position_id=args.position_id,
        )

    with InferenceSession(coord, cfg) as ref_sess:
        ref_out = run_reference_attention(
            ref_sess,
            prompt=args.prompt,
            layer_id=args.layer_id,
            position_id=args.position_id,
        )

    _compare_aligned("hidden_in", ref_out["hidden_in"], runtime_out["hidden_in"])
    _compare_aligned("attention_output", ref_out["output"], runtime_out["output"])

    ref_aux = ref_out["aux"]
    rt_aux = runtime_out["aux"]

    if "q_flash" in ref_aux and "q_flash" in rt_aux:
        _try_compare_variants("attention.q_flash", ref_aux["q_flash"], rt_aux["q_flash"])

    if "blocked_k_token" in ref_aux and "blocked_k_token" in rt_aux:
        _try_compare_variants(
            "attention.blocked_k_token",
            ref_aux["blocked_k_token"],
            rt_aux["blocked_k_token"],
        )


if __name__ == "__main__":
    main()
