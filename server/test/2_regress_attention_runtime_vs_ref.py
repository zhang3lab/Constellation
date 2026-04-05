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
    compare_arrays(name, ref_aligned, rt_aligned)


def interleaved_to_half_split(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"last dim must be even, got {x.shape[-1]}")
    d_half = x.shape[-1] // 2
    y = x.reshape(*x.shape[:-1], d_half, 2)
    axes = list(range(y.ndim))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    y = y.transpose(axes)
    return y.reshape(*x.shape[:-1], x.shape[-1])


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


def compare_q_flash(ref_aux: dict[str, Any], rt_aux: dict[str, Any]) -> None:
    required_ref = ["q_flash"]
    required_rt = ["q_flash", "q_rope_post_rotary"]

    for k in required_ref:
        if k not in ref_aux:
            print(f"[skip] ref aux missing {k}")
            return
    for k in required_rt:
        if k not in rt_aux:
            print(f"[skip] runtime aux missing {k}")
            return

    ref_q_flash = np.squeeze(to_numpy_f32(ref_aux["q_flash"]))
    rt_q_flash = np.squeeze(to_numpy_f32(rt_aux["q_flash"]))
    rt_q_rope = np.squeeze(to_numpy_f32(rt_aux["q_rope_post_rotary"]))

    rope_dim = rt_q_rope.shape[-1]
    prefix_dim = rt_q_flash.shape[-1] - rope_dim
    if prefix_dim < 0:
        raise RuntimeError(
            f"attention.q_flash: invalid dims, total={rt_q_flash.shape[-1]} rope={rope_dim}"
        )

    ref_prefix = ref_q_flash[..., :prefix_dim]
    ref_rope = ref_q_flash[..., prefix_dim:]
    ref_rope_hf = interleaved_to_half_split(ref_rope)
    ref_q_flash_hf = np.concatenate([ref_prefix, ref_rope_hf], axis=-1)

    _compare_aligned("attention.q_flash", ref_q_flash_hf, rt_q_flash)


def compare_blocked_k_token(ref_aux: dict[str, Any], rt_aux: dict[str, Any]) -> None:
    required_ref = ["blocked_k_token"]
    required_rt = ["blocked_k_token", "cache_latent"]

    for k in required_ref:
        if k not in ref_aux:
            print(f"[skip] ref aux missing {k}")
            return
    for k in required_rt:
        if k not in rt_aux:
            print(f"[skip] runtime aux missing {k}")
            return

    ref_blocked_k = np.squeeze(to_numpy_f32(ref_aux["blocked_k_token"]))
    rt_blocked_k = np.squeeze(to_numpy_f32(rt_aux["blocked_k_token"]))
    rt_cache_latent = np.squeeze(to_numpy_f32(rt_aux["cache_latent"]))

    latent_dim = rt_cache_latent.shape[-1]

    ref_latent = ref_blocked_k[..., :latent_dim]
    ref_k_rope = ref_blocked_k[..., latent_dim:]
    ref_k_rope_hf = interleaved_to_half_split(ref_k_rope)
    ref_blocked_k_hf = np.concatenate([ref_latent, ref_k_rope_hf], axis=-1)

    _compare_aligned("attention.blocked_k_token", ref_blocked_k_hf, rt_blocked_k)


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

    compare_q_flash(ref_aux, rt_aux)
    compare_blocked_k_token(ref_aux, rt_aux)


if __name__ == "__main__":
    main()
