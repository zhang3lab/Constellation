import numpy as np

from server.array_utils import as_f32_1d
from server.deepseek_full_model_ref import (
    DeepseekFullModelExecutorBase,
    _post_attention_ffn_input,
)
from server.full_model_ref import AttentionSharedSegmentResult, ModelExecResult
from server.moe_layer_runtime import run_moe_layer
from server.test_utils import print_stats


def _get_full_model_executor(session) -> DeepseekFullModelExecutorBase:
    ref = getattr(session, "full_model_executor", None)
    if ref is None:
        raise RuntimeError("session.full_model_executor is not initialized")
    return ref


def _normalize_model_exec_result(result: ModelExecResult, name: str) -> ModelExecResult:
    result.output = as_f32_1d(result.output, f"{name}.output")
    return result


def _normalize_attention_shared_result(
    result: AttentionSharedSegmentResult,
    hidden_shape,
    name: str,
) -> AttentionSharedSegmentResult:
    result.attention_output = as_f32_1d(
        result.attention_output,
        f"{name}.attention_output",
    )
    result.shared_expert_output = as_f32_1d(
        result.shared_expert_output,
        f"{name}.shared_expert_output",
    )

    if result.attention_output.shape != hidden_shape:
        raise RuntimeError(
            f"{name}.attention_output shape mismatch: "
            f"got={result.attention_output.shape} expected={hidden_shape}"
        )
    if result.shared_expert_output.shape != hidden_shape:
        raise RuntimeError(
            f"{name}.shared_expert_output shape mismatch: "
            f"got={result.shared_expert_output.shape} expected={hidden_shape}"
        )

    return result


def run_dense_layer(
    session,
    hidden_in: np.ndarray,
    layer_id: int,
    *,
    position_ids=None,
    attention_mask=None,
    kv_cache=None,
    return_aux: bool = False,
):
    ref = _get_full_model_executor(session)
    hidden_in = as_f32_1d(hidden_in, f"dense_layer{layer_id}.input")

    attn = ref.run_attention_block(
        hidden_in,
        layer_id,
        position_ids=position_ids,
        attention_mask=attention_mask,
        kv_cache=kv_cache,
        return_aux=return_aux,
    )
    attn = _normalize_model_exec_result(attn, f"dense_layer{layer_id}.attention")

    post_attn_hidden = hidden_in + attn.output
    post_attn_hidden = as_f32_1d(
        post_attn_hidden,
        f"dense_layer{layer_id}.post_attention_hidden",
    )

    dense_ffn = ref.run_dense_ffn_block(
        post_attn_hidden,
        layer_id,
        return_aux=return_aux,
    )
    dense_ffn = _normalize_model_exec_result(
        dense_ffn,
        f"dense_layer{layer_id}.dense_ffn",
    )

    output = post_attn_hidden + dense_ffn.output
    output = as_f32_1d(output, f"dense_layer{layer_id}.output")

    result = {
        "layer_id": int(layer_id),
        "layer_type": "dense",
        "output": output,
        "attention_output": attn.output,
        "dense_ffn_output": dense_ffn.output,
    }
    if return_aux:
        result["attention_aux"] = attn.aux
        result["dense_ffn_aux"] = dense_ffn.aux
        result["post_attention_hidden"] = post_attn_hidden.copy()
    return result


def run_sparse_layer(
    session,
    hidden_in: np.ndarray,
    layer_id: int,
    *,
    position_ids=None,
    attention_mask=None,
    kv_cache=None,
    return_aux: bool = False,
):
    ref = _get_full_model_executor(session)
    hidden_in = as_f32_1d(hidden_in, f"sparse_layer{layer_id}.input")

    attn = ref.run_attention_block(
        hidden_in,
        layer_id,
        position_ids=position_ids,
        attention_mask=attention_mask,
        kv_cache=kv_cache,
        return_aux=return_aux,
    )
    attn = _normalize_model_exec_result(attn, f"sparse_layer{layer_id}.attention")

    post_attn_hidden = hidden_in + attn.output
    post_attn_hidden = as_f32_1d(
        post_attn_hidden,
        f"sparse_layer{layer_id}.post_attention_hidden",
    )

    ffn_hidden = _post_attention_ffn_input(
        session,
        post_attn_hidden,
        layer_id,
    )
    ffn_hidden = as_f32_1d(
        ffn_hidden,
        f"sparse_layer{layer_id}.ffn_hidden",
    )

    shared = ref.run_shared_expert_block(
        ffn_hidden,
        layer_id,
        return_aux=return_aux,
    )
    shared = _normalize_model_exec_result(
        shared,
        f"sparse_layer{layer_id}.shared_expert",
    )

    routed = run_moe_layer(
        session,
        ffn_hidden,
        layer_id,
        return_aux=return_aux,
    )
    routed_out = as_f32_1d(
        routed["output"],
        f"sparse_layer{layer_id}.routed_output",
    )
    if routed_out.shape != hidden_in.shape:
        raise RuntimeError(
            f"sparse_layer{layer_id}.routed_output shape mismatch: "
            f"got={routed_out.shape} expected={hidden_in.shape}"
        )

    ffn_total = shared.output + routed_out
    ffn_total = as_f32_1d(ffn_total, f"sparse_layer{layer_id}.ffn_total")

    output = post_attn_hidden + ffn_total
    output = as_f32_1d(output, f"sparse_layer{layer_id}.output")

    result = {
        "layer_id": int(layer_id),
        "layer_type": "sparse",
        "output": output,
        "attention_output": attn.output,
        "shared_expert_output": shared.output,
        "routed_output": routed_out,
    }
    if return_aux:
        result["attention_aux"] = attn.aux
        result["shared_expert_aux"] = shared.aux
        result["routed_aux"] = routed.get("aux")
        result["post_attention_hidden"] = post_attn_hidden.copy()
        result["ffn_hidden"] = ffn_hidden.copy()
        result["ffn_total"] = ffn_total.copy()
    return result


def run_full_model(
    session,
    hidden_in: np.ndarray,
    *,
    start_layer: int,
    end_layer: int,
    position_ids=None,
    attention_mask=None,
    kv_cache=None,
    collect_per_layer: bool = False,
):
    ref = _get_full_model_executor(session)
    hidden_in = as_f32_1d(hidden_in, "full_model.input")

    start_layer = int(start_layer)
    end_layer = int(end_layer)
    if end_layer < start_layer:
        raise RuntimeError(
            f"invalid layer range: start_layer={start_layer}, end_layer={end_layer}"
        )

    cur = hidden_in
    per_layer = []

    dense_prefix_end = min(end_layer + 1, ref.dense_layer_count())

    if start_layer < dense_prefix_end:
        prefix = ref.run_prefix_segment(
            cur,
            start_layer=start_layer,
            end_layer=dense_prefix_end - 1,
            position_ids=position_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            return_aux=collect_per_layer,
        )
        prefix = _normalize_model_exec_result(prefix, "full_model.prefix_segment")
        cur = prefix.output
        print_stats(f"layer{dense_prefix_end - 1}_output", cur)

        if collect_per_layer:
            per_layer.append(
                {
                    "segment_type": "prefix",
                    "start_layer": int(start_layer),
                    "end_layer": int(dense_prefix_end - 1),
                    "output": cur.copy(),
                    "aux": prefix.aux,
                }
            )

        next_layer = dense_prefix_end
    else:
        next_layer = start_layer

    for layer_id in range(next_layer, end_layer + 1):
        if ref.is_sparse_layer(layer_id):
            layer_result = run_sparse_layer(
                session,
                cur,
                layer_id,
                position_ids=position_ids,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                return_aux=collect_per_layer,
            )
        else:
            layer_result = run_dense_layer(
                session,
                cur,
                layer_id,
                position_ids=position_ids,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                return_aux=collect_per_layer,
            )

        cur = as_f32_1d(
            layer_result["output"],
            f"full_model.layer{layer_id}.output",
        )
        print_stats(f"layer{layer_id}_output", cur)

        if collect_per_layer:
            saved = dict(layer_result)
            saved["output"] = cur.copy()
            per_layer.append(saved)

    result = {
        "output": cur,
        "start_layer": int(start_layer),
        "end_layer": int(end_layer),
    }
    if collect_per_layer:
        result["per_layer"] = per_layer
    return result
