import numpy as np

from server.array_utils import (
    ARRCFG_HIDDEN_NUMPY_F32,
    ARRCFG_VECTOR_NUMPY_F32,
    ARRCFG_HIDDEN_TORCH,
    ARRCFG_VECTOR_TORCH,
    as_array,
    torch_dtype_name,
)
from server.deepseek_full_model_executor import (
    DeepseekFullModelExecutorBase,
    _post_attention_ffn_input,
)
from server.full_model_types import AttentionSharedSegmentResult, ModelExecResult
from server.moe_layer_runtime import run_moe_layer
from server.test.utils import print_stats


def _get_full_model_executor(session) -> DeepseekFullModelExecutorBase:
    ref = getattr(session, "full_model_executor", None)
    if ref is None:
        raise RuntimeError("session.full_model_executor is not initialized")
    return ref


def _normalize_model_exec_result(result: ModelExecResult, name: str) -> ModelExecResult:
    if not isinstance(result.output, torch.Tensor):
        raise TypeError(
            f"{name}.output expected torch.Tensor, got {type(result.output).__name__}"
        )

    dtype_name = torch_dtype_name(result.output.dtype)
    dev = str(result.output.device)

    result.output = as_array(
        result.output,
        f"{name}.output",
        ARRCFG_HIDDEN_TORCH(dtype_name, dev),
    )
    return result


def _normalize_attention_shared_result(
    result: AttentionSharedSegmentResult,
    hidden_shape,
    name: str,
) -> AttentionSharedSegmentResult:
    if not isinstance(result.attention_output, torch.Tensor):
        raise TypeError(
            f"{name}.attention_output expected torch.Tensor, "
            f"got {type(result.attention_output).__name__}"
        )
    if not isinstance(result.shared_expert_output, torch.Tensor):
        raise TypeError(
            f"{name}.shared_expert_output expected torch.Tensor, "
            f"got {type(result.shared_expert_output).__name__}"
        )

    attn_cfg = ARRCFG_HIDDEN_TORCH(
        torch_dtype_name(result.attention_output.dtype),
        str(result.attention_output.device),
    )
    shared_cfg = ARRCFG_HIDDEN_TORCH(
        torch_dtype_name(result.shared_expert_output.dtype),
        str(result.shared_expert_output.device),
    )

    result.attention_output = as_array(
        result.attention_output,
        f"{name}.attention_output",
        attn_cfg,
    )
    result.shared_expert_output = as_array(
        result.shared_expert_output,
        f"{name}.shared_expert_output",
        shared_cfg,
    )

    if tuple(result.attention_output.shape) != tuple(hidden_shape):
        raise RuntimeError(
            f"{name}.attention_output shape mismatch: "
            f"got={tuple(result.attention_output.shape)} expected={tuple(hidden_shape)}"
        )
    if tuple(result.shared_expert_output.shape) != tuple(hidden_shape):
        raise RuntimeError(
            f"{name}.shared_expert_output shape mismatch: "
            f"got={tuple(result.shared_expert_output.shape)} expected={tuple(hidden_shape)}"
        )

    return result


def run_dense_layer(
    session,
    hidden_in,
    layer_id: int,
    *,
    position_ids=None,
    attention_mask=None,
    kv_cache=None,
    return_aux: bool = False,
):
    ref = _get_full_model_executor(session)
    layer_id = int(layer_id)

    if session.backbone_store is None:
        raise RuntimeError("session.backbone_store is not initialized")

    layer_entry = session.backbone_store.layer(layer_id)
    attn_dev = str(layer_entry["device"])
    runtime_dtype = session.backbone_store.dtype
    dtype_name = torch_dtype_name(runtime_dtype)

    hidden_cfg = ARRCFG_HIDDEN_TORCH(dtype_name, attn_dev)

    hidden_in = as_array(
        hidden_in,
        f"dense_layer{layer_id}.input",
        hidden_cfg,
    )

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
    post_attn_hidden = as_array(
        post_attn_hidden,
        f"dense_layer{layer_id}.post_attention_hidden",
        hidden_cfg,
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
    output = as_array(
        output,
        f"dense_layer{layer_id}.output",
        hidden_cfg,
    )

    result = {
        "layer_id": layer_id,
        "layer_type": "dense",
        "output": output,
        "attention_output": attn.output,
        "dense_ffn_output": dense_ffn.output,
    }
    if return_aux:
        result["attention_aux"] = attn.aux
        result["dense_ffn_aux"] = dense_ffn.aux
        result["post_attention_hidden"] = post_attn_hidden.clone()
    return result


def run_sparse_layer(
    session,
    hidden_in,
    layer_id: int,
    *,
    position_ids=None,
    attention_mask=None,
    kv_cache=None,
    return_aux: bool = False,
):
    ref = _get_full_model_executor(session)
    layer_id = int(layer_id)

    if session.backbone_store is None:
        raise RuntimeError("session.backbone_store is not initialized")

    layer_entry = session.backbone_store.layer(layer_id)
    attn_dev = str(layer_entry["device"])
    runtime_dtype = session.backbone_store.dtype
    dtype_name = torch_dtype_name(runtime_dtype)

    hidden_cfg = ARRCFG_HIDDEN_TORCH(dtype_name, attn_dev)
    vector_cfg = ARRCFG_VECTOR_TORCH(dtype_name, attn_dev)

    hidden_in = as_array(
        hidden_in,
        f"sparse_layer{layer_id}.input",
        hidden_cfg,
    )
    was_1d = (hidden_in.ndim == 1)
    seq_len = 1 if was_1d else int(hidden_in.shape[0])

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
    post_attn_hidden = as_array(
        post_attn_hidden,
        f"sparse_layer{layer_id}.post_attention_hidden",
        hidden_cfg,
    )

    ffn_hidden = _post_attention_ffn_input(
        session,
        post_attn_hidden,
        layer_id,
    )
    ffn_hidden = as_array(
        ffn_hidden,
        f"sparse_layer{layer_id}.ffn_hidden",
        hidden_cfg,
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

    # routed MoE 仍然走单 token 协议；2D 时按 seq 维逐 token 调用
    if was_1d:
        routed = run_moe_layer(
            session,
            ffn_hidden,
            layer_id,
            return_aux=return_aux,
        )
        routed_out = as_array(
            routed["output"],
            f"sparse_layer{layer_id}.routed_output",
            vector_cfg,
        )
        routed_aux = routed.get("aux")
    else:
        routed_out_list = []
        routed_aux_list = []

        for tok_idx in range(seq_len):
            routed_tok = run_moe_layer(
                session,
                ffn_hidden[tok_idx],
                layer_id,
                return_aux=return_aux,
            )
            routed_tok_out = as_array(
                routed_tok["output"],
                f"sparse_layer{layer_id}.routed_output[{tok_idx}]",
                vector_cfg,
            )
            routed_out_list.append(routed_tok_out)

            if return_aux:
                routed_aux_list.append(routed_tok.get("aux"))

        routed_out = torch.stack(routed_out_list, dim=0)
        routed_out = as_array(
            routed_out,
            f"sparse_layer{layer_id}.routed_output",
            hidden_cfg,
        )
        routed_aux = routed_aux_list if return_aux else None

    if tuple(routed_out.shape) != tuple(hidden_in.shape):
        raise RuntimeError(
            f"sparse_layer{layer_id}.routed_output shape mismatch: "
            f"got={tuple(routed_out.shape)} expected={tuple(hidden_in.shape)}"
        )

    ffn_total = shared.output + routed_out
    ffn_total = as_array(
        ffn_total,
        f"sparse_layer{layer_id}.ffn_total",
        hidden_cfg,
    )

    output = post_attn_hidden + ffn_total
    output = as_array(
        output,
        f"sparse_layer{layer_id}.output",
        hidden_cfg,
    )

    result = {
        "layer_id": layer_id,
        "layer_type": "sparse",
        "output": output,
        "attention_output": attn.output,
        "shared_expert_output": shared.output,
        "routed_output": routed_out,
    }
    if return_aux:
        result["attention_aux"] = attn.aux
        result["shared_expert_aux"] = shared.aux
        result["routed_aux"] = routed_aux
        result["post_attention_hidden"] = post_attn_hidden.clone()
        result["ffn_hidden"] = ffn_hidden.clone()
        result["ffn_total"] = ffn_total.clone()
    return result


def run_full_model(
    session,
    hidden_in,
    *,
    start_layer: int,
    end_layer: int,
    position_ids=None,
    attention_mask=None,
    kv_cache=None,
    collect_per_layer: bool = False,
):
    ref = _get_full_model_executor(session)

    start_layer = int(start_layer)
    end_layer = int(end_layer)
    if end_layer < start_layer:
        raise RuntimeError(
            f"invalid layer range: start_layer={start_layer}, end_layer={end_layer}"
        )

    if session.backbone_store is None:
        raise RuntimeError("session.backbone_store is not initialized")

    runtime_dtype = session.backbone_store.dtype
    dtype_name = torch_dtype_name(runtime_dtype)

    start_layer_entry = session.backbone_store.layer(start_layer)
    start_dev = str(start_layer_entry["device"])

    hidden_in = as_array(
        hidden_in,
        "full_model.input",
        ARRCFG_HIDDEN_TORCH(dtype_name, start_dev),
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
                    "output": cur.clone(),
                    "aux": prefix.aux,
                }
            )

        next_layer = dense_prefix_end
    else:
        next_layer = start_layer

    for layer_id in range(next_layer, end_layer + 1):
        layer_entry = session.backbone_store.layer(layer_id)
        layer_dev = str(layer_entry["device"])

        cur = as_array(
            cur,
            f"full_model.layer{layer_id}.input",
            ARRCFG_HIDDEN_TORCH(dtype_name, layer_dev),
        )

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

        cur = as_array(
            layer_result["output"],
            f"full_model.layer{layer_id}.output",
            ARRCFG_HIDDEN_TORCH(dtype_name, layer_dev),
        )
        print_stats(f"layer{layer_id}_output", cur)

        if collect_per_layer:
            saved = dict(layer_result)
            saved["output"] = cur.clone()
            per_layer.append(saved)

    result = {
        "output": cur,
        "start_layer": int(start_layer),
        "end_layer": int(end_layer),
    }
    if collect_per_layer:
        result["per_layer"] = per_layer
    return result
