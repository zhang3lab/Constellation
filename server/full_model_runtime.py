import numpy as np
import torch

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
    if not isinstance(attn.output, torch.Tensor):
        raise TypeError(
            f"dense_layer{layer_id}.attention.output expected torch.Tensor, "
            f"got {type(attn.output).__name__}"
        )
    attn_out = as_array(
        attn.output,
        f"dense_layer{layer_id}.attention.output",
        hidden_cfg,
    )

    post_attn_hidden = hidden_in + attn_out
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
    if not isinstance(dense_ffn.output, torch.Tensor):
        raise TypeError(
            f"dense_layer{layer_id}.dense_ffn.output expected torch.Tensor, "
            f"got {type(dense_ffn.output).__name__}"
        )
    dense_ffn_out = as_array(
        dense_ffn.output,
        f"dense_layer{layer_id}.dense_ffn.output",
        hidden_cfg,
    )

    output = post_attn_hidden + dense_ffn_out
    output = as_array(
        output,
        f"dense_layer{layer_id}.output",
        hidden_cfg,
    )

    result = {
        "layer_id": layer_id,
        "layer_type": "dense",
        "output": output,
        "attention_output": attn_out,
        "dense_ffn_output": dense_ffn_out,
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
    if not isinstance(attn.output, torch.Tensor):
        raise TypeError(
            f"sparse_layer{layer_id}.attention.output expected torch.Tensor, "
            f"got {type(attn.output).__name__}"
        )
    attn_out = as_array(
        attn.output,
        f"sparse_layer{layer_id}.attention.output",
        hidden_cfg,
    )

    post_attn_hidden = hidden_in + attn_out
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
    if not isinstance(shared.output, torch.Tensor):
        raise TypeError(
            f"sparse_layer{layer_id}.shared_expert.output expected torch.Tensor, "
            f"got {type(shared.output).__name__}"
        )
    shared_out = as_array(
        shared.output,
        f"sparse_layer{layer_id}.shared_expert.output",
        hidden_cfg,
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
        "attention_output": attn_out,
        "shared_expert_output": shared_out,
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
        if not isinstance(prefix.output, torch.Tensor):
            raise TypeError(
                f"full_model.prefix_segment.output expected torch.Tensor, "
                f"got {type(prefix.output).__name__}"
            )
        prefix_layer_dev = str(session.backbone_store.layer(dense_prefix_end - 1)["device"])
        prefix_cfg = ARRCFG_HIDDEN_TORCH(dtype_name, prefix_layer_dev)
        cur = as_array(
            prefix.output,
            "full_model.prefix_segment.output",
            prefix_cfg,
        )
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
