from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.deepseek_full_model_executor import DeepseekFullModelExecutor
from server.full_model_runtime import run_dense_layer, run_sparse_layer
from server.inference_session import InferenceSession


def save_pt(outdir: Path, name: str, x) -> str:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} expected tensor/ndarray, got {type(x).__name__}")
    p = outdir / f"{name}.pt"
    torch.save(x.detach().float().cpu(), p)
    return str(p)


def save_json(outdir: Path, name: str, obj) -> str:
    p = outdir / f"{name}.json"
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return str(p)


def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    return obj


def collect_aux_keys(aux) -> list[str]:
    if aux is None:
        return []
    if isinstance(aux, dict):
        return sorted(aux.keys())
    if isinstance(aux, list):
        s = set()
        for item in aux:
            if isinstance(item, dict):
                s.update(item.keys())
        return sorted(s)
    return [type(aux).__name__]


def maybe_save_dense_aux_tensors(outdir: Path, saved: list[str], prefix: str, aux) -> None:
    if not isinstance(aux, dict):
        return
    for name in [
        "attention_input",
        "attention_output",
        "post_attention_hidden",
        "ffn_hidden",
        "dense_ffn_output",
        "output",
    ]:
        if name in aux and isinstance(aux[name], torch.Tensor):
            saved.append(save_pt(outdir, f"{prefix}_{name}", aux[name]))


def maybe_save_sparse_aux_tensors(outdir: Path, saved: list[str], prefix: str, result: dict) -> None:
    for name in [
        "attention_input",
        "attention_output",
        "post_attention_hidden",
        "ffn_hidden",
        "shared_expert_output",
        "routed_output",
        "ffn_total",
        "output",
    ]:
        if name in result and isinstance(result[name], torch.Tensor):
            saved.append(save_pt(outdir, f"{prefix}_{name}", result[name]))


def maybe_save_moe_aux_tensors(outdir: Path, saved: list[str], prefix: str, moe_aux) -> None:
    tensor_names = [
        "router_hidden",
        "logits",
        "scores_raw",
        "scores",
        "scores_for_choice_raw",
        "scores_for_choice",
        "group_scores",
        "selected_group_idx",
        "group_mask",
        "score_mask",
        "resident_mask",
        "topk_idx",
        "topk_choice_vals",
        "topk_weight_pre_norm",
        "topk_weight_post_norm",
        "topk_weight",
        "combined_pre_cast",
    ]
    if isinstance(moe_aux, dict):
        for name in tensor_names:
            if name in moe_aux:
                saved.append(save_pt(outdir, f"{prefix}_{name}", moe_aux[name]))
    elif isinstance(moe_aux, list):
        for tok_idx, item in enumerate(moe_aux):
            if not isinstance(item, dict):
                continue
            for name in tensor_names:
                if name in item:
                    saved.append(save_pt(outdir, f"{prefix}_tok{tok_idx}_{name}", item[name]))


def maybe_save_moe_aux_json(outdir: Path, saved: list[str], prefix: str, moe_aux) -> None:
    json_names = [
        "resident_local_expert_ids",
        "routes",
        "local_routes",
        "selected_global_expert_ids",
        "selected_weights",
        "effective_top_k",
        "layer_id",
    ]
    if isinstance(moe_aux, dict):
        for name in json_names:
            if name in moe_aux:
                saved.append(save_json(outdir, f"{prefix}_{name}", to_jsonable(moe_aux[name])))
        if "weighted_outputs" in moe_aux:
            saved.append(save_json(outdir, f"{prefix}_weighted_outputs", to_jsonable(moe_aux["weighted_outputs"])))
    elif isinstance(moe_aux, list):
        saved.append(save_json(outdir, f"{prefix}_moe_aux_list", to_jsonable(moe_aux)))


def maybe_save_expert_outputs(outdir: Path, saved: list[str], prefix: str, moe_aux) -> None:
    if isinstance(moe_aux, dict):
        expert_outputs = moe_aux.get("expert_outputs")
        if isinstance(expert_outputs, list):
            meta = []
            for i, item in enumerate(expert_outputs):
                if not isinstance(item, dict):
                    continue
                meta.append(
                    {
                        "expert_id": int(item["expert_id"]) if "expert_id" in item else None,
                        "weight": float(item["weight"]) if "weight" in item else None,
                        "output_dtype": int(item["output_dtype"]) if "output_dtype" in item else None,
                    }
                )
                if "output" in item:
                    saved.append(save_pt(outdir, f"{prefix}_expert{i}_output", item["output"]))
                if "weighted_output" in item:
                    saved.append(save_pt(outdir, f"{prefix}_expert{i}_weighted_output", item["weighted_output"]))
            saved.append(save_json(outdir, f"{prefix}_expert_outputs_meta", to_jsonable(meta)))
    elif isinstance(moe_aux, list):
        for tok_idx, item in enumerate(moe_aux):
            if not isinstance(item, dict):
                continue
            expert_outputs = item.get("expert_outputs")
            if not isinstance(expert_outputs, list):
                continue
            meta = []
            for i, ex in enumerate(expert_outputs):
                if not isinstance(ex, dict):
                    continue
                meta.append(
                    {
                        "expert_id": int(ex["expert_id"]) if "expert_id" in ex else None,
                        "weight": float(ex["weight"]) if "weight" in ex else None,
                        "output_dtype": int(ex["output_dtype"]) if "output_dtype" in ex else None,
                    }
                )
                if "output" in ex:
                    saved.append(save_pt(outdir, f"{prefix}_tok{tok_idx}_expert{i}_output", ex["output"]))
                if "weighted_output" in ex:
                    saved.append(save_pt(outdir, f"{prefix}_tok{tok_idx}_expert{i}_weighted_output", ex["weighted_output"]))
            saved.append(save_json(outdir, f"{prefix}_tok{tok_idx}_expert_outputs_meta", to_jsonable(meta)))


def is_sparse_layer(model_cfg, layer_id: int) -> bool:
    first_dense_replace = int(getattr(model_cfg, "first_k_dense_replace"))
    moe_layer_freq = int(getattr(model_cfg, "moe_layer_freq"))
    n_routed_experts = getattr(model_cfg, "n_routed_experts", None)
    return (
        n_routed_experts is not None
        and layer_id >= first_dense_replace
        and (layer_id % moe_layer_freq == 0)
    )


def move_hidden_to_layer_device(session, hidden, layer_id: int):
    if session.backbone_store is None:
        raise RuntimeError("session.backbone_store is not initialized")
    layer_entry = session.backbone_store.layer(int(layer_id))
    dev = str(layer_entry["device"])
    dtype = session.backbone_store.dtype

    if not isinstance(hidden, torch.Tensor):
        hidden = torch.as_tensor(hidden)

    if str(hidden.device) != dev or hidden.dtype != dtype:
        hidden = hidden.to(device=dev, dtype=dtype)
    return hidden.contiguous()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/test/config.json")
    ap.add_argument("--model-dir", type=str, default="/model/ModelScope/deepseek-ai/DeepSeek-V3.1")
    ap.add_argument("--input-json", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--layer-id", type=int, required=True)
    args = ap.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        inp = json.load(f)

    prompt = inp.get("prompt")
    input_ids = inp.get("input_ids")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    coord = Coordinator(cfg["nodes"], log_level=cfg["log_level"])
    setup_control_plane(coord, cfg)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    if input_ids is not None:
        if not isinstance(input_ids, list):
            raise TypeError(f"input_ids must be a list, got {type(input_ids).__name__}")
        ids = [int(x) for x in input_ids]
        if len(ids) == 0:
            raise RuntimeError("input_ids must not be empty")

        if prompt is None:
            prompt = tokenizer.decode(ids)

    else:
        if prompt is None:
            raise RuntimeError("input_json must contain either prompt or input_ids")

        ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]

    decoded = tokenizer.decode(ids)
    target_layer = int(args.layer_id)

    with InferenceSession(coord, cfg) as session:
        session.full_model_executor = DeepseekFullModelExecutor(session)

        kv_cache_cfg = cfg["kv_cache"]
        session.initialize_full_model_runtime(
            tensor_cache_dir="tmp/non_moe_backbone_cache",
            split_layer=30,
            backbone_dtype=torch.bfloat16,
            kv_cache_cfg=kv_cache_cfg,
        )

        if session.backbone_store is None:
            raise RuntimeError("session.backbone_store is not initialized")

        embed_weight = session.backbone_store.embed_tokens()
        if embed_weight is None:
            raise RuntimeError("embed_tokens not loaded")

        ids_t = torch.tensor(ids, dtype=torch.long, device=embed_weight.device)
        hidden = embed_weight[ids_t].to(
            device=str(session.backbone_store.partition.embed_device()),
            dtype=session.backbone_store.dtype,
        ).contiguous()

        position_ids = np.arange(hidden.shape[0], dtype=np.int64)
        saved: list[str] = []

        model_cfg = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
        num_layers = int(getattr(model_cfg, "num_hidden_layers"))
        is_last_layer = (target_layer == num_layers - 1)

        # run previous layers, saving all outputs
        for layer_id in range(target_layer):
            hidden = move_hidden_to_layer_device(session, hidden, layer_id)

            if is_sparse_layer(model_cfg, layer_id):
                result = run_sparse_layer(
                    session,
                    hidden,
                    layer_id,
                    position_ids=position_ids,
                    attention_mask=None,
                    kv_cache=session.page_attention_cache_managers,
                    return_aux=False,
                )
            else:
                result = run_dense_layer(
                    session,
                    hidden,
                    layer_id,
                    position_ids=position_ids,
                    attention_mask=None,
                    kv_cache=session.page_attention_cache_managers,
                    return_aux=False,
                )
            hidden = result["output"]
            saved.append(save_pt(outdir, f"layer_{layer_id}_output", hidden))

        prefix = f"layer_{target_layer}"
        hidden = move_hidden_to_layer_device(session, hidden, target_layer)

        # run target layer with aux
        if is_sparse_layer(model_cfg, target_layer):
            result = run_sparse_layer(
                session,
                hidden,
                target_layer,
                position_ids=position_ids,
                attention_mask=None,
                kv_cache=session.page_attention_cache_managers,
                return_aux=True,
            )
            maybe_save_sparse_aux_tensors(outdir, saved, prefix, result)
            moe_aux = result.get("routed_aux")
            maybe_save_moe_aux_tensors(outdir, saved, prefix, moe_aux)
            maybe_save_moe_aux_json(outdir, saved, prefix, moe_aux)
            maybe_save_expert_outputs(outdir, saved, prefix, moe_aux)
            aux_keys = collect_aux_keys(moe_aux)
        else:
            result = run_dense_layer(
                session,
                hidden,
                target_layer,
                position_ids=position_ids,
                attention_mask=None,
                kv_cache=session.page_attention_cache_managers,
                return_aux=True,
            )
            if isinstance(result, dict):
                maybe_save_dense_aux_tensors(outdir, saved, prefix, result)
                aux_keys = collect_aux_keys(result.get("aux"))
            else:
                aux_keys = []

        if is_last_layer:
            if session.backbone_store is None:
                raise RuntimeError("session.backbone_store is not initialized")
         
            norm_w = session.backbone_store.model_norm()
            lm_head_w = session.backbone_store.lm_head()
         
            if norm_w is None:
                raise RuntimeError("session.backbone_store.model_norm is not initialized")
            if lm_head_w is None:
                raise RuntimeError("session.backbone_store.lm_head is not initialized")
            if not isinstance(norm_w, torch.Tensor):
                raise TypeError(f"model_norm expected torch.Tensor, got {type(norm_w).__name__}")
            if not isinstance(lm_head_w, torch.Tensor):
                raise TypeError(f"lm_head expected torch.Tensor, got {type(lm_head_w).__name__}")
         
            frms_norminal_hidden_in = result["output"]
            if not isinstance(final_hidden_in, torch.Tensor):
                final_hidden_in = torch.as_tensor(final_hidden_in)
         
            dev = str(norm_w.device)
            dtype = norm_w.dtype
         
            if str(final_hidden_in.device) != dev or final_hidden_in.dtype != dtype:
                final_hidden_in = final_hidden_in.to(device=dev, dtype=dtype)
         
            was_1d = (final_hidden_in.ndim == 1)
            x = final_hidden_in.unsqueeze(0) if was_1d else final_hidden_in
         
            eps = float(getattr(model_cfg, "rms_norm_eps", 1e-6))
            final_hidden = torch.nn.functional.rms_norm(
                x,
                (x.shape[-1],),
                norm_w,
                eps,
            )
            logits = torch.matmul(final_hidden, lm_head_w.t())
         
            saved.append(save_pt(outdir, "final_hidden", final_hidden[0] if was_1d else final_hidden))
            saved.append(save_pt(outdir, "logits", logits[0] if was_1d else logits))

        report = {
            "backend": "runtime",
            "layer_id": target_layer,
            "prompt": prompt,
            "input_ids": ids,
            "decoded_input": decoded,
            "saved": saved,
            "is_sparse": bool(is_sparse_layer(model_cfg, target_layer)),
            "aux_keys": aux_keys,
            "has_final_hidden": bool(is_last_layer),
            "has_logits": bool(is_last_layer),
        }
        with (outdir / "runtime_single_layer_manual.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(f"[runtime-single-layer] wrote {outdir / 'runtime_single_layer_manual.json'}")


if __name__ == "__main__":
    main()
