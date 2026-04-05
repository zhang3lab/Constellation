from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

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


def collect_moe_aux_keys(moe_aux) -> list[str]:
    if moe_aux is None:
        return []
    if isinstance(moe_aux, dict):
        return sorted(list(moe_aux.keys()))
    if isinstance(moe_aux, list):
        key_set = set()
        for item in moe_aux:
            if isinstance(item, dict):
                key_set.update(item.keys())
        return sorted(key_set)
    return [type(moe_aux).__name__]


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
        return

    if isinstance(moe_aux, list):
        for tok_idx, item in enumerate(moe_aux):
            if not isinstance(item, dict):
                continue
            for name in tensor_names:
                if name in item:
                    saved.append(save_pt(outdir, f"{prefix}_tok{tok_idx}_{name}", item[name]))


def maybe_save_expert_outputs(outdir: Path, saved: list[str], prefix: str, moe_aux) -> None:
    if isinstance(moe_aux, dict):
        expert_outputs = moe_aux.get("expert_outputs")
        if not isinstance(expert_outputs, list):
            return

        meta = []
        for i, item in enumerate(expert_outputs):
            if not isinstance(item, dict):
                continue

            record = {
                "expert_id": int(item["expert_id"]) if "expert_id" in item else None,
                "weight": float(item["weight"]) if "weight" in item else None,
                "output_dtype": int(item["output_dtype"]) if "output_dtype" in item else None,
            }

            if "output" in item:
                saved.append(save_pt(outdir, f"{prefix}_expert{i}_output", item["output"]))
            if "weighted_output" in item:
                saved.append(save_pt(outdir, f"{prefix}_expert{i}_weighted_output", item["weighted_output"]))

            meta.append(record)

        saved.append(save_json(outdir, f"{prefix}_expert_outputs_meta", to_jsonable(meta)))
        return

    if isinstance(moe_aux, list):
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

                record = {
                    "expert_id": int(ex["expert_id"]) if "expert_id" in ex else None,
                    "weight": float(ex["weight"]) if "weight" in ex else None,
                    "output_dtype": int(ex["output_dtype"]) if "output_dtype" in ex else None,
                }

                if "output" in ex:
                    saved.append(save_pt(outdir, f"{prefix}_tok{tok_idx}_expert{i}_output", ex["output"]))
                if "weighted_output" in ex:
                    saved.append(save_pt(outdir, f"{prefix}_tok{tok_idx}_expert{i}_weighted_output", ex["weighted_output"]))

                meta.append(record)

            saved.append(
                save_json(
                    outdir,
                    f"{prefix}_tok{tok_idx}_expert_outputs_meta",
                    to_jsonable(meta),
                )
            )


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
        return

    if isinstance(moe_aux, list):
        saved.append(save_json(outdir, f"{prefix}_moe_aux_list", to_jsonable(moe_aux)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/test/config.json")
    ap.add_argument("--model-dir", type=str, default="/model/ModelScope/deepseek-ai/DeepSeek-V3.1")
    ap.add_argument("--input-json", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    args = ap.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        inp = json.load(f)

    prompt = inp.get("prompt")
    input_ids = inp.get("input_ids")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    coord = Coordinator(cfg["nodes"])
    setup_control_plane(coord, cfg)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if prompt is None:
        if input_ids is None:
            raise RuntimeError("input_json must contain either prompt or input_ids")
        prompt = tokenizer.decode(input_ids)

    ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    decoded = tokenizer.decode(ids)

    with InferenceSession(coord, cfg) as session:
        session.full_model_executor = DeepseekFullModelExecutor(session)

        kv_cache_cfg = cfg["kv_cache"]
        session.ensure_full_model_runtime(
            tensor_cache_dir="tmp/non_moe_backbone_cache",
            split_layer=30,
            backbone_dtype=torch.bfloat16,
            kv_cache_cfg=kv_cache_cfg,
        )
        session.reset_full_model_kv_cache(kv_cache_cfg=kv_cache_cfg)

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

        for layer_id in [0, 1, 2]:
            result = run_dense_layer(
                session,
                hidden,
                layer_id,
                position_ids=position_ids,
                attention_mask=None,
                kv_cache=session.page_attention_cache_managers,
                return_aux=True,
            )
            hidden = result["output"]
            saved.append(save_pt(outdir, f"layer_{layer_id}_output", hidden))

        result = run_sparse_layer(
            session,
            hidden,
            4,
            position_ids=position_ids,
            attention_mask=None,
            kv_cache=session.page_attention_cache_managers,
            return_aux=True,
        )

        saved.append(save_pt(outdir, "layer_4_attention_output", result["attention_output"]))
        saved.append(save_pt(outdir, "layer_4_post_attention_hidden", result["post_attention_hidden"]))
        saved.append(save_pt(outdir, "layer_4_ffn_hidden", result["ffn_hidden"]))
        saved.append(save_pt(outdir, "layer_4_shared_expert_output", result["shared_expert_output"]))
        saved.append(save_pt(outdir, "layer_4_routed_output", result["routed_output"]))
        saved.append(save_pt(outdir, "layer_4_ffn_total", result["ffn_total"]))
        saved.append(save_pt(outdir, "layer_4_output", result["output"]))

        moe_aux = result.get("moe_aux")
        moe_aux_keys = collect_moe_aux_keys(moe_aux)

        maybe_save_moe_aux_tensors(outdir, saved, "layer_3", moe_aux)
        maybe_save_moe_aux_json(outdir, saved, "layer_3", moe_aux)
        maybe_save_expert_outputs(outdir, saved, "layer_3", moe_aux)

        report = {
            "backend": "runtime",
            "prompt": prompt,
            "input_ids": ids,
            "decoded_input": decoded,
            "saved": saved,
            "moe_aux_keys": moe_aux_keys,
        }
        with (outdir / "runtime_layer4_manual.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(f"[runtime-layer4] wrote {outdir / 'runtime_layer4_manual.json'}")


if __name__ == "__main__":
    main()
