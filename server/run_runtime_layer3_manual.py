from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.deepseek_full_model_executor import DeepseekFullModelExecutor
from server.full_model_runtime import run_dense_layer, run_sparse_layer
from server.inference_session import InferenceSession


def save_pt(outdir: Path, name: str, x) -> str:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} expected torch.Tensor, got {type(x).__name__}")
    p = outdir / f"{name}.pt"
    torch.save(x.detach().float().cpu(), p)
    return str(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/test/config.json")
    ap.add_argument("--model-dir", type=str, default="/root/DeepSeek-V3.1-partial")
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

        position_ids = torch.arange(hidden.shape[0], dtype=torch.long)

        saved: list[str] = []

        # layer 0,1,2 dense prefix one by one
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

        # layer 3 sparse with full aux
        result = run_sparse_layer(
            session,
            hidden,
            3,
            position_ids=position_ids,
            attention_mask=None,
            kv_cache=session.page_attention_cache_managers,
            return_aux=True,
        )

        saved.append(save_pt(outdir, "layer_3_attention_output", result["attention_output"]))
        saved.append(save_pt(outdir, "layer_3_post_attention_hidden", result["post_attention_hidden"]))
        saved.append(save_pt(outdir, "layer_3_ffn_hidden", result["ffn_hidden"]))
        saved.append(save_pt(outdir, "layer_3_shared_expert_output", result["shared_expert_output"]))
        saved.append(save_pt(outdir, "layer_3_routed_output", result["routed_output"]))
        saved.append(save_pt(outdir, "layer_3_ffn_total", result["ffn_total"]))
        saved.append(save_pt(outdir, "layer_3_output", result["output"]))

        report = {
            "backend": "runtime",
            "prompt": prompt,
            "input_ids": ids,
            "decoded_input": decoded,
            "saved": saved,
        }
        with (outdir / "runtime_layer3_manual.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(f"[runtime-layer3] wrote {outdir / 'runtime_layer3_manual.json'}")


if __name__ == "__main__":
    main()
