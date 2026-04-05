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
from server.full_model_runtime import run_full_model
from server.inference_session import InferenceSession


def to_torch_f32_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu()
    raise TypeError(f"expected torch.Tensor, got {type(x).__name__}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/test/config.json")
    ap.add_argument("--model-dir", type=str, default="tmp/deepseek_restricted_ref")
    ap.add_argument("--input-json", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--start-layer", type=int, default=0)
    ap.add_argument("--end-layer", type=int, default=60)
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

        if prompt is None:
            if input_ids is None:
                raise RuntimeError("input_json must contain either prompt or input_ids")
            prompt = tokenizer.decode(input_ids)

        prepared = session.full_model_executor.prepare_prompt_hidden_input(prompt)
        hidden = prepared["hidden_in"]

        result = run_full_model(
            session,
            hidden,
            start_layer=args.start_layer,
            end_layer=args.end_layer,
            position_ids=prepared.get("position_ids"),
            attention_mask=prepared.get("attention_mask"),
            kv_cache=session.page_attention_cache_managers,
            collect_per_layer=True,
        )

        logits_result = session.full_model_executor.run_final_norm_and_lm_head(
            result["output"],
            return_aux=False,
        )

        report = {
            "backend": "runtime",
            "input_ids": input_ids,
            "prompt": prompt,
            "start_layer": int(args.start_layer),
            "end_layer": int(args.end_layer),
            "saved": [],
        }

        for item in result.get("per_layer", []):
            if "layer_id" not in item:
                continue
            layer_id = int(item["layer_id"])
            p = outdir / f"layer_{layer_id}_output.pt"
            torch.save(to_torch_f32_cpu(item["output"]), p)
            report["saved"].append(str(p))

        p = outdir / "logits.pt"
        torch.save(to_torch_f32_cpu(logits_result.output), p)
        report["saved"].append(str(p))

        with (outdir / "runtime_full_model.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(f"[runtime-full] wrote {outdir / 'runtime_full_model.json'}")


if __name__ == "__main__":
    main()
