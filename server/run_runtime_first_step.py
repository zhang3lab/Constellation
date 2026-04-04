from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.deepseek_full_model_executor import DeepseekFullModelExecutor
from server.generation_runner import GenerationRunner
from server.inference_session import InferenceSession


def topk_summary(logits: torch.Tensor, k: int) -> tuple[list[int], list[float]]:
    vals, ids = torch.topk(logits, k=k)
    return (
        [int(x) for x in ids.tolist()],
        [float(x) for x in vals.tolist()],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/test/config.json")
    ap.add_argument("--input-json", type=str, required=True)
    ap.add_argument("--output-json", type=str, required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--save-logits", type=str, default=None)
    args = ap.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        inp = json.load(f)

    input_ids = inp["input_ids"]
    if not isinstance(input_ids, list) or not input_ids:
        raise ValueError("input_json must contain non-empty list input_ids")

    cfg = load_config(args.config)
    coord = Coordinator(cfg["nodes"])
    setup_control_plane(coord, cfg)
    kv_cache_cfg = cfg["kv_cache"]

    with InferenceSession(coord, cfg) as session:
        session.full_model_executor = DeepseekFullModelExecutor(session)

        session.ensure_full_model_runtime(
            tensor_cache_dir="tmp/non_moe_backbone_cache",
            split_layer=30,
            backbone_dtype=torch.bfloat16,
            kv_cache_cfg=kv_cache_cfg,
        )
        session.reset_full_model_kv_cache(kv_cache_cfg=kv_cache_cfg)

        runner = GenerationRunner(session)
        gen = runner.create_generation()
        prefill_result = runner.prefill(gen, input_ids)

        if gen.last_logits is None:
            raise RuntimeError("runtime prefill did not produce last_logits")

        logits = gen.last_logits.detach().float().cpu()
        topk_ids, topk_vals = topk_summary(logits, args.topk)

        report = {
            "backend": "runtime",
            "model_name": gen.model_name,
            "input_ids": input_ids,
            "prompt_tokens": gen.prompt_tokens_count,
            "prefill_time_ms": float(prefill_result.prefill_time_ms),
            "argmax": int(torch.argmax(logits).item()),
            "topk_ids": topk_ids,
            "topk_vals": topk_vals,
            "logits_shape": list(logits.shape),
            "dtype": str(logits.dtype),
        }

        if args.save_logits:
            logits_path = Path(args.save_logits)
            logits_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(logits, logits_path)
            report["logits_path"] = str(logits_path)
            print(f"[runtime] saved logits to {logits_path}")

        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(f"[runtime] wrote {output_path}")
        print(f"[runtime] argmax = {report['argmax']}")
        print(f"[runtime] topk_ids = {topk_ids}")
        print(f"[runtime] topk_vals = {topk_vals}")
        print(f"[runtime] prefill_time_ms = {report['prefill_time_ms']:.3f}")
