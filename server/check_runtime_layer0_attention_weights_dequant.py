from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.deepseek_full_model_executor import DeepseekFullModelExecutor
from server.inference_session import InferenceSession
from server.deepseek_model_loader import DeepseekModelLoader


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def compare(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    print(f"=== {name} ===")
    print("runtime shape:", tuple(a.shape))
    print("dequant shape:", tuple(b.shape))
    if a.shape != b.shape:
        print("shape mismatch")
        return
    diff = (a - b).abs()
    print("cosine  =", cosine_similarity(a, b))
    print("max_abs =", float(diff.max().item()))
    print("mean_abs=", float(diff.mean().item()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/test/config.json")
    ap.add_argument("--raw-model-dir", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    coord = Coordinator(cfg["nodes"])
    setup_control_plane(coord, cfg)
    kv_cache_cfg = cfg["kv_cache"]

    loader = DeepseekModelLoader(args.raw_model_dir)

    with InferenceSession(coord, cfg) as session:
        session.full_model_executor = DeepseekFullModelExecutor(session)
        session.ensure_full_model_runtime(
            tensor_cache_dir="tmp/non_moe_backbone_cache",
            split_layer=30,
            backbone_dtype=torch.bfloat16,
            kv_cache_cfg=kv_cache_cfg,
        )

        layer_entry = session.backbone_store.layer(0)
        attn = layer_entry["attention"]

        mapping = {
            "q_a_proj": "model.layers.0.self_attn.q_a_proj.weight",
            "q_a_layernorm": "model.layers.0.self_attn.q_a_layernorm.weight",
            "q_b_proj": "model.layers.0.self_attn.q_b_proj.weight",
            "kv_a_proj_with_mqa": "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
            "kv_a_layernorm": "model.layers.0.self_attn.kv_a_layernorm.weight",
            "kv_b_proj": "model.layers.0.self_attn.kv_b_proj.weight",
            "o_proj": "model.layers.0.self_attn.o_proj.weight",
            "input_layernorm": "model.layers.0.input_layernorm.weight",
        }

        for rt_name, raw_name in mapping.items():
            runtime_tensor = attn[rt_name] if rt_name in attn else layer_entry[rt_name]
            dequant_tensor = loader.load_tensor_fp32_by_name(raw_name)
            compare(rt_name, runtime_tensor, dequant_tensor)


if __name__ == "__main__":
    main()
