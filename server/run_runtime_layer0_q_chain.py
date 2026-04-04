from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.deepseek_full_model_executor import DeepseekFullModelExecutor
from server.inference_session import InferenceSession
from server.test.utils import prenorm_hidden_for_attention


def _normalize_hidden(x):
    if isinstance(x, dict):
        if "hidden" in x:
            x = x["hidden"]
        elif "hidden_out" in x:
            x = x["hidden_out"]
        elif "output" in x:
            x = x["output"]
    elif hasattr(x, "hidden"):
        x = x.hidden
    elif hasattr(x, "hidden_out"):
        x = x.hidden_out
    elif hasattr(x, "output"):
        x = x.output

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected tensor-like hidden result, got {type(x).__name__}")
    return x


def _extract_aux(out) -> dict:
    if hasattr(out, "aux"):
        return out.aux or {}
    if isinstance(out, dict) and "aux" in out:
        return out["aux"] or {}
    return {}


def _save_numpy_or_tensor(outdir: Path, name: str, x) -> str:
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x).float().cpu()
    elif isinstance(x, torch.Tensor):
        t = x.detach().float().cpu()
    else:
        raise TypeError(f"{name}: unsupported type {type(x).__name__}")
    p = outdir / f"{name}.pt"
    torch.save(t, p)
    return str(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/test/config.json")
    ap.add_argument("--input-json", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--layer-id", type=int, default=0)
    args = ap.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        inp = json.load(f)
    input_ids = inp["input_ids"]
    layer_id = int(args.layer_id)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

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

        executor = session.full_model_executor

        hidden = executor.embed_token_ids(input_ids)
        hidden = _normalize_hidden(hidden)
        hidden_prenorm = prenorm_hidden_for_attention(session, hidden, layer_id)

        pos = np.arange(len(input_ids), dtype=np.int64)

        out = executor.run_attention_block(
            hidden_prenorm,
            layer_id,
            position_ids=pos,
            attention_mask=None,
            kv_cache=session.page_attention_cache_managers,
            return_aux=True,
        )
        aux = _extract_aux(out)

        saved = []
        for name in [
            "q_latent_pre_norm",
            "q_latent_post_norm",
            "q_pre_split",
            "q_rope_pre_rotary",
            "q_rope_post_rotary",
        ]:
            if name in aux:
                saved.append(_save_numpy_or_tensor(outdir, name, aux[name]))

        report = {
            "backend": "runtime",
            "layer_id": layer_id,
            "input_ids": input_ids,
            "saved": saved,
            "aux_keys": sorted(list(aux.keys())),
        }
        with (outdir / "runtime_layer0_q_chain.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(f"[runtime-qchain] wrote {outdir / 'runtime_layer0_q_chain.json'}")


if __name__ == "__main__":
    main()
