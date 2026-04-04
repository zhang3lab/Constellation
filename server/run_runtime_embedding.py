from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.deepseek_full_model_executor import DeepseekFullModelExecutor
from server.inference_session import InferenceSession


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

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected tensor-like hidden result, got {type(x).__name__}")
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/test/config.json")
    ap.add_argument("--input-json", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    args = ap.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        inp = json.load(f)
    input_ids = inp["input_ids"]

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    coord = Coordinator(cfg["nodes"])
    setup_control_plane(coord, cfg)

    with InferenceSession(coord, cfg) as session:
        session.full_model_executor = DeepseekFullModelExecutor(session)

        executor = session.full_model_executor
        hidden = executor.embed_token_ids(input_ids)
        hidden = _normalize_hidden(hidden).detach().float().cpu()

        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)

        torch.save(hidden, outdir / "embedding.pt")

        report = {
            "backend": "runtime",
            "input_ids": input_ids,
            "saved": [str(outdir / "embedding.pt")],
        }
        with (outdir / "runtime_embedding.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(f"[runtime-embed] wrote {outdir / 'runtime_embedding.json'}")


if __name__ == "__main__":
    main()
