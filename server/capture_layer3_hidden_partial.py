import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.inference_session import InferenceSession
from server.moe_layer_runtime import run_moe_layer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--server-config", type=str, default="server/config.json")
    p.add_argument("--prompt", type=str, default="Hello world")
    p.add_argument("--token-index", type=int, default=-1)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_modeling_file(model_root: Path) -> Path:
    p = model_root / "modeling_deepseek.py"
    if not p.exists():
        raise RuntimeError(f"modeling_deepseek.py not found under {model_root}")
    return p


def import_deepseek_module(model_root: Path):
    modeling_file = find_modeling_file(model_root)
    module_name = "local_modeling_deepseek_partial"

    spec = importlib.util.spec_from_file_location(module_name, modeling_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to create import spec for {modeling_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_partial_config(model_root: Path, max_layer_id: int):
    cfg_dict = load_json(model_root / "config.json")

    cfg_dict["num_hidden_layers"] = max_layer_id + 1
    if "num_nextn_predict_layers" in cfg_dict:
        cfg_dict["num_nextn_predict_layers"] = 0

    return cfg_dict


def build_config_object(module, cfg_dict):
    # remote code config class name from DeepSeek config
    if not hasattr(module, "DeepseekV3Config"):
        raise RuntimeError("DeepseekV3Config not found in modeling/config module path")
    ConfigCls = module.DeepseekV3Config
    return ConfigCls(**cfg_dict)


def list_needed_tensor_names(max_layer_id: int):
    needed_prefixes = ["model.embed_tokens."]
    for i in range(max_layer_id + 1):
        needed_prefixes.append(f"model.layers.{i}.")
    return needed_prefixes


def should_keep_tensor(name: str, needed_prefixes):
    return any(name.startswith(p) for p in needed_prefixes)


def load_partial_state_dict(model_root: Path, max_layer_id: int):
    shard_paths = sorted(model_root.rglob("*.safetensors"))
    if not shard_paths:
        raise RuntimeError(f"no safetensors found under {model_root}")

    needed_prefixes = list_needed_tensor_names(max_layer_id)
    state = {}

    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            keep = [k for k in keys if should_keep_tensor(k, needed_prefixes)]
            if not keep:
                continue

            for k in keep:
                state[k] = f.get_tensor(k)

    return state


def strip_model_prefix_for_base_model(state_dict):
    # If DeepseekV3Model expects keys without leading "model.", strip it.
    out = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            out[k[len("model."):]] = v
        else:
            out[k] = v
    return out


def get_layers_root(model):
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("cannot find layers on partial model")


def get_embed_tokens(model):
    if hasattr(model, "embed_tokens"):
        return model.embed_tokens
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    raise RuntimeError("cannot find embed_tokens on partial model")


def main():
    args = parse_args()
    server_cfg = load_config(args.server_config)

    model_root = Path(server_cfg["model"]["root"])
    layer_id = int(server_cfg["run"]["layer_id"])

    # import DeepSeek remote code locally
    module = import_deepseek_module(model_root)

    # build partial config
    cfg_dict = build_partial_config(model_root, max_layer_id=layer_id)
    config = build_config_object(module, cfg_dict)

    if not hasattr(module, "DeepseekV3Model"):
        raise RuntimeError("DeepseekV3Model not found in modeling_deepseek.py")
    ModelCls = module.DeepseekV3Model

    model = ModelCls(config)
    model.to(args.device)
    model.eval()

    # load only embedding + layers 0..layer_id
    raw_state = load_partial_state_dict(model_root, max_layer_id=layer_id)
    state = strip_model_prefix_for_base_model(raw_state)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[partial-load] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("[partial-load] first missing keys:", missing[:20])
    if unexpected:
        print("[partial-load] first unexpected keys:", unexpected[:20])

    # tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_root), trust_remote_code=True)

    captured = {}

    def hook_mlp_input(module_, inputs, output):
        captured["hidden"] = inputs[0].detach()

    layers = get_layers_root(model)
    handle = layers[layer_id].mlp.register_forward_hook(hook_mlp_input)

    enc = tokenizer(args.prompt, return_tensors="pt")
    enc = {k: v.to(args.device) for k, v in enc.items()}

    with torch.no_grad():
        _ = model(**enc)

    handle.remove()

    if "hidden" not in captured:
        raise RuntimeError("failed to capture mlp input")

    hidden_t = captured["hidden"]  # [B, S, H]
    if hidden_t.ndim != 3 or hidden_t.shape[0] != 1:
        raise RuntimeError(f"unexpected hidden shape: {tuple(hidden_t.shape)}")

    seq_len = hidden_t.shape[1]
    tok_idx = args.token_index
    if tok_idx < 0:
        tok_idx = seq_len + tok_idx
    if tok_idx < 0 or tok_idx >= seq_len:
        raise RuntimeError(f"token_index out of range: {tok_idx}, seq_len={seq_len}")

    hidden_np = (
        hidden_t[0, tok_idx]
        .detach()
        .cpu()
        .float()
        .numpy()
        .reshape(-1)
        .astype(np.float32, copy=False)
    )

    print(f"[capture] layer_id={layer_id} token_index={tok_idx}")
    print(f"[capture] hidden shape={hidden_np.shape} dtype={hidden_np.dtype}")
    print(f"[capture] hidden[:8]={hidden_np[:8]}")

    coord = Coordinator(server_cfg["nodes"])
    setup_control_plane(coord, server_cfg)

    with InferenceSession(coord, server_cfg) as session:
        result = run_moe_layer(session, hidden_np, layer_id, return_aux=True)
        print("[moe] routes =", result["routes"])
        print("[moe] output[:8] =", result["output"][:8])


if __name__ == "__main__":
    main()
