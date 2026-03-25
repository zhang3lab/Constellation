import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.fp8_utils import dequant_fp8_weight_blockwise
from server.inference_session import InferenceSession
from server.moe_layer_runtime import run_moe_layer, run_one_expert_reference


class StopBeforeMoE(Exception):
    pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--server-config", type=str, default="server/config.json")
    p.add_argument("--prompt", type=str, default="Hello world")
    p.add_argument("--token-index", type=int, default=-1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--pkg-root", type=str, default="/root/Constellation/tmp")
    p.add_argument("--package-name", type=str, default="DeepSeek_V3_1")
    return p.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_modeling_file(model_root: Path) -> Path:
    p = model_root / "modeling_deepseek.py"
    if not p.exists():
        raise RuntimeError(f"modeling_deepseek.py not found under {model_root}")
    return p


def import_deepseek_modules(pkg_root: str, package_name: str):
    import importlib
    import sys

    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)

    config_module = importlib.import_module(f"{package_name}.configuration_deepseek")
    modeling_module = importlib.import_module(f"{package_name}.modeling_deepseek")
    return config_module, modeling_module


def build_partial_config(model_root: Path, max_layer_id: int):
    cfg_dict = load_json(model_root / "config.json")

    cfg_dict["num_hidden_layers"] = max_layer_id + 1
    if "num_nextn_predict_layers" in cfg_dict:
        cfg_dict["num_nextn_predict_layers"] = 0

    return cfg_dict


def build_config_object(config_module, cfg_dict):
    if not hasattr(config_module, "DeepseekV3Config"):
        raise RuntimeError("DeepseekV3Config not found in configuration_deepseek")
    ConfigCls = config_module.DeepseekV3Config
    return ConfigCls(**cfg_dict)


def should_keep_tensor(name: str, max_layer_id: int):
    if name.startswith("model.embed_tokens."):
        return True

    # full keep for layers before target layer
    for i in range(max_layer_id):
        if name.startswith(f"model.layers.{i}."):
            return True

    # for target layer, keep only attention-side tensors and norms
    last = max_layer_id
    if name.startswith(f"model.layers.{last}.input_layernorm."):
        return True
    if name.startswith(f"model.layers.{last}.self_attn."):
        return True
    if name.startswith(f"model.layers.{last}.post_attention_layernorm."):
        return True

    return False


def load_partial_state_dict(model_root: Path, max_layer_id: int, device: str):
    shard_paths = sorted(model_root.rglob("*.safetensors"))
    if not shard_paths:
        raise RuntimeError(f"no safetensors found under {model_root}")

    raw = {}

    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            keep = [k for k in keys if should_keep_tensor(k, max_layer_id)]
            if not keep:
                continue

            for k in keep:
                raw[k] = f.get_tensor(k)

    state = {}

    for k, v in raw.items():
        if k.endswith(".weight_scale_inv"):
            continue

        if v.dtype == torch.float8_e4m3fn:
            scale_key = k + "_scale_inv"
            if scale_key not in raw:
                raise RuntimeError(f"missing scale tensor for fp8 weight: {k}")
            v = dequant_fp8_weight_blockwise(v, raw[scale_key]).to(torch.bfloat16)
        elif v.dtype == torch.float32:
            v = v.to(torch.float32)
        else:
            v = v.to(torch.bfloat16)

        state[k] = v.to(device)

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
    pkg_root = "/root/Constellation/tmp"
    package_name = "DeepSeek_V3_1"

    config_module, modeling_module = import_deepseek_modules(pkg_root, package_name)

    # build partial config
    cfg_dict = build_partial_config(model_root, max_layer_id=layer_id)
    config = build_config_object(config_module, cfg_dict)

    if not hasattr(modeling_module, "DeepseekV3Model"):
        raise RuntimeError("DeepseekV3Model not found in modeling_deepseek.py")
    ModelCls = modeling_module.DeepseekV3Model

    with torch.device("meta"):
        model = ModelCls(config)
    model.eval()

    # load only embedding + layers 0..layer_id
    raw_state = load_partial_state_dict(model_root, max_layer_id=layer_id, device=args.device)
    state = strip_model_prefix_for_base_model(raw_state)

    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    print(f"[partial-load] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("[partial-load] first missing keys:", missing[:20])
    if unexpected:
        print("[partial-load] first unexpected keys:", unexpected[:20])

    # tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_root), trust_remote_code=True)

    captured = {}

    def hook_mlp_input(module_, inputs):
        captured["hidden"] = inputs[0].detach()
        raise StopBeforeMoE

    layers = get_layers_root(model)
    handle = layers[layer_id].mlp.register_forward_pre_hook(hook_mlp_input)

    enc = tokenizer(args.prompt, return_tensors="pt")
    enc = {k: v.to(args.device) for k, v in enc.items()}

    try:
        with torch.no_grad():
            _ = model(**enc)
    except StopBeforeMoE:
        pass
    finally:
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

    print(
        f"[capture] hidden finite={np.isfinite(hidden_np).sum()}/{hidden_np.size} "
        f"min={hidden_np.min():.6e} max={hidden_np.max():.6e} "
        f"mean={hidden_np.mean():.6e} std={hidden_np.std():.6e}"
    )
    print("[capture] hidden[:8] =", hidden_np[:8])

    coord = Coordinator(server_cfg["nodes"])
    setup_control_plane(coord, server_cfg)

    with InferenceSession(coord, server_cfg) as session:
        y_ref = run_one_expert_reference(session, 4, hidden_np)
        print(
            "[ref] finite=", np.isfinite(y_ref).sum(), "/", y_ref.size,
            "min=", np.nanmin(y_ref),
            "max=", np.nanmax(y_ref),
            "mean=", np.nanmean(y_ref),
        )
        print("[ref] y[:8] =", y_ref[:8])

        result = run_moe_layer(session, hidden_np, layer_id, return_aux=True)
        print("[moe] routes =", result["routes"])
        print("[moe] output[:8] =", result["output"][:8])


if __name__ == "__main__":
    main()
