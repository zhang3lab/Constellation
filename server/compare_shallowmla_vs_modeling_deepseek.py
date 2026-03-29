import argparse
import numpy as np
import torch
from pathlib import Path

from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.inference_session import InferenceSession
from server.capture_layer3_hidden_partial import (
    build_partial_config,
    build_config_object,
    import_deepseek_modules,
    load_partial_state_dict,
    strip_model_prefix_for_base_model,
    get_layers_root,
)
from server.shallowmla_adapter import ShallowMLAAttentionWrapper
from server.test_utils import compare_arrays


class CaptureDone(Exception):
    pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--server-config", type=str, required=True)
    p.add_argument("--prompt", type=str, default="Hello world")
    p.add_argument("--token-index", type=int, default=-1)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    server_cfg = load_config(args.server_config)

    model_root = Path(server_cfg["model"]["root"])
    layer_id = int(server_cfg["run"]["layer_id"])

    pkg_root = "/root/Constellation/tmp"
    package_name = "DeepSeek_V3_1"

    config_module, modeling_module = import_deepseek_modules(pkg_root, package_name)

    cfg_dict = build_partial_config(model_root, max_layer_id=layer_id)
    config = build_config_object(config_module, cfg_dict)

    if not hasattr(modeling_module, "DeepseekV3Model"):
        raise RuntimeError("DeepseekV3Model not found in modeling_deepseek.py")
    ModelCls = modeling_module.DeepseekV3Model

    with torch.device("meta"):
        model = ModelCls(config)
    model.eval()

    raw_state = load_partial_state_dict(model_root, max_layer_id=layer_id, device=args.device)
    state = strip_model_prefix_for_base_model(raw_state)

    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    print(f"[partial-load] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("[partial-load] first missing keys:", missing[:20])
    if unexpected:
        print("[partial-load] first unexpected keys:", unexpected[:20])

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_root), trust_remote_code=True)

    captured = {}

    def hook_attn_input(module_, args, kwargs):
        captured["attn_hidden"] = kwargs["hidden_states"].detach()

        attention_mask = kwargs.get("attention_mask")
        position_ids = kwargs.get("position_ids")
        past_key_value = kwargs.get("past_key_value")
        use_cache = kwargs.get("use_cache")

        captured["attention_mask"] = attention_mask.detach().cpu() if attention_mask is not None else None
        captured["position_ids"] = position_ids.detach().cpu() if position_ids is not None else None
        captured["past_key_value_is_none"] = past_key_value is None
        captured["use_cache"] = bool(use_cache) if use_cache is not None else None

    def hook_attn_output(module_, args, kwargs, output):
        if isinstance(output, tuple):
            captured["attn_output"] = output[0].detach()
        else:
            captured["attn_output"] = output.detach()
        raise CaptureDone

    layers = get_layers_root(model)
    layer = layers[layer_id]
    h1 = layer.self_attn.register_forward_pre_hook(hook_attn_input, with_kwargs=True)
    h2 = layer.self_attn.register_forward_hook(hook_attn_output, with_kwargs=True)

    enc = tokenizer(args.prompt, return_tensors="pt")
    enc = {k: v.to(args.device) for k, v in enc.items()}

    try:
        with torch.no_grad():
            _ = model(**enc)
    except CaptureDone:
        pass
    finally:
        h1.remove()
        h2.remove()

    if "attn_hidden" not in captured or "attn_output" not in captured:
        raise RuntimeError("failed to capture attention input/output")

    attn_in = captured["attn_hidden"]      # [B, S, H]
    attn_out = captured["attn_output"]    # [B, S, H]

    if attn_in.ndim != 3 or attn_in.shape[0] != 1:
        raise RuntimeError(f"unexpected attn_input shape: {tuple(attn_in.shape)}")
    if attn_out.ndim != 3 or attn_out.shape[0] != 1:
        raise RuntimeError(f"unexpected attn_output shape: {tuple(attn_out.shape)}")

    seq_len = int(attn_in.shape[1])
    tok_idx = int(args.token_index)
    if tok_idx < 0:
        tok_idx = seq_len + tok_idx
    if tok_idx < 0 or tok_idx >= seq_len:
        raise RuntimeError(f"token_index out of range: {tok_idx}, seq_len={seq_len}")

    x_prefix = (
        attn_in[0, : tok_idx + 1]
        .detach()
        .cpu()
        .float()
        .numpy()
        .astype(np.float32, copy=False)
    )
    y_ref = (
        attn_out[0, tok_idx]
        .detach()
        .cpu()
        .float()
        .numpy()
        .reshape(-1)
        .astype(np.float32, copy=False)
    )
    mask_ref = captured["attention_mask"]   # [1,1,3,3]
    mask_for_shallow = mask_ref[0, 0].to(device=args.device, dtype=x_prefix_t.dtype)

    print(f"[capture] layer_id={layer_id} token_index={tok_idx} seq_len={seq_len}")
    print(f"[capture] x_prefix shape={x_prefix.shape} dtype={x_prefix.dtype}")
    print(f"[capture] y_ref shape={y_ref.shape} dtype={y_ref.dtype}")
    print(f"[capture] y_ref[:8]={y_ref[:8]}")

    coord = Coordinator(server_cfg["nodes"])
    setup_control_plane(coord, server_cfg)

    with InferenceSession(coord, server_cfg) as session:
        model_loader = session.get_deepseek_model_loader()
        wrapper = ShallowMLAAttentionWrapper(
            model_loader=model_loader,
            layer_id=layer_id,
            dtype=torch.float16,
            device=args.device,
            max_batch_size=1,
            optim_type="triton",
        )

        x_prefix_t = (
            torch.from_numpy(x_prefix)
            .to(device=args.device, dtype=torch.float16)
            .unsqueeze(0)
        )

        seq_len = int(x_prefix_t.shape[1])
        freq_cis = wrapper.freq_cis[:seq_len]

        with torch.no_grad():
            y_prefix = wrapper.mla(
                x_prefix_t,
                start_pos=0,
                freq_cis=freq_cis,
                mask=mask_for_shallow,
            )

        y_shallow = (
            y_prefix[0, -1]
            .detach()
            .cpu()
            .float()
            .numpy()
            .reshape(-1)
            .astype(np.float32, copy=False)
        )

    compare_arrays("modeling_deepseek_vs_shallowmla", y_ref, y_shallow)
    print("[ref     ] y[:8] =", y_ref[:8])
    print("[shallow ] y[:8] =", y_shallow[:8])

    print("[capture] attn_hidden shape =", tuple(captured["attn_hidden"].shape))

    pm = captured["position_ids"]
    print("[capture] position_ids =", None if pm is None else pm.tolist())

    am = captured["attention_mask"]
    print("[capture] attention_mask shape =", None if am is None else tuple(am.shape))
    if am is not None:
        print("[capture] attention_mask dtype =", am.dtype)
        flat = am.reshape(-1)
        print("[capture] attention_mask min/max =", flat.min().item(), flat.max().item())

    print("[capture] past_key_value_is_none =", captured["past_key_value_is_none"])
    print("[capture] use_cache =", captured["use_cache"])


if __name__ == "__main__":
    main()
