import argparse
import json
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel

from server.config import load_config
from server.coordinator import Coordinator
from server.inference_session import InferenceSession
from server.control_plane import setup_control_plane
from server.moe_layer_runtime import run_moe_layer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--server-config", type=str, default="server/config.json")
    p.add_argument("--prompt", type=str, default="Hello world")
    p.add_argument("--token-index", type=int, default=-1,
                   help="which token in the sequence to extract; -1 means last token")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    server_cfg = load_config(args.server_config)

    model_root = str(server_cfg["model"]["root"])
    layer_id = int(server_cfg["run"]["layer_id"])

    model_cfg = AutoConfig.from_pretrained(model_root, trust_remote_code=True)
    if hasattr(model_cfg, "quantization_config"):
        delattr(model_cfg, "quantization_config")

    # 只保留前 layer_id+1 层
    model_cfg.num_hidden_layers = layer_id + 1

    # 可选：如果这个字段在配置里存在，也关掉额外 nextn 预测层
    if hasattr(model_cfg, "num_nextn_predict_layers"):
        model_cfg.num_nextn_predict_layers = 0

    tokenizer = AutoTokenizer.from_pretrained(model_root, trust_remote_code=True)

    model = AutoModel.from_pretrained(
        model_root,
        config=model_cfg,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(args.device)

    captured = {}

    def hook_mlp_input(module, inputs, output):
        # inputs[0]: [batch, seq, hidden]
        x = inputs[0]
        captured["hidden"] = x.detach()

    handle = model.model.layers[layer_id].mlp.register_forward_hook(hook_mlp_input)

    enc = tokenizer(args.prompt, return_tensors="pt")
    enc = {k: v.to(args.device) for k, v in enc.items()}

    with torch.no_grad():
        _ = model(**enc)

    handle.remove()

    if "hidden" not in captured:
        raise RuntimeError("failed to capture mlp input hidden")

    hidden_t = captured["hidden"]  # [B, S, H]
    if hidden_t.ndim != 3 or hidden_t.shape[0] != 1:
        raise RuntimeError(f"unexpected captured hidden shape: {tuple(hidden_t.shape)}")

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

    print(f"[capture] prompt={args.prompt!r}")
    print(f"[capture] layer_id={layer_id} token_index={tok_idx}")
    print(f"[capture] hidden shape={hidden_np.shape} dtype={hidden_np.dtype}")
    print(f"[capture] hidden[:8]={hidden_np[:8]}")

    coord = Coordinator(server_cfg["nodes"])
    setup_control_plane(coord, server_cfg)

    with InferenceSession(coord, server_cfg) as session:
        result = run_moe_layer(session, hidden_np, layer_id, return_aux=True)

        print("[moe] routes =", result["routes"])
        print("[moe] topk_idx =", result["aux"]["topk_idx"])
        print("[moe] topk_weight =", result["aux"]["topk_weight"])
        print("[moe] output[:8] =", result["output"][:8])


if __name__ == "__main__":
    main()
