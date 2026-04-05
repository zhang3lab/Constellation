from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from server.deepseek_model_loader import DeepseekModelLoader


def get_attr_by_dotted_name(obj, dotted_name: str):
    cur = obj
    for part in dotted_name.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def copy_named_tensor_into_model(
    model,
    *,
    loader: DeepseekModelLoader,
    tensor_name: str,
) -> None:
    target = get_attr_by_dotted_name(model, tensor_name)
    loaded = loader.load_tensor_fp32_by_name(tensor_name)
    with torch.no_grad():
        target.copy_(loaded.to(device=target.device, dtype=target.dtype))


def load_hf_model(model_dir: str, device: str):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map=device,
    )
    model.eval()
    return model


def full_model_names(num_layers: int = 61) -> list[str]:
    names = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
    for layer_id in range(num_layers):
        prefix = f"model.layers.{layer_id}"
        names.extend(
            [
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.self_attn.q_a_proj.weight",
                f"{prefix}.self_attn.q_a_layernorm.weight",
                f"{prefix}.self_attn.q_b_proj.weight",
                f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
                f"{prefix}.self_attn.kv_a_layernorm.weight",
                f"{prefix}.self_attn.kv_b_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.post_attention_layernorm.weight",
            ]
        )
        if layer_id < 3:
            names.extend(
                [
                    f"{prefix}.mlp.gate_proj.weight",
                    f"{prefix}.mlp.up_proj.weight",
                    f"{prefix}.mlp.down_proj.weight",
                ]
            )
        else:
            names.extend(
                [
                    f"{prefix}.mlp.shared_experts.gate_proj.weight",
                    f"{prefix}.mlp.shared_experts.up_proj.weight",
                    f"{prefix}.mlp.shared_experts.down_proj.weight",
                    f"{prefix}.mlp.gate.weight",
                    f"{prefix}.mlp.gate.e_score_correction_bias",
                ]
            )
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--input-json", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        inp = json.load(f)
    input_ids = inp["input_ids"]

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    print("[hf-full] decoded =", repr(tok.decode(input_ids)))

    loader = DeepseekModelLoader(args.model_dir)
    model = load_hf_model(args.model_dir, args.device)
    try:
        for name in full_model_names():
            print(f"[hf-full] copy {name}")
            copy_named_tensor_into_model(model, loader=loader, tensor_name=name)

        ids = torch.tensor([input_ids], dtype=torch.long, device=next(model.parameters()).device)

        layer_hidden: dict[str, torch.Tensor] = {}

        def make_hook(name: str):
            def _hook(_module, _inp, out):
                x = out[0] if isinstance(out, tuple) else out
                layer_hidden[name] = x.detach().float().cpu()
            return _hook

        hooks = []
        for i in range(61):
            hooks.append(model.model.layers[i].register_forward_hook(make_hook(f"layer_{i}_output")))

        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False, return_dict=True)
            logits = out.logits.detach().float().cpu()

        for h in hooks:
            h.remove()

        report = {
            "backend": "hf_absorbed",
            "input_ids": input_ids,
            "decoded_input": tok.decode(input_ids),
            "saved": [],
        }

        for k, v in layer_hidden.items():
            p = outdir / f"{k}.pt"
            torch.save(v, p)
            report["saved"].append(str(p))

        p = outdir / "logits.pt"
        torch.save(logits, p)
        report["saved"].append(str(p))

        with (outdir / "hf_full_model.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(f"[hf-full] wrote {outdir / 'hf_full_model.json'}")
    finally:
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
