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
            if isinstance(cur, torch.nn.ModuleDict):
                cur = cur[part]
            else:
                try:
                    cur = cur[int(part)]
                except Exception:
                    cur = getattr(cur, part)
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


def load_restricted_expert_ids_from_model_dir(model_dir: str) -> list[int]:
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"missing config.json under {model_dir}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    resident = cfg.get("resident_expert_ids")
    if not isinstance(resident, list):
        raise RuntimeError(
            f"{config_path}: expected resident_expert_ids to be a list, got {type(resident).__name__}"
        )

    return sorted({int(x) for x in resident})


def is_sparse_layer(cfg, layer_id: int) -> bool:
    first_dense_replace = int(getattr(cfg, "first_k_dense_replace"))
    moe_layer_freq = int(getattr(cfg, "moe_layer_freq"))
    n_routed_experts = getattr(cfg, "n_routed_experts", None)
    return (
        n_routed_experts is not None
        and layer_id >= first_dense_replace
        and (layer_id % moe_layer_freq == 0)
    )


def names_full_model(cfg, restricted_local_expert_ids: list[int]) -> list[str]:
    num_layers = int(getattr(cfg, "num_hidden_layers"))
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

        if is_sparse_layer(cfg, layer_id):
            names.extend(
                [
                    f"{prefix}.mlp.shared_experts.gate_proj.weight",
                    f"{prefix}.mlp.shared_experts.up_proj.weight",
                    f"{prefix}.mlp.shared_experts.down_proj.weight",
                    f"{prefix}.mlp.gate.weight",
                    f"{prefix}.mlp.gate.e_score_correction_bias",
                ]
            )
            for expert_id in restricted_local_expert_ids:
                names.extend(
                    [
                        f"{prefix}.mlp.experts.{expert_id}.gate_proj.weight",
                        f"{prefix}.mlp.experts.{expert_id}.up_proj.weight",
                        f"{prefix}.mlp.experts.{expert_id}.down_proj.weight",
                    ]
                )
        else:
            names.extend(
                [
                    f"{prefix}.mlp.gate_proj.weight",
                    f"{prefix}.mlp.up_proj.weight",
                    f"{prefix}.mlp.down_proj.weight",
                ]
            )

    return names


def save_tensor(outdir: Path, name: str, tensor: torch.Tensor, saved: list[str]) -> None:
    p = outdir / f"{name}.pt"
    torch.save(tensor.detach().float().cpu(), p)
    saved.append(str(p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--input-json", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        inp = json.load(f)

    prompt = inp.get("prompt")
    input_ids = inp.get("input_ids")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if prompt is None:
        if input_ids is None:
            raise RuntimeError("input_json must contain either prompt or input_ids")
        prompt = tok.decode(input_ids)

    ids = tok(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"][0].tolist()
    decoded = tok.decode(ids)

    restricted_local_expert_ids = load_restricted_expert_ids_from_model_dir(args.model_dir)
    print(f"[hf-full] restricted_local_expert_ids={restricted_local_expert_ids}")

    loader = DeepseekModelLoader(args.model_dir)
    model = load_hf_model(args.model_dir, args.device)
    cfg = model.config

    try:
        copy_names = names_full_model(cfg, restricted_local_expert_ids)
        for name in copy_names:
            print(f"[hf-full] copy {name}")
            copy_named_tensor_into_model(model, loader=loader, tensor_name=name)

        ids_t = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)

        layer_outputs: dict[str, torch.Tensor] = {}
        misc_outputs: dict[str, torch.Tensor] = {}

        def make_hook(name: str):
            def _hook(_module, _inp, out):
                x = out[0] if isinstance(out, tuple) else out
                if not isinstance(x, torch.Tensor):
                    raise TypeError(f"hook {name}: expected tensor, got {type(x).__name__}")
                if x.ndim >= 1 and x.shape[0] == 1:
                    x = x.squeeze(0)
                if name.startswith("layer_"):
                    layer_outputs[name] = x.detach().float().cpu()
                else:
                    misc_outputs[name] = x.detach().float().cpu()
            return _hook

        hooks = []
        num_layers = int(getattr(cfg, "num_hidden_layers"))
        for i in range(num_layers):
            hooks.append(model.model.layers[i].register_forward_hook(make_hook(f"layer_{i}_output")))

        if hasattr(model.model, "norm"):
            hooks.append(model.model.norm.register_forward_hook(make_hook("final_hidden")))
        if hasattr(model, "lm_head"):
            hooks.append(model.lm_head.register_forward_hook(make_hook("final_logits")))

        with torch.no_grad():
            outputs = model(input_ids=ids_t, use_cache=False, return_dict=True)

        for h in hooks:
            h.remove()

        saved: list[str] = []

        for i in range(num_layers):
            name = f"layer_{i}_output"
            if name not in layer_outputs:
                raise RuntimeError(f"missing hook output for {name}")
            save_tensor(outdir, name, layer_outputs[name], saved)

        if "final_hidden" in misc_outputs:
            save_tensor(outdir, "final_hidden", misc_outputs["final_hidden"], saved)

        if "final_logits" in misc_outputs:
            save_tensor(outdir, "final_logits", misc_outputs["final_logits"], saved)
        elif hasattr(outputs, "logits") and isinstance(outputs.logits, torch.Tensor):
            x = outputs.logits
            if x.ndim >= 1 and x.shape[0] == 1:
                x = x.squeeze(0)
            save_tensor(outdir, "final_logits", x, saved)

        lm_head_weight = model.lm_head.weight.detach().float().cpu()
        save_tensor(outdir, "lm_head_weight", lm_head_weight, saved)

        norm_weight = model.model.norm.weight.detach().float().cpu()
        save_tensor(outdir, "final_norm_weight", norm_weight, saved)

        report = {
            "backend": "hf_absorbed",
            "prompt": prompt,
            "input_ids": ids,
            "decoded_input": decoded,
            "saved": saved,
            "num_hidden_layers": num_layers,
            "restricted_local_expert_ids": restricted_local_expert_ids,
        }
        with (outdir / "hf_full_model_manual.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(f"[hf-full] wrote {outdir / 'hf_full_model_manual.json'}")
    finally:
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
