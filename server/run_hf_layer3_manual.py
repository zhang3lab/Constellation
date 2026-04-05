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


def names_0_to_3(restricted_local_expert_ids: list[int] | None = None) -> list[str]:
    names = ["model.embed_tokens.weight"]
    for layer_id in [0, 1, 2, 3]:
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

            if restricted_local_expert_ids is not None:
                for expert_id in sorted({int(x) for x in restricted_local_expert_ids}):
                    names.extend(
                        [
                            f"{prefix}.mlp.experts.{expert_id}.gate_proj.weight",
                            f"{prefix}.mlp.experts.{expert_id}.up_proj.weight",
                            f"{prefix}.mlp.experts.{expert_id}.down_proj.weight",
                        ]
                    )
    return names


def save_tensor_if_present(saved: list[str], outdir: Path, dbg: dict, src: str, dst: str | None = None) -> None:
    x = dbg.get(src)
    if isinstance(x, torch.Tensor):
        p = outdir / (dst or f"{src}.pt")
        torch.save(x.detach().float().cpu(), p)
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
    print(f"[hf-layer3] restricted_local_expert_ids={restricted_local_expert_ids}")

    loader = DeepseekModelLoader(args.model_dir)
    model = load_hf_model(args.model_dir, args.device)

    try:
        for name in names_0_to_3(restricted_local_expert_ids):
            print(f"[hf-layer3] copy {name}")
            copy_named_tensor_into_model(model, loader=loader, tensor_name=name)

        ids_t = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)

        layer_outputs: dict[str, torch.Tensor] = {}

        def make_hook(name: str):
            def _hook(_module, _inp, out):
                x = out[0] if isinstance(out, tuple) else out
                layer_outputs[name] = x.detach().float().cpu()
            return _hook

        hooks = []
        for i in [0, 1, 2]:
            hooks.append(model.model.layers[i].register_forward_hook(make_hook(f"layer_{i}_output")))

        with torch.no_grad():
            _ = model(input_ids=ids_t, use_cache=False, return_dict=True)

        for h in hooks:
            h.remove()

        dbg = getattr(model.model.layers[3], "last_debug", {}) or {}
        router_dbg = getattr(model.model.layers[3].mlp.gate, "last_router_debug", {}) or {}

        saved: list[str] = []

        for name, tensor in layer_outputs.items():
            p = outdir / f"{name}.pt"
            torch.save(tensor, p)
            saved.append(str(p))

        save_tensor_if_present(saved, outdir, dbg, "attention_output", "layer_3_attention_output.pt")
        save_tensor_if_present(saved, outdir, dbg, "post_attention_hidden", "layer_3_post_attention_hidden.pt")
        save_tensor_if_present(saved, outdir, dbg, "ffn_hidden", "layer_3_ffn_hidden.pt")
        save_tensor_if_present(saved, outdir, dbg, "shared_expert_output", "layer_3_shared_expert_output.pt")
        save_tensor_if_present(saved, outdir, dbg, "routed_output", "layer_3_routed_output.pt")
        save_tensor_if_present(saved, outdir, dbg, "ffn_total", "layer_3_ffn_total.pt")
        save_tensor_if_present(saved, outdir, dbg, "layer_output", "layer_3_output.pt")

        save_tensor_if_present(saved, outdir, router_dbg, "logits", "layer_3_logits.pt")
        save_tensor_if_present(saved, outdir, router_dbg, "scores", "layer_3_scores.pt")
        save_tensor_if_present(saved, outdir, router_dbg, "scores_for_choice", "layer_3_scores_for_choice.pt")
        save_tensor_if_present(saved, outdir, router_dbg, "group_scores", "layer_3_group_scores.pt")
        save_tensor_if_present(saved, outdir, router_dbg, "selected_group_idx", "layer_3_selected_group_idx.pt")
        save_tensor_if_present(saved, outdir, router_dbg, "score_mask", "layer_3_score_mask.pt")
        save_tensor_if_present(saved, outdir, router_dbg, "topk_choice_vals", "layer_3_topk_choice_vals.pt")
        save_tensor_if_present(saved, outdir, router_dbg, "topk_idx", "layer_3_topk_idx.pt")
        save_tensor_if_present(saved, outdir, router_dbg, "topk_weight", "layer_3_topk_weight.pt")
        save_tensor_if_present(saved, outdir, router_dbg, "resident_mask", "layer_3_resident_mask.pt")

        save_tensor_if_present(saved, outdir, dbg, "topk_ids", "layer_3_moe_topk_ids.pt")
        save_tensor_if_present(saved, outdir, dbg, "topk_weight", "layer_3_moe_topk_weight.pt")
        save_tensor_if_present(saved, outdir, dbg, "restored_by_token", "layer_3_moe_restored_by_token.pt")
        save_tensor_if_present(saved, outdir, dbg, "weighted_by_token", "layer_3_moe_weighted_by_token.pt")
        save_tensor_if_present(saved, outdir, dbg, "final_out", "layer_3_moe_final_out.pt")

        report = {
            "backend": "hf_absorbed",
            "prompt": prompt,
            "input_ids": ids,
            "decoded_input": decoded,
            "saved": saved,
            "debug_keys": sorted(list(dbg.keys())),
            "router_debug_keys": sorted(list(router_dbg.keys())),
        }
        with (outdir / "hf_layer3_manual.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(f"[hf-layer3] wrote {outdir / 'hf_layer3_manual.json'}")
    finally:
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
