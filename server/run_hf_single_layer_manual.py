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


def names_to_target_layer(cfg, restricted_local_expert_ids: list[int], target_layer: int) -> list[str]:
    names = ["model.embed_tokens.weight"]

    for layer_id in range(target_layer + 1):
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
    ap.add_argument("--layer-id", type=int, required=True)
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

    target_layer = int(args.layer_id)
    restricted_local_expert_ids = load_restricted_expert_ids_from_model_dir(args.model_dir)
    print(f"[hf-single-layer] target_layer={target_layer}")
    print(f"[hf-single-layer] restricted_local_expert_ids={restricted_local_expert_ids}")

    loader = DeepseekModelLoader(args.model_dir)
    model = load_hf_model(args.model_dir, args.device)
    cfg = model.config

    try:
        for name in names_to_target_layer(cfg, restricted_local_expert_ids, target_layer):
            print(f"[hf-single-layer] copy {name}")
            copy_named_tensor_into_model(model, loader=loader, tensor_name=name)

        ids_t = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)

        layer_outputs: dict[str, torch.Tensor] = {}

        def make_hook(name: str):
            def _hook(_module, _inp, out):
                x = out[0] if isinstance(out, tuple) else out
                if not isinstance(x, torch.Tensor):
                    raise TypeError(f"hook {name}: expected tensor, got {type(x).__name__}")
                if x.ndim >= 1 and x.shape[0] == 1:
                    x = x.squeeze(0)
                layer_outputs[name] = x.detach().float().cpu()
            return _hook

        hooks = []
        for i in range(target_layer):
            hooks.append(model.model.layers[i].register_forward_hook(make_hook(f"layer_{i}_output")))

        with torch.no_grad():
            _ = model(input_ids=ids_t, use_cache=False, return_dict=True)

        for h in hooks:
            h.remove()

        dbg = getattr(model.model.layers[target_layer], "last_debug", {}) or {}
        router_dbg = {}
        if is_sparse_layer(cfg, target_layer):
            router_dbg = getattr(model.model.layers[target_layer].mlp.gate, "last_router_debug", {}) or {}

        saved: list[str] = []

        for i in range(target_layer):
            name = f"layer_{i}_output"
            if name not in layer_outputs:
                raise RuntimeError(f"missing hook output for {name}")
            p = outdir / f"{name}.pt"
            torch.save(layer_outputs[name], p)
            saved.append(str(p))

        prefix = f"layer_{target_layer}"

        if is_sparse_layer(cfg, target_layer):
            save_tensor_if_present(saved, outdir, dbg, "attention_input", f"{prefix}_attention_input.pt")
            save_tensor_if_present(saved, outdir, dbg, "attention_output", f"{prefix}_attention_output.pt")
            save_tensor_if_present(saved, outdir, dbg, "post_attention_hidden", f"{prefix}_post_attention_hidden.pt")
            save_tensor_if_present(saved, outdir, dbg, "ffn_hidden", f"{prefix}_ffn_hidden.pt")
            save_tensor_if_present(saved, outdir, dbg, "shared_expert_output", f"{prefix}_shared_expert_output.pt")
            save_tensor_if_present(saved, outdir, dbg, "routed_output", f"{prefix}_routed_output.pt")
            save_tensor_if_present(saved, outdir, dbg, "ffn_total", f"{prefix}_ffn_total.pt")
            save_tensor_if_present(saved, outdir, dbg, "layer_output", f"{prefix}_output.pt")

            save_tensor_if_present(saved, outdir, router_dbg, "logits", f"{prefix}_logits.pt")
            save_tensor_if_present(saved, outdir, router_dbg, "scores", f"{prefix}_scores.pt")
            save_tensor_if_present(saved, outdir, router_dbg, "scores_for_choice", f"{prefix}_scores_for_choice.pt")
            save_tensor_if_present(saved, outdir, router_dbg, "group_scores", f"{prefix}_group_scores.pt")
            save_tensor_if_present(saved, outdir, router_dbg, "selected_group_idx", f"{prefix}_selected_group_idx.pt")
            save_tensor_if_present(saved, outdir, router_dbg, "score_mask", f"{prefix}_score_mask.pt")
            save_tensor_if_present(saved, outdir, router_dbg, "topk_choice_vals", f"{prefix}_topk_choice_vals.pt")
            save_tensor_if_present(saved, outdir, router_dbg, "topk_idx", f"{prefix}_topk_idx.pt")
            save_tensor_if_present(saved, outdir, router_dbg, "topk_weight", f"{prefix}_topk_weight.pt")
            save_tensor_if_present(saved, outdir, router_dbg, "resident_mask", f"{prefix}_resident_mask.pt")

            save_tensor_if_present(saved, outdir, dbg, "topk_ids", f"{prefix}_moe_topk_ids.pt")
            save_tensor_if_present(saved, outdir, dbg, "topk_weight", f"{prefix}_moe_topk_weight.pt")
            save_tensor_if_present(saved, outdir, dbg, "flat_topk_ids", f"{prefix}_moe_flat_topk_ids.pt")
            save_tensor_if_present(saved, outdir, dbg, "idxs", f"{prefix}_moe_idxs.pt")
            save_tensor_if_present(saved, outdir, dbg, "kept_positions", f"{prefix}_moe_kept_positions.pt")
            save_tensor_if_present(saved, outdir, dbg, "gatherd_idxs", f"{prefix}_moe_gatherd_idxs.pt")
            save_tensor_if_present(saved, outdir, dbg, "outs", f"{prefix}_moe_outs.pt")
            save_tensor_if_present(saved, outdir, dbg, "restored_sorted", f"{prefix}_moe_restored_sorted.pt")
            save_tensor_if_present(saved, outdir, dbg, "inv_idxs", f"{prefix}_moe_inv_idxs.pt")
            save_tensor_if_present(saved, outdir, dbg, "restored_flat", f"{prefix}_moe_restored_flat.pt")
            save_tensor_if_present(saved, outdir, dbg, "restored_by_token", f"{prefix}_moe_restored_by_token.pt")
            save_tensor_if_present(saved, outdir, dbg, "weighted_by_token", f"{prefix}_moe_weighted_by_token.pt")
            save_tensor_if_present(saved, outdir, dbg, "final_out", f"{prefix}_moe_final_out.pt")

            aux_keys = sorted(set(dbg.keys()) | set(router_dbg.keys()))
        else:
            save_tensor_if_present(saved, outdir, dbg, "attention_input", f"{prefix}_attention_input.pt")
            save_tensor_if_present(saved, outdir, dbg, "attention_output", f"{prefix}_attention_output.pt")
            save_tensor_if_present(saved, outdir, dbg, "post_attention_hidden", f"{prefix}_post_attention_hidden.pt")
            save_tensor_if_present(saved, outdir, dbg, "ffn_hidden", f"{prefix}_ffn_hidden.pt")
            save_tensor_if_present(saved, outdir, dbg, "dense_ffn_output", f"{prefix}_dense_ffn_output.pt")
            save_tensor_if_present(saved, outdir, dbg, "layer_output", f"{prefix}_output.pt")
            aux_keys = sorted(dbg.keys())

        report = {
            "backend": "hf_absorbed",
            "layer_id": target_layer,
            "is_sparse": bool(is_sparse_layer(cfg, target_layer)),
            "prompt": prompt,
            "input_ids": ids,
            "decoded_input": decoded,
            "saved": saved,
            "debug_keys": sorted(list(dbg.keys())),
            "router_debug_keys": sorted(list(router_dbg.keys())),
            "aux_keys": aux_keys,
        }
        with (outdir / "hf_single_layer_manual.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(f"[hf-single-layer] wrote {outdir / 'hf_single_layer_manual.json'}")
    finally:
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
