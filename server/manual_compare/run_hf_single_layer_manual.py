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


def assert_named_tensors_materialized(
    model,
    tensor_names: list[str],
) -> None:
    bad = []
    for name in tensor_names:
        x = get_attr_by_dotted_name(model, name)
        if getattr(x, "is_meta", False):
            bad.append(name)

    if bad:
        preview = bad[:20]
        raise RuntimeError(
            "Some required tensors are still on meta after copy:\n"
            + "\n".join(preview)
            + (f"\n... and {len(bad) - len(preview)} more" if len(bad) > len(preview) else "")
        )

def copy_named_tensor_into_model(
    model,
    *,
    loader: DeepseekModelLoader,
    tensor_name: str,
    device: str,
    dtype: torch.dtype,
) -> None:
    loaded = loader.load_tensor_fp32_by_name(tensor_name).to(device=device, dtype=dtype).contiguous()

    parent_name, leaf_name = tensor_name.rsplit(".", 1)
    parent = get_attr_by_dotted_name(model, parent_name)
    target = get_attr_by_dotted_name(model, tensor_name)

    with torch.no_grad():
        if isinstance(target, torch.nn.Parameter):
            if getattr(target, "is_meta", False):
                new_param = torch.nn.Parameter(
                    loaded,
                    requires_grad=target.requires_grad,
                )
                if not hasattr(parent, "_parameters") or leaf_name not in parent._parameters:
                    raise RuntimeError(
                        f"{tensor_name}: expected parameter leaf '{leaf_name}' under {parent_name}"
                    )
                parent._parameters[leaf_name] = new_param
            else:
                target.copy_(loaded.to(device=target.device, dtype=target.dtype))
        else:
            if getattr(target, "is_meta", False):
                if hasattr(parent, "_buffers") and leaf_name in parent._buffers:
                    parent._buffers[leaf_name] = loaded
                else:
                    setattr(parent, leaf_name, loaded)
            else:
                target.copy_(loaded.to(device=target.device, dtype=target.dtype))


def load_hf_model_skeleton(model_dir: str):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    config._attn_implementation = "eager"

    try:
        from accelerate import init_empty_weights
    except ImportError:
        init_empty_weights = None

    if init_empty_weights is not None:
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=True,
            )
    else:
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
        )

    model.eval()
    return model


def load_resident_expert_ids_from_model_dir(model_dir: str) -> list[int] | None:
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"missing config.json under {model_dir}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    resident = cfg.get("resident_expert_ids", None)
    if resident is None:
        return None

    if not isinstance(resident, list):
        raise RuntimeError(
            f"{config_path}: expected resident_expert_ids to be a list or absent, got {type(resident).__name__}"
        )

    out = sorted({int(x) for x in resident})
    if not out:
        raise RuntimeError(f"{config_path}: resident_expert_ids must not be empty when provided")

    return out


def is_sparse_layer(cfg, layer_id: int) -> bool:
    first_dense_replace = int(getattr(cfg, "first_k_dense_replace"))
    moe_layer_freq = int(getattr(cfg, "moe_layer_freq"))
    n_routed_experts = getattr(cfg, "n_routed_experts", None)
    return (
        n_routed_experts is not None
        and layer_id >= first_dense_replace
        and (layer_id % moe_layer_freq == 0)
    )


def is_routed_expert_tensor_name(name: str) -> bool:
    return ".mlp.experts." in name


def is_shared_expert_tensor_name(name: str) -> bool:
    return ".mlp.shared_experts." in name


def names_to_target_layer(cfg, resident_expert_ids: list[int] | None, target_layer: int) -> list[str]:
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
        else:
            names.extend(
                [
                    f"{prefix}.mlp.gate_proj.weight",
                    f"{prefix}.mlp.up_proj.weight",
                    f"{prefix}.mlp.down_proj.weight",
                ]
            )

    num_layers = int(getattr(cfg, "num_hidden_layers"))
    if int(target_layer) == num_layers - 1:
        names.extend(
            [
                "model.norm.weight",
                "lm_head.weight",
            ]
        )

    return names


def module_names_to_target_layer(cfg, target_layer: int) -> list[str]:
    names = ["model.embed_tokens"]

    for layer_id in range(target_layer + 1):
        prefix = f"model.layers.{layer_id}"
        names.extend(
            [
                f"{prefix}.input_layernorm",
                f"{prefix}.self_attn",
                f"{prefix}.post_attention_layernorm",
            ]
        )

        if is_sparse_layer(cfg, layer_id):
            names.extend(
                [
                    f"{prefix}.mlp.gate",
                    f"{prefix}.mlp.shared_experts",
                ]
            )
        else:
            names.append(f"{prefix}.mlp")

    num_layers = int(getattr(cfg, "num_hidden_layers"))
    if target_layer == num_layers - 1:
        names.extend(
            [
                "model.norm",
                "lm_head",
            ]
        )

    out = []
    seen = set()
    for x in names:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def materialize_modules_to_device(
    model,
    module_names: list[str],
    *,
    device: str,
) -> None:
    for name in module_names:
        print(f"[hf-single-layer] to_empty {name} -> {device}")
        mod = get_attr_by_dotted_name(model, name)
        if isinstance(mod, torch.nn.Module):
            mod.to_empty(device=device)


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

    if args.device == "cuda":
        backbone_device = "cuda:0"
    else:
        backbone_device = str(args.device)

    if str(backbone_device).startswith("cuda"):
        torch.cuda.set_device(backbone_device)
        _ = torch.empty(1, device=backbone_device)
        torch.cuda.empty_cache()

    with open(args.input_json, "r", encoding="utf-8") as f:
        inp = json.load(f)

    prompt = inp.get("prompt")
    input_ids = inp.get("input_ids")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    if input_ids is not None:
        if not isinstance(input_ids, list):
            raise TypeError(f"input_ids must be a list, got {type(input_ids).__name__}")
        ids = [int(x) for x in input_ids]
        if len(ids) == 0:
            raise RuntimeError("input_ids must not be empty")

        if prompt is None:
            prompt = tok.decode(ids)
    else:
        if prompt is None:
            raise RuntimeError("input_json must contain either prompt or input_ids")

        ids = tok(prompt, add_special_tokens=True)["input_ids"][0].tolist()

    decoded = tok.decode(ids)

    target_layer = int(args.layer_id)
    resident_expert_ids = load_resident_expert_ids_from_model_dir(args.model_dir)
    print(f"[hf-single-layer] target_layer={target_layer}")
    print(f"[hf-single-layer] resident_expert_ids={resident_expert_ids}")

    loader = DeepseekModelLoader(args.model_dir)
    model = load_hf_model_skeleton(args.model_dir)
    cfg = model.config
    num_layers = int(getattr(cfg, "num_hidden_layers"))
    is_last_layer = (target_layer == num_layers - 1)

    backbone_dtype = torch.bfloat16

    try:
        for layer_id in range(target_layer + 1):
            if is_sparse_layer(cfg, layer_id):
                moe = model.model.layers[layer_id].mlp
                moe.layer_id = int(layer_id)
                moe._manual_loader = loader
                moe._manual_device = backbone_device
                moe._manual_dtype = backbone_dtype
                moe._loaded_expert_ids = set()

                print(
                    f"[hf-bind] layer={layer_id}",
                    "layer_id=", moe.layer_id,
                    "loader_is_none=", moe._manual_loader is None,
                    "device=", moe._manual_device,
                    "dtype=", moe._manual_dtype,
                )

        needed_module_names = module_names_to_target_layer(cfg, target_layer)
        materialize_modules_to_device(
            model,
            needed_module_names,
            device=backbone_device,
        )

        if target_layer == int(getattr(cfg, "num_hidden_layers")) - 1:
            model.lm_head.to_empty(device=backbone_device)

        needed_names = names_to_target_layer(cfg, resident_expert_ids, target_layer)

        for name in needed_names:
            print(f"[hf-single-layer] copy {name}")
            copy_named_tensor_into_model(
                model,
                loader=loader,
                tensor_name=name,
                device=backbone_device,
                dtype=backbone_dtype,
            )

        assert_named_tensors_materialized(model, needed_names)

        ids_t = torch.tensor([ids], dtype=torch.long, device=backbone_device)

        layer_outputs: dict[str, torch.Tensor] = {}
        misc_outputs: dict[str, torch.Tensor] = {}

        def make_layer_hook(name: str):
            def _hook(_module, _inp, out):
                x = out[0] if isinstance(out, tuple) else out
                if not isinstance(x, torch.Tensor):
                    raise TypeError(f"hook {name}: expected tensor, got {type(x).__name__}")
                if x.ndim >= 1 and x.shape[0] == 1:
                    x = x.squeeze(0)
                layer_outputs[name] = x.detach().float().cpu()
            return _hook

        def make_misc_hook(name: str):
            def _hook(_module, _inp, out):
                x = out[0] if isinstance(out, tuple) else out
                if not isinstance(x, torch.Tensor):
                    raise TypeError(f"hook {name}: expected tensor, got {type(x).__name__}")
                if x.ndim >= 1 and x.shape[0] == 1:
                    x = x.squeeze(0)
                misc_outputs[name] = x.detach().float().cpu()
            return _hook

        hooks = []
        outputs = None
        try:
            for i in range(target_layer):
                hooks.append(
                    model.model.layers[i].register_forward_hook(
                        make_layer_hook(f"layer_{i}_output")
                    )
                )

            if is_last_layer:
                hooks.append(
                    model.model.norm.register_forward_hook(
                        make_misc_hook("final_hidden")
                    )
                )

            with torch.no_grad():
                hidden_states = model.model.embed_tokens(ids_t)

                position_ids = torch.arange(
                    hidden_states.shape[1],
                    device=hidden_states.device,
                    dtype=torch.long,
                ).unsqueeze(0)

                hidden_states_list = [hidden_states]

                for i in range(target_layer + 1):
                    layer_outputs_i = model.model.layers[i](
                        hidden_states,
                        attention_mask=None,
                        position_ids=position_ids,
                        past_key_value=None,
                        output_attentions=False,
                        use_cache=False,
                    )

                    if isinstance(layer_outputs_i, tuple):
                        hidden_states = layer_outputs_i[0]
                    else:
                        hidden_states = layer_outputs_i

                    hidden_states_list.append(hidden_states)

                class _ManualOutputs:
                    pass

                outputs = _ManualOutputs()
                outputs.hidden_states = tuple(
                    x.detach() for x in hidden_states_list
                )

                if is_last_layer:
                    final_hidden_manual = model.model.norm(hidden_states)
                    logits_manual = model.lm_head(final_hidden_manual)
                    outputs.logits = logits_manual
                else:
                    outputs.logits = None
        finally:
            for h in hooks:
                h.remove()

        dbg = getattr(model.model.layers[target_layer], "last_debug", {}) or {}
        router_dbg = {}
        if is_sparse_layer(cfg, target_layer):
            router_dbg = getattr(
                model.model.layers[target_layer].mlp.gate,
                "last_router_debug",
                {},
            ) or {}

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

        expert_dbg = dbg.get("debug_expert_outputs")
        if isinstance(expert_dbg, list):
            meta = []
            for j, item in enumerate(expert_dbg):
                if not isinstance(item, dict):
                    continue
                meta.append(
                    {
                        "expert_local_id": int(item["expert_local_id"]),
                        "num_tokens": int(item["num_tokens"]),
                    }
                )
                if "tokens_for_this_expert" in item:
                    p = outdir / f"{prefix}_hf_expert{j}_tokens.pt"
                    torch.save(item["tokens_for_this_expert"], p)
                    saved.append(str(p))
                if "expert_out" in item:
                    p = outdir / f"{prefix}_hf_expert{j}_output.pt"
                    torch.save(item["expert_out"], p)
                    saved.append(str(p))
         
            p = outdir / f"{prefix}_hf_expert_outputs_meta.json"
            with p.open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
                f.write("\n")
            saved.append(str(p))

        if is_last_layer:
            final_hidden = misc_outputs.get("final_hidden")
            if final_hidden is None:
                raise RuntimeError("missing final_hidden hook output from model.model.norm")

            p = outdir / "final_hidden.pt"
            torch.save(final_hidden, p)
            saved.append(str(p))

            if outputs is None:
                raise RuntimeError("model forward outputs is None")
            if outputs.hidden_states is None or len(outputs.hidden_states) == 0:
                raise RuntimeError("model forward did not return hidden_states")
            if not hasattr(outputs, "logits") or not isinstance(outputs.logits, torch.Tensor):
                raise RuntimeError("model forward did not return tensor logits")

            hidden_from_hook = final_hidden                                  # cpu, [T,H]
            hidden_from_outputs = outputs.hidden_states[-1].detach().float().cpu()
            if hidden_from_outputs.ndim >= 1 and hidden_from_outputs.shape[0] == 1:
                hidden_from_outputs = hidden_from_outputs.squeeze(0).contiguous()

            logits_from_outputs = outputs.logits.detach().float().cpu()
            if logits_from_outputs.ndim >= 1 and logits_from_outputs.shape[0] == 1:
                logits_from_outputs = logits_from_outputs.squeeze(0).contiguous()

            lm_head_w = model.lm_head.weight
            final_hidden_dev = hidden_from_hook.to(
                device=lm_head_w.device,
                dtype=lm_head_w.dtype,
            )
            with torch.no_grad():
                logits_manual = torch.matmul(final_hidden_dev, lm_head_w.t())
            logits_manual = logits_manual.detach().float().cpu()

            p = outdir / "logits.pt"
            torch.save(logits_manual, p)
            saved.append(str(p))

            p = outdir / "outputs0_hidden.pt"
            torch.save(hidden_from_outputs, p)
            saved.append(str(p))

            p = outdir / "outputs_logits.pt"
            torch.save(logits_from_outputs, p)
            saved.append(str(p))

            hidden_diff = (hidden_from_hook - hidden_from_outputs).abs()
            logits_diff = (logits_from_outputs - logits_manual).abs()

            debug_compare = {
                "hidden_hook_vs_outputs0": {
                    "shape_hook": list(hidden_from_hook.shape),
                    "shape_outputs0": list(hidden_from_outputs.shape),
                    "max_abs": float(hidden_diff.max().item()),
                    "mean_abs": float(hidden_diff.mean().item()),
                },
                "logits_outputs_vs_manual": {
                    "shape_outputs_logits": list(logits_from_outputs.shape),
                    "shape_manual_logits": list(logits_manual.shape),
                    "max_abs": float(logits_diff.max().item()),
                    "mean_abs": float(logits_diff.mean().item()),
                },
            }

            p = outdir / "final_head_debug_compare.json"
            with p.open("w", encoding="utf-8") as f:
                json.dump(debug_compare, f, ensure_ascii=False, indent=2)
                f.write("\n")
            saved.append(str(p))

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
            "has_final_hidden": bool(is_last_layer),
            "has_logits": bool(is_last_layer),
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
