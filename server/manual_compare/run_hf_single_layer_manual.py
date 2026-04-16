from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

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


def build_layer_to_device_map(
    target_layer: int,
    devices: list[str],
) -> dict[int, str]:
    if target_layer < 0:
        raise ValueError(f"target_layer must be >= 0, got {target_layer}")
    if not devices:
        raise ValueError("devices must be non-empty")

    num_layers = target_layer + 1
    num_devices = len(devices)
    out: dict[int, str] = {}

    for layer_id in range(num_layers):
        dev_idx = (layer_id * num_devices) // num_layers
        out[layer_id] = str(devices[dev_idx])

    return out


def target_dtype_for_tensor_name(
    tensor_name: str,
    default_dtype: torch.dtype,
) -> torch.dtype:
    if tensor_name.endswith(".weight_scale_inv"):
        return torch.float32
    if tensor_name.endswith(".mlp.gate.e_score_correction_bias"):
        return torch.float32
    return default_dtype


def module_placements_to_target_layer(
    cfg,
    target_layer: int,
    devices: list[str],
) -> list[tuple[str, str]]:
    layer_to_device = build_layer_to_device_map(target_layer, devices)
    out: list[tuple[str, str]] = []

    out.append(("model.embed_tokens", devices[0]))

    for layer_id in range(target_layer + 1):
        dev = layer_to_device[layer_id]
        prefix = f"model.layers.{layer_id}"

        out.append((f"{prefix}.input_layernorm", dev))
        out.append((f"{prefix}.self_attn", dev))
        out.append((f"{prefix}.post_attention_layernorm", dev))

        if is_sparse_layer(cfg, layer_id):
            out.append((f"{prefix}.mlp.gate", dev))
            out.append((f"{prefix}.mlp.shared_experts", dev))
        else:
            out.append((f"{prefix}.mlp", dev))

    if target_layer == int(getattr(cfg, "num_hidden_layers")) - 1:
        out.append(("model.norm", devices[-1]))
        out.append(("lm_head", devices[-1]))

    return out


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

def target_dtype_for_tensor_name(
    tensor_name: str,
    default_dtype: torch.dtype,
) -> torch.dtype:
    if tensor_name.endswith(".weight_scale_inv"):
        return torch.float32
    if tensor_name.endswith(".mlp.gate.e_score_correction_bias"):
        return torch.float32
    return default_dtype


def copy_named_tensor_into_model(
    model,
    *,
    loader: DeepseekModelLoader,
    tensor_name: str,
    dtype: torch.dtype,
) -> None:
    target_dtype = target_dtype_for_tensor_name(tensor_name, dtype)

    parent_name, leaf_name = tensor_name.rsplit(".", 1)
    parent = get_attr_by_dotted_name(model, parent_name)
    target = get_attr_by_dotted_name(model, tensor_name)

    if isinstance(target, torch.nn.Parameter):
        target_device = str(target.device)
        requires_grad = target.requires_grad
    elif isinstance(target, torch.Tensor):
        target_device = str(target.device)
        requires_grad = False
    else:
        raise TypeError(f"{tensor_name}: unsupported target type {type(target).__name__}")

    loaded = loader.load_tensor_fp32_by_name(tensor_name).to(
        device=target_device,
        dtype=target_dtype,
    ).contiguous()

    print(
        f"[hf-single-layer] copy {tensor_name} -> "
        f"device={target_device} dtype={target_dtype}"
    )

    with torch.no_grad():
        if isinstance(target, torch.nn.Parameter):
            if getattr(target, "is_meta", False) or target.dtype != target_dtype:
                new_param = torch.nn.Parameter(
                    loaded,
                    requires_grad=requires_grad,
                )
                if not hasattr(parent, "_parameters") or leaf_name not in parent._parameters:
                    raise RuntimeError(
                        f"{tensor_name}: expected parameter leaf '{leaf_name}' under {parent_name}"
                    )
                parent._parameters[leaf_name] = new_param
            else:
                target.copy_(loaded)
        else:
            if getattr(target, "is_meta", False) or target.dtype != target_dtype:
                if hasattr(parent, "_buffers") and leaf_name in parent._buffers:
                    parent._buffers[leaf_name] = loaded
                else:
                    setattr(parent, leaf_name, loaded)
            else:
                target.copy_(loaded)


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


def materialize_modules_with_placements(
    model,
    placements: list[tuple[str, str]],
    *,
    default_dtype: torch.dtype,
) -> None:
    for name, device in placements:
        print(f"[hf-single-layer] to_empty {name} -> {device}")
        mod = get_attr_by_dotted_name(model, name)
        if not isinstance(mod, torch.nn.Module):
            continue

        mod.to_empty(device=device)

        # Recreate selected params/buffers with original fp32 dtype when needed.
        for param_name, param in list(mod.named_parameters(recurse=True)):
            full_name = f"{name}.{param_name}"
            want_dtype = target_dtype_for_tensor_name(full_name, default_dtype)
            if param.dtype == want_dtype:
                continue

            new_param = torch.nn.Parameter(
                torch.empty_like(param, device=device, dtype=want_dtype),
                requires_grad=param.requires_grad,
            )

            parent_rel, leaf = param_name.rsplit(".", 1) if "." in param_name else ("", param_name)
            parent_mod = mod.get_submodule(parent_rel) if parent_rel else mod
            parent_mod._parameters[leaf] = new_param

        for buffer_name, buf in list(mod.named_buffers(recurse=True)):
            full_name = f"{name}.{buffer_name}"
            want_dtype = target_dtype_for_tensor_name(full_name, default_dtype)
            if buf.dtype == want_dtype:
                continue

            new_buf = torch.empty_like(buf, device=device, dtype=want_dtype)

            parent_rel, leaf = buffer_name.rsplit(".", 1) if "." in buffer_name else ("", buffer_name)
            parent_mod = mod.get_submodule(parent_rel) if parent_rel else mod
            parent_mod._buffers[leaf] = new_buf


def move_tensor_to_module_device(x: torch.Tensor, mod: torch.nn.Module) -> torch.Tensor:
    try:
        dev = next(mod.parameters()).device
    except StopIteration:
        try:
            dev = next(mod.buffers()).device
        except StopIteration:
            return x

    if x.device != dev:
        x = x.to(dev)
    return x


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
    ap.add_argument(
        "--devices",
        type=str,
        default="cuda:0",
        help="Comma-separated CUDA devices, e.g. cuda:0,cuda:1,cuda:2,cuda:3",
    )
    ap.add_argument("--layer-id", type=int, required=True)
    args = ap.parse_args()

    devices = [x.strip() for x in args.devices.split(",") if x.strip()]
    if not devices:
        raise RuntimeError("no devices provided")

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

    layer_to_device = build_layer_to_device_map(target_layer, devices)
    placements = module_placements_to_target_layer(cfg, target_layer, devices)

    print("[hf-single-layer] devices =", devices)
    print("[hf-single-layer] layer_to_device =", layer_to_device)

    num_layers = int(getattr(cfg, "num_hidden_layers"))
    is_last_layer = (target_layer == num_layers - 1)

    backbone_dtype = torch.bfloat16

    try:
        for layer_id in range(target_layer + 1):
            if is_sparse_layer(cfg, layer_id):
                moe = model.model.layers[layer_id].mlp
                moe.layer_id = int(layer_id)
                moe._manual_loader = loader
                moe._manual_device = layer_to_device[layer_id]
                moe._manual_dtype = backbone_dtype
                moe._loaded_expert_ids = set()

                print(
                    f"[hf-bind] layer={layer_id}",
                    "layer_id=", moe.layer_id,
                    "loader_is_none=", moe._manual_loader is None,
                    "device=", moe._manual_device,
                    "dtype=", moe._manual_dtype,
                )

        materialize_modules_with_placements(
            model,
            placements,
            default_dtype=backbone_dtype,
        )

        needed_names = names_to_target_layer(cfg, resident_expert_ids, target_layer)

        for name in needed_names:
            print(f"[hf-single-layer] copy {name}")
            copy_named_tensor_into_model(
                model,
                loader=loader,
                tensor_name=name,
                dtype=backbone_dtype,
            )
        gate = model.model.layers[target_layer].mlp.gate
        print(
            "[hf-check] gate.weight dtype =", gate.weight.dtype,
            "gate.bias_corr dtype =", gate.e_score_correction_bias.dtype,
        )

        assert_named_tensors_materialized(model, needed_names)

        embed_weight = model.model.embed_tokens.weight
        ids_t = torch.tensor([ids], dtype=torch.long, device=embed_weight.device)
        hidden_states = model.model.embed_tokens(ids_t)

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

                layer_timing = []

                for i in range(target_layer + 1):
                    layer_mod = model.model.layers[i]

                    hidden_states = move_tensor_to_module_device(hidden_states, layer_mod)

                    #if attention_mask is not None:
                    #    attention_mask = attention_mask.to(hidden_states.device)
                    if position_ids is not None:
                        position_ids = position_ids.to(hidden_states.device)

                    dev = hidden_states.device
                    if dev.type == "cuda":
                        torch.cuda.synchronize(dev)
                    t0 = time.perf_counter()

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

                    dev = hidden_states.device
                    if dev.type == "cuda":
                        torch.cuda.synchronize(dev)
                    t1 = time.perf_counter()

                    layer_ms = (t1 - t0) * 1000.0
                    layer_timing.append(
                        {
                            "layer_id": int(i),
                            "device": str(dev),
                            "layer_ms": layer_ms,
                            "is_sparse": bool(is_sparse_layer(cfg, i)),
                        }
                    )

                    print(
                        f"[hf-layer-timing] layer={i} "
                        f"device={dev} "
                        f"is_sparse={is_sparse_layer(cfg, i)} "
                        f"ms={layer_ms:.3f}"
                    )

                class _ManualOutputs:
                    pass

                outputs = _ManualOutputs()
                outputs.hidden_states = tuple(
                    x.detach() for x in hidden_states_list
                )

                if is_last_layer:
                    hidden_states = hidden_states.to(model.model.norm.weight.device)
                    final_hidden_manual = model.model.norm(hidden_states)
                    final_hidden_manual = final_hidden_manual.to(model.lm_head.weight.device)
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
            tensor_keys = [
                "tokens_for_this_expert",
                "expert_out",
                "input",
                "gate_linear",
                "up_linear",
                "act",
                "mul",
                "down_proj",
            ]

            for j, item in enumerate(expert_dbg):
                if not isinstance(item, dict):
                    continue

                meta_item = {
                    "expert_local_id": int(item["expert_local_id"]) if "expert_local_id" in item else None,
                    "num_tokens": int(item["num_tokens"]) if "num_tokens" in item else None,
                }
                meta.append(meta_item)

                for key in tensor_keys:
                    x = item.get(key)
                    if isinstance(x, torch.Tensor):
                        p = outdir / f"{prefix}_hf_expert{j}_{key}.pt"
                        torch.save(x.detach().float().cpu(), p)
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
            "layer_timing": layer_timing,
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
