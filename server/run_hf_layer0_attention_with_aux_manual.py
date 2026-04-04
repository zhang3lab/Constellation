from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_index(model_dir: str) -> dict:
    with open(f"{model_dir}/model.safetensors.index.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_raw_tensor(model_dir: str, tensor_name: str, index: dict | None = None) -> torch.Tensor:
    if index is None:
        index = load_index(model_dir)
    shard = index["weight_map"][tensor_name]
    with safe_open(f"{model_dir}/{shard}", framework="pt", device="cpu") as f:
        return f.get_tensor(tensor_name)


def get_attr_by_dotted_name(obj, dotted_name: str):
    cur = obj
    for part in dotted_name.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def copy_named_tensor_into_model(model, *, model_dir: str, tensor_name: str, index: dict | None = None) -> None:
    target = get_attr_by_dotted_name(model, tensor_name)
    raw = load_raw_tensor(model_dir, tensor_name, index=index)
    with torch.no_grad():
        target.copy_(raw.to(device=target.device, dtype=target.dtype))


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


def layer0_attention_names() -> list[str]:
    return [
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.q_a_layernorm.weight",
        "model.layers.0.self_attn.q_b_proj.weight",
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.0.self_attn.kv_a_layernorm.weight",
        "model.layers.0.self_attn.kv_b_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
    ]


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

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    print("[hf-attn0-aux] decoded =", repr(tokenizer.decode(input_ids)))

    model = load_hf_model(args.model_dir, args.device)
    try:
        index = load_index(args.model_dir)
        for name in layer0_attention_names():
            print(f"[hf-attn0-aux] copy {name}")
            copy_named_tensor_into_model(model, model_dir=args.model_dir, tensor_name=name, index=index)

        ids = torch.tensor([input_ids], dtype=torch.long, device=next(model.parameters()).device)

        captured = {}

        def hook(_module, _inp, out):
            x = out[0] if isinstance(out, tuple) else out
            captured["layer0_attn_output"] = x.detach().float().cpu()

        h = model.model.layers[0].self_attn.register_forward_hook(hook)
        with torch.no_grad():
            _ = model(input_ids=ids, use_cache=False, return_dict=True)
        h.remove()

        attn = model.model.layers[0].self_attn
        dbg = getattr(attn, "last_debug", {}) or {}

        saved = []

        p = outdir / "layer0_attn_output.pt"
        torch.save(captured["layer0_attn_output"], p)
        saved.append(str(p))

        q_nope_absorb = dbg.get("q_nope_absorb")
        q_pe_post_rope = dbg.get("q_pe_post_rope")
        if isinstance(q_nope_absorb, torch.Tensor) and isinstance(q_pe_post_rope, torch.Tensor):
            # q_nope_absorb: [B, T, H, K]
            # q_pe_post_rope: [B, H, T, R]
            q_rope_bthd = q_pe_post_rope.permute(0, 2, 1, 3).contiguous()
            q_flash = torch.cat(
                [q_nope_absorb.float().cpu(), q_rope_bthd.float().cpu()],
                dim=-1,
            )
            p = outdir / "q_flash.pt"
            torch.save(q_flash, p)
            saved.append(str(p))

        cache_latent = dbg.get("cache_latent")
        cache_k_rope = dbg.get("cache_k_rope")
        if isinstance(cache_latent, torch.Tensor) and isinstance(cache_k_rope, torch.Tensor):
            # both expected [B, T, D]
            blocked_k_token = torch.cat(
                [cache_latent.float().cpu(), cache_k_rope.float().cpu()],
                dim=-1,
            )
            p = outdir / "blocked_k_token.pt"
            torch.save(blocked_k_token, p)
            saved.append(str(p))

        x = dbg.get("rope_cos")
        if isinstance(x, torch.Tensor):
            p = outdir / "rope_cos.pt"
            torch.save(x.detach().float().cpu(), p)
            saved.append(str(p))

        x = dbg.get("rope_sin")
        if isinstance(x, torch.Tensor):
            p = outdir / "rope_sin.pt"
            torch.save(x.detach().float().cpu(), p)
            saved.append(str(p))

        x = dbg.get("cache_latent_raw")
        if isinstance(x, torch.Tensor):
            p = outdir / "cache_latent_raw.pt"
            torch.save(x.detach().float().cpu(), p)
            saved.append(str(p))

        x = dbg.get("cache_latent")
        if isinstance(x, torch.Tensor):
            p = outdir / "cache_latent.pt"
            torch.save(x.detach().float().cpu(), p)
            saved.append(str(p))

        x = dbg.get("cache_k_rope")
        if isinstance(x, torch.Tensor):
            p = outdir / "cache_k_rope.pt"
            torch.save(x.detach().float().cpu(), p)
            saved.append(str(p))

        x = dbg.get("q_nope_absorb")
        if isinstance(x, torch.Tensor):
            p = outdir / "q_nope_absorb.pt"
            torch.save(x.detach().float().cpu(), p)
            saved.append(str(p))

        x = dbg.get("last_value_heads")
        if isinstance(x, torch.Tensor):
            p = outdir / "last_value_heads.pt"
            torch.save(x.detach().float().cpu(), p)
            saved.append(str(p))

        x = dbg.get("scores_nope")
        if isinstance(x, torch.Tensor):
            p = outdir / "scores_nope.pt"
            torch.save(x.detach().float().cpu(), p)
            saved.append(str(p))

        report = {
            "backend": "hf_absorbed",
            "input_ids": input_ids,
            "decoded_input": tokenizer.decode(input_ids),
            "saved": saved,
            "debug_keys": sorted(list(dbg.keys())),
        }
        with (outdir / "hf_layer0_attention_aux.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(f"[hf-attn0-aux] wrote {outdir / 'hf_layer0_attention_aux.json'}")
    finally:
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
