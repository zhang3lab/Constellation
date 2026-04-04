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

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    print("[hf-qparts] decoded =", repr(tok.decode(input_ids)))

    model = load_hf_model(args.model_dir, args.device)
    try:
        index = load_index(args.model_dir)
        for name in layer0_attention_names():
            print(f"[hf-qparts] copy {name}")
            copy_named_tensor_into_model(model, model_dir=args.model_dir, tensor_name=name, index=index)

        ids = torch.tensor([input_ids], dtype=torch.long, device=next(model.parameters()).device)
        with torch.no_grad():
            _ = model(input_ids=ids, use_cache=False, return_dict=True)

        dbg = getattr(model.model.layers[0].self_attn, "last_debug", {}) or {}

        saved = []

        q_nope_absorb = dbg.get("q_nope_absorb")
        if isinstance(q_nope_absorb, torch.Tensor):
            p = outdir / "q_nope_absorb.pt"
            torch.save(q_nope_absorb.detach().float().cpu(), p)
            saved.append(str(p))

        q_pe_post_rope = dbg.get("q_pe_post_rope")
        if isinstance(q_pe_post_rope, torch.Tensor):
            p = outdir / "q_pe_post_rope.pt"
            torch.save(q_pe_post_rope.detach().float().cpu(), p)
            saved.append(str(p))

        report = {
            "backend": "hf_absorbed",
            "input_ids": input_ids,
            "decoded_input": tok.decode(input_ids),
            "debug_keys": sorted(list(dbg.keys())),
            "saved": saved,
        }
        with (outdir / "hf_layer0_qparts.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(f"[hf-qparts] wrote {outdir / 'hf_layer0_qparts.json'}")
    finally:
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
