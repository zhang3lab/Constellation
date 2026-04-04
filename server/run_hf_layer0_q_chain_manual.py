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
    input_ids = inp["input_ids"]

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    print("[hf-qchain] decoded =", repr(tok.decode(input_ids)))

    loader = DeepseekModelLoader(args.model_dir)
    model = load_hf_model(args.model_dir, args.device)
    try:
        for name in layer0_attention_names():
            print(f"[hf-qchain] copy {name}")
            copy_named_tensor_into_model(model, loader=loader, tensor_name=name)

        ids = torch.tensor([input_ids], dtype=torch.long, device=next(model.parameters()).device)
        with torch.no_grad():
            _ = model(input_ids=ids, use_cache=False, return_dict=True)

        dbg = getattr(model.model.layers[0].self_attn, "last_debug", {}) or {}

        saved: list[str] = []
        save_tensor_if_present(saved, outdir, dbg, "q_latent_pre_norm")
        save_tensor_if_present(saved, outdir, dbg, "q_latent_post_norm")
        save_tensor_if_present(saved, outdir, dbg, "q_pre_split")
        save_tensor_if_present(saved, outdir, dbg, "q_pe_pre_rope", "q_rope_pre_rotary.pt")
        save_tensor_if_present(saved, outdir, dbg, "q_pe_post_rope", "q_rope_post_rotary.pt")

        report = {
            "backend": "hf_absorbed",
            "input_ids": input_ids,
            "decoded_input": tok.decode(input_ids),
            "debug_keys": sorted(list(dbg.keys())),
            "saved": saved,
        }
        with (outdir / "hf_layer0_q_chain.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(f"[hf-qchain] wrote {outdir / 'hf_layer0_q_chain.json'}")
    finally:
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
