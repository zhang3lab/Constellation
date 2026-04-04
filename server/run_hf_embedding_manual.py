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
    print("[hf-embed] decoded =", repr(tokenizer.decode(input_ids)))

    model = load_hf_model(args.model_dir, args.device)
    try:
        index = load_index(args.model_dir)
        copy_named_tensor_into_model(
            model,
            model_dir=args.model_dir,
            tensor_name="model.embed_tokens.weight",
            index=index,
        )

        ids = torch.tensor([input_ids], dtype=torch.long, device=next(model.parameters()).device)

        with torch.no_grad():
            hidden = model.model.embed_tokens(ids).detach().float().cpu()

        torch.save(hidden, outdir / "embedding.pt")

        report = {
            "backend": "hf_absorbed",
            "input_ids": input_ids,
            "decoded_input": tokenizer.decode(input_ids),
            "saved": [str(outdir / "embedding.pt")],
        }
        with (outdir / "hf_embedding.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(f"[hf-embed] wrote {outdir / 'hf_embedding.json'}")
    finally:
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
