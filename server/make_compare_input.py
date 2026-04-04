from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from server.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="server/test/config.json")
    ap.add_argument("--model-dir", type=str, default=None)
    ap.add_argument("--system", type=str, default="You are a helpful assistant.")
    ap.add_argument("--prompt", type=str, default="Hello world")
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_dir = args.model_dir or str(cfg["model"]["root"])

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]

    templated = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    input_ids_obj = templated["input_ids"]
    if input_ids_obj and isinstance(input_ids_obj[0], list):
        input_ids = [int(x) for x in input_ids_obj[0]]
    else:
        input_ids = [int(x) for x in input_ids_obj]

    out = {
        "model_name": cfg["model"]["name"],
        "messages": messages,
        "input_ids": input_ids,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"[make-input] wrote {output_path}")
    print(f"[make-input] num_input_ids = {len(input_ids)}")
    print(f"[make-input] input_ids = {input_ids}")


if __name__ == "__main__":
    main()
