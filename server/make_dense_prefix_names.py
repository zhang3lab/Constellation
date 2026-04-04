from __future__ import annotations

import json


def dense_layer_names(layer_id: int) -> list[str]:
    prefix = f"model.layers.{layer_id}"
    return [
        f"{prefix}.input_layernorm.weight",
        f"{prefix}.self_attn.q_a_proj.weight",
        f"{prefix}.self_attn.q_a_layernorm.weight",
        f"{prefix}.self_attn.q_b_proj.weight",
        f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
        f"{prefix}.self_attn.kv_a_layernorm.weight",
        f"{prefix}.self_attn.kv_b_proj.weight",
        f"{prefix}.self_attn.o_proj.weight",
        f"{prefix}.post_attention_layernorm.weight",
        f"{prefix}.mlp.gate_proj.weight",
        f"{prefix}.mlp.up_proj.weight",
        f"{prefix}.mlp.down_proj.weight",
    ]


def main() -> None:
    names = ["model.embed_tokens.weight"]
    for layer_id in [0, 1, 2]:
        names.extend(dense_layer_names(layer_id))

    with open("tmp/dense_prefix_names.json", "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("[names] wrote tmp/dense_prefix_names.json")
    print("[names] count =", len(names))


if __name__ == "__main__":
    main()
