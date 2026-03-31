from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.deepseek_full_model_ref import DeepseekFullModelRef
from server.full_model_runtime import run_full_model
from server.inference_session import InferenceSession
from server.moe_layer_runtime import run_moe_layer
from server.test_utils import make_safe_input
from server.validation_suite import run_validation_suite


def run_runtime_validation(coord, cfg):
    with InferenceSession(coord, cfg) as session:
        run_validation_suite(session)


def run_runtime_demo(coord, cfg):
    run_cfg = cfg["run"]
    layer_id = int(run_cfg["layer_id"])

    with InferenceSession(coord, cfg) as session:
        hidden_size = int(session.get_router_config()["hidden_size"])

        hidden = make_safe_input(hidden_size)
        result = run_moe_layer(session, hidden, layer_id)

        print("[demo] routes =", result["routes"])
        print("[demo] output[:8] =", result["output"][:8])


def run_full_model_debug(coord, cfg):
    import numpy as np
    import torch

    run_cfg = cfg["run"]

    start_layer = int(run_cfg["start_layer"])
    end_layer = int(run_cfg["end_layer"])
    collect_per_layer = bool(run_cfg["collect_per_layer"])
    prompt = "Hello world"

    with InferenceSession(coord, cfg) as session:
        session.full_model_ref = DeepseekFullModelRef(session)

        kv_cache_cfg = cfg["kv_cache"]
        session.ensure_full_model_runtime(
            tensor_cache_dir="tmp/non_moe_backbone_cache",
            split_layer=30,
            backbone_dtype=torch.bfloat16,
            kv_cache_cfg = kv_cache_cfg
        )
        session.reset_full_model_kv_cache(kv_cache_cfg = kv_cache_cfg)

        print("cuda:0 allocated GB =", torch.cuda.memory_allocated("cuda:0") / 1024**3)
        print("cuda:1 allocated GB =", torch.cuda.memory_allocated("cuda:1") / 1024**3)

        prepared = session.full_model_ref.prepare_prompt_hidden_input(prompt)
        hidden = np.asarray(prepared["hidden_in"], dtype=np.float32)

        hidden_size = int(session.get_router_config()["hidden_size"])
        if hidden.shape != (hidden_size,):
            raise RuntimeError(
                f"[full-model] hidden shape mismatch: "
                f"got={hidden.shape} expected={(hidden_size,)}"
            )

        print(f"[full-model] prompt={prompt!r}")
        print(f"[full-model] input_ids={prepared['input_ids']}")
        print(
            f"[full-model] start_layer={start_layer} "
            f"end_layer={end_layer} hidden_size={hidden_size}"
        )

        result = run_full_model(
            session,
            hidden,
            start_layer=start_layer,
            end_layer=end_layer,
            position_ids=prepared.get("position_ids"),
            attention_mask=prepared.get("attention_mask"),
            kv_cache=session.page_attention_cache_managers,
            collect_per_layer=collect_per_layer,
        )

        out = result["output"]
        print("[full-model] output[:8] =", out[:8])

        logits_result = session.full_model_ref.run_final_norm_and_lm_head(
            out,
            return_aux=collect_per_layer,
        )
        logits = np.asarray(logits_result.output, dtype=np.float32)

        topk = 10
        top_ids = np.argsort(logits)[-topk:][::-1]
        top_vals = logits[top_ids]

        print("[full-model] output[:8] =", out[:8])
        print("[full-model] logits_top_ids =", top_ids.tolist())
        print("[full-model] logits_top_vals =", top_vals.tolist())

        top_tokens = session.full_model_ref.decode_token_ids(top_ids.tolist())
        print("[full-model] logits_top_tokens =", top_tokens)

        if not np.isfinite(out).all():
            raise RuntimeError("[full-model] output contains non-finite values")

        if collect_per_layer:
            per_layer = result["per_layer"]
            print(f"[full-model] collected_layers={len(per_layer)}")


def main():
    cfg = load_config("server/config.json")

    coord = Coordinator(cfg["nodes"])
    setup_control_plane(coord, cfg)

    mode = str(cfg["run"]["mode"])
    if mode == "validation":
        run_runtime_validation(coord, cfg)
    elif mode == "demo":
        run_runtime_demo(coord, cfg)
    elif mode == "partial_61layer_debug":
        run_runtime_validation(coord, cfg)
    elif mode == "full_model_debug":
        run_full_model_debug(coord, cfg)
    else:
        raise RuntimeError(f"unknown mode={mode}")


if __name__ == "__main__":
    main()
