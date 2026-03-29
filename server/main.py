from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.deepseek_full_model_ref import PlaceholderDeepseekFullModelRef
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
    run_cfg = cfg["run"]

    start_layer = int(run_cfg.get("start_layer", 0))
    end_layer = int(run_cfg.get("end_layer", 3))
    collect_per_layer = bool(run_cfg.get("collect_per_layer", True))
    prompt = "Hello world"

    with InferenceSession(coord, cfg) as session:
        session.full_model_ref = PlaceholderDeepseekFullModelRef(session)

        hidden_size = int(session.get_router_config()["hidden_size"])

        print(f"[full-model] prompt={prompt!r}")
        print(
            f"[full-model] start_layer={start_layer} "
            f"end_layer={end_layer} hidden_size={hidden_size}"
        )

        # Temporary placeholder until prompt->embedding path is wired in.
        hidden = make_safe_input(hidden_size)

        result = run_full_model(
            session,
            hidden,
            start_layer=start_layer,
            end_layer=end_layer,
            position_ids=None,
            attention_mask=None,
            kv_cache=None,
            collect_per_layer=collect_per_layer,
        )

        print("[full-model] output[:8] =", result["output"][:8])
        if collect_per_layer:
            print(f"[full-model] collected_layers={len(result['per_layer'])}")


def main():
    cfg = load_config("server/config.json")

    coord = Coordinator(cfg["nodes"])
    setup_control_plane(coord, cfg)

    mode = str(cfg["run"].get("mode", "validation"))
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
