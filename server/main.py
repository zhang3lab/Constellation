from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.inference_session import InferenceSession
from server.moe_layer_runtime import run_moe_layer
from server.router_runtime import load_router_config
from server.test_utils import make_safe_input
from server.validation_suite import run_validation_suite


def run_runtime_validation(coord, cfg):
    with InferenceSession(coord, cfg) as session:
        run_validation_suite(session)


def run_runtime_demo(coord, cfg):
    run_cfg = cfg["run"]
    layer_id = int(run_cfg["layer_id"])
    model_root = str(cfg["model"]["root"])

    router_cfg = load_router_config(model_root)
    hidden_size = int(router_cfg["hidden_size"])

    with InferenceSession(coord, cfg) as session:
        hidden = make_safe_input(hidden_size)
        result = run_moe_layer(session, hidden, layer_id)

        print("[demo] routes =", result["routes"])
        print("[demo] combined[:8] =", result["combined"][:8])


def main():
    cfg = load_config("server/config.json")

    coord = Coordinator(cfg["nodes"])
    setup_control_plane(coord, cfg)

    mode = cfg["run"].get("mode", "validation")
    if mode == "validation":
        run_runtime_validation(coord, cfg)
    elif mode == "demo":
        run_runtime_demo(coord, cfg)
    else:
        raise RuntimeError(f"unknown mode={mode}")


if __name__ == "__main__":
    main()
