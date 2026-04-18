from __future__ import annotations

import torch

from server.config import load_config
from server.control_plane import setup_control_plane
from server.coordinator import Coordinator
from server.deepseek_full_model_executor import DeepseekFullModelExecutor
from server.inference_session import InferenceSession


class OpenAIAPIRuntime:
    def __init__(self, config_path: str):
        self.config_path = str(config_path)
        self.cfg = None
        self.coord = None
        self.session = None

    def start(self) -> None:
        if self.session is not None:
            return

        cfg = load_config(self.config_path)
        coord = Coordinator(cfg["nodes"], log_level=cfg["log_level"])
        setup_control_plane(coord, cfg)

        session = InferenceSession(coord, cfg)
        session.__enter__()

        try:
            session.full_model_executor = DeepseekFullModelExecutor(session)
            session.initialize_full_model_runtime(
                tensor_cache_dir="tmp/non_moe_backbone_cache",
                split_layer=30,
                backbone_dtype=torch.bfloat16,
                kv_cache_cfg=cfg["kv_cache"],
            )

            if not session.is_chat_runtime_ready():
                raise RuntimeError("chat runtime is not ready")
        except Exception:
            session.__exit__(None, None, None)
            raise

        self.cfg = cfg
        self.coord = coord
        self.session = session

    def close(self) -> None:
        session = self.session
        self.session = None
        self.coord = None
        self.cfg = None

        if session is not None:
            session.__exit__(None, None, None)

    def get_session(self):
        if self.session is None:
            raise RuntimeError("runtime session is not initialized")
        return self.session
