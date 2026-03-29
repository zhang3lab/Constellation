from __future__ import annotations

import threading
from typing import Dict, Tuple

from server.client import NodeClient
from server.deepseek_model_loader import DeepseekModelLoader


class SessionClientPool:
    def __init__(self):
        self._lock = threading.Lock()
        self._clients: Dict[Tuple[str, int], NodeClient] = {}

    def get(self, host: str, port: int) -> NodeClient:
        key = (host, int(port))
        with self._lock:
            client = self._clients.get(key)
            if client is not None:
                return client

            client = NodeClient(host, int(port))
            client.__enter__()
            self._clients[key] = client
            return client

    def invalidate(self, host: str, port: int) -> None:
        key = (host, int(port))
        with self._lock:
            client = self._clients.pop(key, None)

        if client is not None:
            try:
                client.__exit__(None, None, None)
            except Exception:
                pass

    def close_all(self) -> None:
        with self._lock:
            items = list(self._clients.items())
            self._clients.clear()

        for _, client in items:
            try:
                client.__exit__(None, None, None)
            except Exception:
                pass


class InferenceSession:
    def __init__(self, coord, cfg):
        self.coord = coord
        self.cfg = cfg
        self.client_pool = SessionClientPool()

        self.deepseek_model_loader = None
        self.full_model_ref = None

    def get_deepseek_model_loader(self) -> DeepseekModelLoader:
        if self.deepseek_model_loader is None:
            model_root = str(self.cfg["model"]["root"])
            self.deepseek_model_loader = DeepseekModelLoader(model_root)
        return self.deepseek_model_loader

    def get_router_config(self) -> dict:
        return self.get_deepseek_model_loader().router_config()

    def get_mla_config(self) -> dict:
        return self.get_deepseek_model_loader().mla_config()

    def close(self) -> None:
        self.client_pool.close_all()
        self.deepseek_model_loader = None
        self.full_model_ref = None

    def __enter__(self) -> "InferenceSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
