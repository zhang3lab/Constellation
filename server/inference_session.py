from __future__ import annotations

import threading
from typing import Dict, Tuple

from server.client import NodeClient


class SessionClientPool:
    def __init__(self):
        self._lock = threading.Lock()
        self._clients: Dict[Tuple[str, int], NodeClient] = {}
        self.reference_weight_cache = {}
        self.model_locator = None

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

        # router-related caches
        self.router_cfg = None
        self.router_tensors_by_layer = {}

    def close(self) -> None:
        self.client_pool.close_all()
        self.router_tensors_by_layer.clear()

    def __enter__(self) -> "InferenceSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
