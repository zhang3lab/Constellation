from __future__ import annotations

import json
import socket
from typing import Any


class CacheClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 47000):
        self.host = str(host)
        self.port = int(port)
        self.sock: socket.socket | None = None
        self.rfile = None
        self.wfile = None

    def __enter__(self) -> "CacheClient":
        self.sock = socket.create_connection((self.host, self.port))
        self.rfile = self.sock.makefile("rb")
        self.wfile = self.sock.makefile("wb")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self.rfile is not None:
                self.rfile.close()
        finally:
            try:
                if self.wfile is not None:
                    self.wfile.close()
                finally:
                    if self.sock is not None:
                        self.sock.close()

    def _request(self, obj: dict[str, Any]) -> dict[str, Any]:
        assert self.wfile is not None and self.rfile is not None
        payload = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        self.wfile.write(payload)
        self.wfile.flush()

        line = self.rfile.readline()
        if not line:
            raise RuntimeError("cache daemon closed connection")
        return json.loads(line.decode("utf-8"))

    def borrow_expert(self, layer_id: int, expert_id: int, device_id: int) -> dict[str, Any]:
        return self._request(
            {
                "op": "borrow_expert",
                "layer_id": int(layer_id),
                "expert_id": int(expert_id),
                "device_id": int(device_id),
            }
        )

    def return_expert(self, lease_id: str) -> dict[str, Any]:
        return self._request(
            {
                "op": "return_expert",
                "lease_id": str(lease_id),
            }
        )

    def query(self) -> dict[str, Any]:
        return self._request({"op": "query"})
