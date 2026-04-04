from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class DebugTensorCollector:
    enabled: bool
    data: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, value: Any) -> None:
        if not self.enabled:
            return
        if value is None:
            return

        if isinstance(value, torch.Tensor):
            self.data[name] = value.detach().float().cpu().numpy()
            return

        if isinstance(value, np.ndarray):
            self.data[name] = value.astype(np.float32, copy=False)
            return

        if isinstance(value, (int, float, str, bool)):
            self.data[name] = value
            return

        raise TypeError(f"unsupported debug value for {name}: {type(value).__name__}")

    def add_tensor(self, name: str, value: torch.Tensor) -> None:
        if not self.enabled:
            return
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} expected torch.Tensor, got {type(value).__name__}")
        self.data[name] = value.detach().float().cpu().numpy()

    def add_scalar(self, name: str, value: int | float | str | bool) -> None:
        if not self.enabled:
            return
        self.data[name] = value

    def export(self) -> dict[str, Any]:
        return dict(self.data)

    def add_meta(self, **kwargs: Any) -> None:
        if not self.enabled:
            return
        for k, v in kwargs.items():
            self.add(k, v)
