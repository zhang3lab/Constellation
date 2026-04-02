import json
import tempfile
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch


def collect_non_moe_backbone_tensor_names_deepseek() -> list[str]:
    names: list[str] = []

    names.append("model.embed_tokens.weight")
    names.append("model.norm.weight")
    names.append("lm_head.weight")

    for layer_id in range(61):
        names.append(f"model.layers.{layer_id}.input_layernorm.weight")
        names.append(f"model.layers.{layer_id}.post_attention_layernorm.weight")

        names.append(f"model.layers.{layer_id}.self_attn.q_a_proj.weight")
        names.append(f"model.layers.{layer_id}.self_attn.q_a_layernorm.weight")
        names.append(f"model.layers.{layer_id}.self_attn.q_b_proj.weight")
        names.append(f"model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.weight")
        names.append(f"model.layers.{layer_id}.self_attn.kv_a_layernorm.weight")
        names.append(f"model.layers.{layer_id}.self_attn.kv_b_proj.weight")
        names.append(f"model.layers.{layer_id}.self_attn.o_proj.weight")

    for layer_id in range(3):
        names.append(f"model.layers.{layer_id}.mlp.up_proj.weight")
        names.append(f"model.layers.{layer_id}.mlp.gate_proj.weight")
        names.append(f"model.layers.{layer_id}.mlp.down_proj.weight")

    for layer_id in range(3, 61):
        names.append(f"model.layers.{layer_id}.mlp.shared_experts.up_proj.weight")
        names.append(f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight")
        names.append(f"model.layers.{layer_id}.mlp.shared_experts.down_proj.weight")

        names.append(f"model.layers.{layer_id}.mlp.gate.weight")
        names.append(f"model.layers.{layer_id}.mlp.gate.e_score_correction_bias")

    return list(dict.fromkeys(names))


def _torch_dtype_to_index_string(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    raise ValueError(f"unsupported dtype for tensor cache: {dtype}")


def _index_string_to_numpy_dtype(dtype: str):
    if dtype == "float32":
        return np.float32
    raise ValueError(f"unsupported dtype in tensor cache index: {dtype}")


class TensorCacheBuilder:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.index_path = self.cache_dir / "index.json"
        self.weights_path = self.cache_dir / "weights.bin"

    def build_from_names(
        self,
        model_loader,
        tensor_names: list[str],
        *,
        overwrite: bool = False,
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not overwrite and (self.index_path.exists() or self.weights_path.exists()):
            raise RuntimeError(
                f"tensor cache already exists at {self.cache_dir}; "
                f"use overwrite=True to rebuild"
            )

        tensor_names = list(dict.fromkeys(str(x) for x in tensor_names))

        tmp_dir = Path(tempfile.mkdtemp(prefix="tensor_cache_build_", dir=self.cache_dir))
        tmp_weights_path = tmp_dir / "weights.bin"
        tmp_index_path = tmp_dir / "index.json"

        try:
            index: Dict[str, Dict[str, Any]] = {}
            offset = 0

            with tmp_weights_path.open("wb") as f:
                for i, name in enumerate(tensor_names):
                    print(f"[tensor-cache] [{i+1}/{len(tensor_names)}] {name}")
                    t = model_loader.load_tensor_fp32_by_name(name)
                    if t.dtype != torch.float32:
                        raise RuntimeError(
                            f"expected float32 tensor after loader decode: {name} dtype={t.dtype}"
                        )

                    t = t.detach().contiguous().cpu()
                    arr = t.numpy()
                    if arr.dtype != np.float32:
                        raise RuntimeError(
                            f"expected numpy float32 tensor cache dtype: {name} dtype={arr.dtype}"
                        )

                    raw = arr.tobytes(order="C")
                    nbytes = len(raw)

                    f.write(raw)

                    index[name] = {
                        "offset": int(offset),
                        "nbytes": int(nbytes),
                        "shape": [int(x) for x in arr.shape],
                        "dtype": _torch_dtype_to_index_string(t.dtype),
                    }
                    offset += nbytes

            meta = {
                "format": "constellation_tensor_cache_v1",
                "model_root": str(model_loader.model_root),
                "weights_file": "weights.bin",
                "num_tensors": len(index),
                "tensors": index,
            }
            tmp_index_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            if overwrite:
                if self.index_path.exists():
                    self.index_path.unlink()
                if self.weights_path.exists():
                    self.weights_path.unlink()

            tmp_weights_path.replace(self.weights_path)
            tmp_index_path.replace(self.index_path)

        finally:
            if tmp_dir.exists():
                for p in tmp_dir.iterdir():
                    if p.exists():
                        p.unlink()
                tmp_dir.rmdir()


class MappedTensorStore:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.index_path = self.cache_dir / "index.json"

        if not self.index_path.exists():
            raise RuntimeError(f"tensor cache index not found: {self.index_path}")

        meta = json.loads(self.index_path.read_text(encoding="utf-8"))
        if meta.get("format") != "constellation_tensor_cache_v1":
            raise RuntimeError(
                f"unsupported tensor cache format: {meta.get('format')}"
            )

        weights_file = meta.get("weights_file", "weights.bin")
        self.weights_path = self.cache_dir / weights_file
        if not self.weights_path.exists():
            raise RuntimeError(f"tensor cache weights file not found: {self.weights_path}")

        tensors = meta.get("tensors")
        if not isinstance(tensors, dict):
            raise RuntimeError("tensor cache index missing tensors dict")

        self.meta = meta
        self._index: Dict[str, Dict[str, Any]] = tensors
        self._mmap = np.memmap(self.weights_path, mode="r", dtype=np.uint8)

    def close(self):
        if self._mmap is not None:
            mm = getattr(self._mmap, "_mmap", None)
            if mm is not None:
                mm.close()
            self._mmap = None

    def has_tensor(self, name: str) -> bool:
        return str(name) in self._index

    def tensor_names(self) -> list[str]:
        return sorted(self._index.keys())

    def get_numpy(self, name: str) -> np.ndarray:
        if self._mmap is None:
            raise RuntimeError("mapped tensor store is closed")

        name = str(name)
        rec = self._index.get(name)
        if rec is None:
            raise KeyError(f"tensor not found in mapped store: {name}")

        offset = int(rec["offset"])
        nbytes = int(rec["nbytes"])
        shape = tuple(int(x) for x in rec["shape"])
        dtype = _index_string_to_numpy_dtype(str(rec["dtype"]))

        itemsize = np.dtype(dtype).itemsize
        expected_nbytes = int(np.prod(shape, dtype=np.int64)) * itemsize
        if expected_nbytes != nbytes:
            raise RuntimeError(
                f"tensor cache index nbytes mismatch for {name}: "
                f"index={nbytes} expected={expected_nbytes}"
            )

        arr = np.ndarray(
            shape=shape,
            dtype=dtype,
            buffer=self._mmap,
            offset=offset,
            order="C",
        )
        return arr

    def get_torch_cpu(self, name: str) -> torch.Tensor:
        arr = self.get_numpy(name)
        # read-only zero-copy CPU view backed by memmap; callers must not mutate
        return torch.from_numpy(arr)

    def get_torch(
        self,
        name: str,
        *,
        device: str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        t = self.get_torch_cpu(name)
        if dtype is None:
            dtype = t.dtype
        return t.to(device=device, dtype=dtype)
