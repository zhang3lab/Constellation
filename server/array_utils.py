from dataclasses import dataclass
from typing import Literal
import numpy as np
import torch


ArrayBackend = Literal["numpy", "torch"]
ArrayDType = Literal["preserve", "float32", "float16", "bfloat16"]


@dataclass(frozen=True)
class ArrayConfig:
    backend: ArrayBackend
    dtype: ArrayDType = "preserve"
    device: str | None = None          # assert only; never move
    allow_1d: bool = True
    allow_2d: bool = True
    require_finite: bool = True
    contiguous: bool = True
    allow_cast: bool = False           # allow dtype cast within same backend only


ARRCFG_HIDDEN_NUMPY_F32 = ArrayConfig(
    backend="numpy",
    dtype="float32",
    device=None,
    allow_1d=True,
    allow_2d=True,
    require_finite=True,
    contiguous=True,
    allow_cast=False,
)

ARRCFG_VECTOR_NUMPY_F32 = ArrayConfig(
    backend="numpy",
    dtype="float32",
    device=None,
    allow_1d=True,
    allow_2d=False,
    require_finite=True,
    contiguous=True,
    allow_cast=False,
)


def ARRCFG_HIDDEN_TORCH(dtype: str, device: str) -> ArrayConfig:
    if dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError(f"unsupported torch dtype: {dtype}")
    return ArrayConfig(
        backend="torch",
        dtype=dtype,
        device=device,
        allow_1d=True,
        allow_2d=True,
        require_finite=True,
        contiguous=True,
        allow_cast=False,
    )


def ARRCFG_VECTOR_TORCH(dtype: str, device: str) -> ArrayConfig:
    if dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError(f"unsupported torch dtype: {dtype}")
    return ArrayConfig(
        backend="torch",
        dtype=dtype,
        device=device,
        allow_1d=True,
        allow_2d=False,
        require_finite=True,
        contiguous=True,
        allow_cast=False,
    )


def ARRCFG_PARAM_TORCH(dtype: str, device: str) -> ArrayConfig:
    """
    Torch parameter/config tensor:
    - allow 1D for norm weights
    - allow 2D for linear weights
    - no implicit cast/move
    """
    if dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError(f"unsupported torch dtype: {dtype}")
    return ArrayConfig(
        backend="torch",
        dtype=dtype,
        device=device,
        allow_1d=True,
        allow_2d=True,
        require_finite=True,
        contiguous=True,
        allow_cast=False,
    )


def torch_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    raise ValueError(f"unsupported torch dtype: {dtype}")


def _resolve_numpy_dtype(dtype: ArrayDType):
    if dtype == "preserve":
        return None
    if dtype == "float32":
        return np.float32
    if dtype == "float16":
        return np.float16
    if dtype == "bfloat16":
        raise ValueError("numpy backend does not support bfloat16 reliably")
    raise ValueError(f"unsupported dtype: {dtype}")


def _resolve_torch_dtype(dtype: ArrayDType):
    if dtype == "preserve":
        return None
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {dtype}")


def _shape_tuple(x) -> tuple[int, ...]:
    return tuple(int(v) for v in x.shape)


def _check_hidden_shape(x, name: str, allow_1d: bool, allow_2d: bool) -> None:
    allowed_ndims = set()
    if allow_1d:
        allowed_ndims.add(1)
    if allow_2d:
        allowed_ndims.add(2)

    if x.ndim not in allowed_ndims:
        raise ValueError(
            f"{name} must have ndim in {sorted(allowed_ndims)}, got shape {_shape_tuple(x)}"
        )

    if x.ndim == 2 and x.shape[0] == 0:
        raise ValueError(f"{name} must not have empty seq dimension, got shape {_shape_tuple(x)}")

    if x.shape[-1] == 0:
        raise ValueError(f"{name} must not have empty hidden dimension, got shape {_shape_tuple(x)}")


def as_array(x, name: str, config: ArrayConfig):
    if config.backend == "numpy":
        if isinstance(x, torch.Tensor):
            raise TypeError(
                f"{name} expected numpy backend, got torch.Tensor on device={x.device}"
            )

        np_dtype = _resolve_numpy_dtype(config.dtype)

        arr = np.asarray(x)
        _check_hidden_shape(arr, name, config.allow_1d, config.allow_2d)

        if np_dtype is not None and arr.dtype != np_dtype:
            if config.allow_cast:
                arr = arr.astype(np_dtype, copy=False)
            else:
                raise TypeError(
                    f"{name} expected dtype={np_dtype}, got dtype={arr.dtype}"
                )

        if config.require_finite and not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")

        if config.contiguous and not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)

        return arr

    if config.backend == "torch":
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"{name} expected torch backend, got type={type(x).__name__}"
            )

        t = x
        _check_hidden_shape(t, name, config.allow_1d, config.allow_2d)

        expected_dtype = _resolve_torch_dtype(config.dtype)
        if expected_dtype is not None and t.dtype != expected_dtype:
            if config.allow_cast:
                t = t.to(dtype=expected_dtype)
            else:
                raise TypeError(
                    f"{name} expected dtype={expected_dtype}, got dtype={t.dtype}"
                )

        if config.device is not None and str(t.device) != str(config.device):
            raise RuntimeError(
                f"{name} expected device={config.device}, got device={t.device}"
            )

        if config.require_finite and not torch.all(torch.isfinite(t)).item():
            raise ValueError(f"{name} contains non-finite values")

        if config.contiguous and not t.is_contiguous():
            t = t.contiguous()

        return t

    raise ValueError(f"unsupported backend: {config.backend}")
