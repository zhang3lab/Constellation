import numpy as np


def as_f32_1d(x, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise RuntimeError(f"{name} must be 1-D, got shape={x.shape}")
    if not np.all(np.isfinite(x)):
        raise RuntimeError(f"{name} contains non-finite values")
    return x
