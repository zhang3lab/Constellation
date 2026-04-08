import os
import re
import shutil
import subprocess
from pathlib import Path

from .common import include_flags, obj_path, resolve_src, run

_FALLBACK_SMS = ["89"]
_CACHED_NVCC = None
_CACHED_SMS = None
_CACHED_CUDA_INCLUDE_FLAGS = None
_CACHED_CUDA_LINK_FLAGS = None


def _capture(cmd):
    return subprocess.check_output(cmd, text=True)


def _resolve_nvcc(explicit_nvcc):
    candidates = []

    if explicit_nvcc:
        candidates.append(explicit_nvcc)

    env_nvcc = os.environ.get("NVCC")
    if env_nvcc:
        candidates.append(env_nvcc)

    which_nvcc = shutil.which("nvcc")
    if which_nvcc:
        candidates.append(which_nvcc)

    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    for c in uniq:
        p = shutil.which(c) if Path(c).name == c else c
        if p:
            return p

    raise RuntimeError(
        "CUDA backend enabled, but nvcc was not found. "
        "Set NVCC, or add nvcc to PATH."
    )


def _detect_sms():
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []

    queries = [
        [nvidia_smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
        [nvidia_smi, "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
    ]

    sms = []
    for q in queries:
        try:
            out = _capture(q).strip()
        except Exception:
            continue
        if not out:
            continue

        for line in out.splitlines():
            m = re.match(r"^\s*(\d+)\.(\d+)\s*$", line.strip())
            if m:
                sms.append(f"{m.group(1)}{m.group(2)}")

        if sms:
            break

    return sorted(set(sms), key=lambda x: int(x))


def _resolve_sms():
    sms = _detect_sms()
    if sms:
        print(f"detected sms={sms}")
        return sms

    print(f"warning: nvidia-smi probe failed, using fallback sms={_FALLBACK_SMS}")
    return _FALLBACK_SMS[:]


def _gencode_flags(sms):
    flags = []
    for sm in sms:
        flags += ["-gencode", f"arch=compute_{sm},code=sm_{sm}"]
    return flags


def _detect_cuda_include_dirs():
    dirs = []

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        inc = Path(cuda_home) / "include"
        if inc.exists():
            dirs.append(inc)

    nvcc = shutil.which("nvcc")
    if nvcc:
        nvcc_path = Path(nvcc).resolve()
        for cand in [
            nvcc_path.parent.parent / "include",
            nvcc_path.parent / "include",
        ]:
            if cand.exists() and cand not in dirs:
                dirs.append(cand)

    for cand in [
        Path("/usr/local/cuda/include"),
        Path("/usr/local/include"),
        Path("/usr/include"),
    ]:
        if cand.exists() and cand not in dirs:
            dirs.append(cand)

    return dirs


def get_cuda_include_flags():
    global _CACHED_CUDA_INCLUDE_FLAGS
    if _CACHED_CUDA_INCLUDE_FLAGS is None:
        flags = []
        for d in _detect_cuda_include_dirs():
            flags += ["-I", str(d)]
        _CACHED_CUDA_INCLUDE_FLAGS = flags
    return list(_CACHED_CUDA_INCLUDE_FLAGS)


def get_cuda_link_flags():
    global _CACHED_CUDA_LINK_FLAGS
    if _CACHED_CUDA_LINK_FLAGS is not None:
        return list(_CACHED_CUDA_LINK_FLAGS)

    flags = []

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        lib64 = Path(cuda_home) / "lib64"
        if lib64.exists():
            flags += ["-L", str(lib64)]

    nvcc = shutil.which("nvcc")
    if nvcc:
        nvcc_path = Path(nvcc).resolve()
        for cand in [
            nvcc_path.parent.parent / "lib64",
            nvcc_path.parent.parent / "targets/x86_64-linux/lib",
        ]:
            if cand.exists():
                flags += ["-L", str(cand)]

    flags += ["-lcudart", "-ldl"]
    _CACHED_CUDA_LINK_FLAGS = flags
    return list(_CACHED_CUDA_LINK_FLAGS)


def _ensure_cuda_toolchain(nvcc):
    global _CACHED_NVCC, _CACHED_SMS
    if _CACHED_NVCC is None:
        _CACHED_NVCC = _resolve_nvcc(nvcc)
    if _CACHED_SMS is None:
        _CACHED_SMS = _resolve_sms()
    return _CACHED_NVCC, _CACHED_SMS


def compile_source(
    nvcc: str,
    project_root: Path,
    repo_root: Path,
    build_dir: Path,
    src_rel: str,
    cxx_std: str,
    opt: str,
    defines,
    debug: bool,
):
    nvcc_path, sms = _ensure_cuda_toolchain(nvcc)

    src = resolve_src(project_root, src_rel)
    obj = obj_path(build_dir, src_rel)
    obj.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        nvcc_path,
        f"-std={cxx_std}",
        opt,
        "-c",
        str(src),
        "-o",
        str(obj),
        "--compiler-options",
        "-fPIC",
    ]
    cmd += list(defines)
    cmd += include_flags(project_root, repo_root)
    cmd += _gencode_flags(sms)

    if debug:
        cmd += ["-g", "-G"]

    run(cmd)
    return obj
