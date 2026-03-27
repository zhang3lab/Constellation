import os
import re
import shutil
import subprocess
from pathlib import Path

from .toolchain import include_flags, obj_path, run, resolve_src

_FALLBACK_SMS = ["89"]


def _capture(cmd):
    return subprocess.check_output(cmd, text=True)


def _resolve_nvcc(explicit_nvcc: str | None):
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


def _compile_cu(
    nvcc: str,
    project_root: Path,
    repo_root: Path,
    build_dir: Path,
    src_rel: str,
    cxx_std: str,
    opt: str,
    defines,
    debug: bool,
    sms,
):
    src = resolve_src(project_root, src_rel)
    obj = obj_path(build_dir, src_rel)
    obj.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        nvcc,
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


def build_objects(
    nvcc: str,
    project_root: Path,
    repo_root: Path,
    build_dir: Path,
    sources,
    cxx_std: str,
    opt: str,
    defines,
    debug: bool,
):
    if not sources:
        return []

    nvcc_path = _resolve_nvcc(nvcc)
    sms = _resolve_sms()

    objs = []
    for src_rel in sources:
        objs.append(
            _compile_cu(
                nvcc=nvcc_path,
                project_root=project_root,
                repo_root=repo_root,
                build_dir=build_dir,
                src_rel=src_rel,
                cxx_std=cxx_std,
                opt=opt,
                defines=defines,
                debug=debug,
                sms=sms,
            )
        )
    return objs
