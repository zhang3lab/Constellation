import re
import shutil
import subprocess
from pathlib import Path

from .toolchain import include_flags, obj_path, run, resolve_src

_FALLBACK_SMS = ["89"]


def _capture(cmd):
    return subprocess.check_output(cmd, text=True)


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
    sms = _resolve_sms()
    objs = []
    for src_rel in sources:
        objs.append(
            _compile_cu(
                nvcc=nvcc,
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
