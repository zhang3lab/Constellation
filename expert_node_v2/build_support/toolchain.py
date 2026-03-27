import os
import shlex
import shutil
import subprocess
from pathlib import Path


def quote_cmd(cmd):
    return " ".join(shlex.quote(str(x)) for x in cmd)


def run(cmd, cwd=None):
    print("+", quote_cmd(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def resolve_src(project_root: Path, src_rel: str) -> Path:
    return (project_root / src_rel).resolve()


def obj_path(build_dir: Path, src_rel: str) -> Path:
    safe = src_rel.replace("../", "__PARENT__/").replace("/", "__")
    return build_dir / f"{safe}.o"


def existing_sources(project_root: Path, srcs):
    out = []
    for s in srcs:
        p = resolve_src(project_root, s)
        if p.exists():
            out.append(s)
        else:
            print(f"warning: skip missing source: {s}")
    return out


def common_defines(
    enable_cuda: bool,
    enable_amd: bool,
    enable_intel: bool,
    enable_bf16: bool,
    enable_cuda_bf16: bool,
    debug: bool,
):
    defs = [
        f"-DEXPERT_NODE_V2_ENABLE_CUDA={1 if enable_cuda else 0}",
        f"-DEXPERT_NODE_V2_ENABLE_AMD={1 if enable_amd else 0}",
        f"-DEXPERT_NODE_V2_ENABLE_INTEL={1 if enable_intel else 0}",
        f"-DEXPERT_NODE_V2_ENABLE_BF16={1 if enable_bf16 else 0}",
        f"-DEXPERT_NODE_V2_ENABLE_CUDA_BF16={1 if enable_cuda_bf16 else 0}",
    ]
    if debug:
        defs += ["-g", "-DDEBUG=1"]
    return defs


def include_flags(project_root: Path, repo_root: Path):
    return [
        "-I", str(project_root),
        "-I", str(repo_root),
    ]


def compile_cpp(
    cxx: str,
    project_root: Path,
    repo_root: Path,
    build_dir: Path,
    src_rel: str,
    cxx_std: str,
    opt: str,
    defines,
    debug: bool,
):
    src = resolve_src(project_root, src_rel)
    obj = obj_path(build_dir, src_rel)
    obj.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        cxx,
        f"-std={cxx_std}",
        opt,
        "-c",
        str(src),
        "-o",
        str(obj),
    ]
    cmd += list(defines)
    cmd += include_flags(project_root, repo_root)

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        cmd += ["-I", str(Path(cuda_home) / "include")]

    if debug:
        cmd += ["-g"]

    run(cmd)
    return obj


def link_exe(
    cxx: str,
    output_path: Path,
    objs,
    enable_cuda: bool,
):
    cmd = [cxx, "-o", str(output_path)] + [str(x) for x in objs]

    if enable_cuda:
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_home:
            cmd += ["-L", str(Path(cuda_home) / "lib64")]
        cmd += ["-lcudart", "-ldl"]

    cmd += ["-lpthread"]
    run(cmd)


def clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
        print(f"cleaned: {path}")
