import shlex
import shutil
import subprocess
from pathlib import Path

from . import cuda_backend


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


def common_defines(feature_defines, debug: bool):
    defs = []
    for name, enabled in feature_defines.items():
        defs.append(f"-D{name}={1 if enabled else 0}")
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
    enable_cuda: bool,
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

    if enable_cuda:
        cmd += cuda_backend.get_cuda_include_flags()

    if debug:
        cmd += ["-g"]

    run(cmd)
    return obj


def resolve_source_kind(src_rel: str, source_rules):
    ext = Path(src_rel).suffix
    kind = source_rules.get(ext)
    if kind is None:
        raise RuntimeError(f"unsupported source extension: {src_rel}")
    return kind


def compile_source(
    project_root: Path,
    repo_root: Path,
    build_dir: Path,
    src_rel: str,
    cxx_std: str,
    opt: str,
    defines,
    debug: bool,
    enable_cuda: bool,
    source_rules,
    toolchains,
):
    kind = resolve_source_kind(src_rel, source_rules)

    if kind == "cpp":
        return compile_cpp(
            cxx=toolchains["cpp"]["compiler"],
            project_root=project_root,
            repo_root=repo_root,
            build_dir=build_dir,
            src_rel=src_rel,
            cxx_std=cxx_std,
            opt=opt,
            defines=defines,
            debug=debug,
            enable_cuda=enable_cuda,
        )

    if kind == "cuda":
        return cuda_backend.compile_source(
            nvcc=toolchains["cuda"]["compiler"],
            project_root=project_root,
            repo_root=repo_root,
            build_dir=build_dir,
            src_rel=src_rel,
            cxx_std=cxx_std,
            opt=opt,
            defines=defines,
            debug=debug,
        )

    raise RuntimeError(f"unsupported source kind: {kind}")


def link_exe(
    cxx: str,
    output_path: Path,
    objs,
    enable_cuda: bool,
):
    cmd = [cxx, "-o", str(output_path)] + [str(x) for x in objs]

    if enable_cuda:
        cmd += cuda_backend.get_cuda_link_flags()

    cmd += ["-lpthread"]
    run(cmd)


def clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
        print(f"cleaned: {path}")
