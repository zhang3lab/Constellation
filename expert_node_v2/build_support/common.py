import shlex
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


def include_flags(project_root: Path, repo_root: Path):
    return [
        "-I", str(project_root),
        "-I", str(repo_root),
    ]
