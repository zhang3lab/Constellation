from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


_EXPERT_KEY_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return obj


def dump_json(path: Path, obj: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def should_keep_weight_key(
    key: str,
    *,
    resident_expert_ids: set[int],
) -> bool:
    m = _EXPERT_KEY_RE.match(key)
    if m is None:
        return True

    expert_id = int(m.group(2))
    return expert_id in resident_expert_ids


def ensure_empty_or_create_dir(dst_dir: Path, *, force: bool) -> None:
    if dst_dir.exists():
        if not force:
            raise RuntimeError(
                f"destination already exists: {dst_dir}\n"
                f"Use --force to remove and recreate it."
            )
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)


def symlink_model_dir_contents(
    src_dir: Path,
    dst_dir: Path,
    *,
    skip_names: set[str],
) -> None:
    for src_path in src_dir.iterdir():
        if src_path.name in skip_names:
            continue
        dst_path = dst_dir / src_path.name
        if dst_path.exists() or dst_path.is_symlink():
            continue
        os.symlink(src_path.resolve(), dst_path)


def rewrite_config_json(
    src_config_path: Path,
    dst_config_path: Path,
    *,
    resident_expert_ids: list[int] | None,
) -> None:
    cfg = load_json(src_config_path)

    if resident_expert_ids is None:
        cfg.pop("resident_expert_ids", None)
    else:
        cfg["resident_expert_ids"] = list(resident_expert_ids)

    cfg.pop("quantization_config", None)
    dump_json(dst_config_path, cfg)


def rewrite_weight_index_json(
    src_index_path: Path,
    dst_index_path: Path,
    *,
    resident_expert_ids: list[int] | None,
) -> dict[str, Any]:
    index_obj = load_json(src_index_path)

    metadata = index_obj.get("metadata")
    weight_map = index_obj.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{src_index_path}: missing object field weight_map")

    old_count = len(weight_map)

    if resident_expert_ids is None:
        new_weight_map = dict(weight_map)
    else:
        resident_set = {int(x) for x in resident_expert_ids}
        new_weight_map = {
            key: shard
            for key, shard in weight_map.items()
            if should_keep_weight_key(
                key,
                resident_expert_ids=resident_set,
            )
        }

    new_count = len(new_weight_map)

    out: dict[str, Any] = {"weight_map": new_weight_map}
    if isinstance(metadata, dict):
        out["metadata"] = metadata

    dump_json(dst_index_path, out)

    return {
        "old_weight_count": old_count,
        "new_weight_count": new_count,
        "removed_weight_count": old_count - new_count,
    }


def copy_patched_modeling_file(
    patched_modeling_path: Path,
    dst_dir: Path,
) -> None:
    if not patched_modeling_path.is_file():
        raise FileNotFoundError(f"patched modeling file not found: {patched_modeling_path}")
    shutil.copy2(patched_modeling_path, dst_dir / "modeling_deepseek.py")


def write_patch_info(
    dst_dir: Path,
    *,
    mode: str,
    src_model_dir: Path,
    config_path: Path,
    resident_expert_ids: list[int] | None,
    stats: dict[str, Any],
    patched_modeling_path: Path,
) -> None:
    lines = [
        f"mode={mode}",
        f"source_model_dir={src_model_dir}",
        f"source_config={config_path}",
        f"patched_modeling={patched_modeling_path}",
        (
            f"resident_expert_ids={resident_expert_ids}"
            if resident_expert_ids is not None
            else "resident_expert_ids=<all experts>"
        ),
        f"old_weight_count={stats['old_weight_count']}",
        f"new_weight_count={stats['new_weight_count']}",
        f"removed_weight_count={stats['removed_weight_count']}",
    ]
    (dst_dir / "PATCH_INFO.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default="server/test/config.json",
        help="Path to project config JSON that contains model.root and run.restricted_expert_ids",
    )
    ap.add_argument(
        "--dst",
        type=str,
        required=True,
        help="Destination patched model directory, e.g. tmp/deepseek_restricted_ref",
    )
    ap.add_argument(
        "--patched-modeling",
        type=str,
        required=True,
        help="Path to patched modeling_deepseek.py",
    )
    ap.add_argument(
        "--mode",
        type=str,
        choices=["full", "partial"],
        default="partial",
        help="full: keep all experts; partial: restrict to resident experts",
    )
    ap.add_argument(
        "--resident-experts",
        type=str,
        default=None,
        help="Comma-separated expert ids. If omitted, use run.restricted_expert_ids from config.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Remove destination directory if it already exists.",
    )
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")

    cfg = load_json(config_path)

    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError(f"{config_path}: missing object field model")

    run_cfg = cfg.get("run")
    if not isinstance(run_cfg, dict):
        raise ValueError(f"{config_path}: missing object field run")

    model_root = model_cfg.get("root")
    if not isinstance(model_root, str) or not model_root:
        raise ValueError(f"{config_path}: model.root must be a non-empty string")

    if args.mode == "full":
        resident_expert_ids = None
    else:
        if args.resident_experts is not None:
            resident_expert_ids = [
                int(x.strip()) for x in args.resident_experts.split(",") if x.strip()
            ]
        else:
            restricted = run_cfg.get("restricted_expert_ids")
            if not isinstance(restricted, list) or not restricted:
                raise ValueError(
                    f"{config_path}: run.restricted_expert_ids must be a non-empty list "
                    f"unless --resident-experts is provided"
                )
            resident_expert_ids = [int(x) for x in restricted]

        resident_expert_ids = sorted(set(resident_expert_ids))
        if not resident_expert_ids:
            raise ValueError("resident expert set must be non-empty in partial mode")

    src_model_dir = Path(model_root).resolve()
    if not src_model_dir.is_dir():
        raise FileNotFoundError(f"model.root directory not found: {src_model_dir}")

    dst_dir = Path(args.dst).resolve()
    patched_modeling_path = Path(args.patched_modeling).resolve()

    src_index_path = src_model_dir / "model.safetensors.index.json"
    if not src_index_path.is_file():
        raise FileNotFoundError(f"missing index file: {src_index_path}")

    src_config_json = src_model_dir / "config.json"
    if not src_config_json.is_file():
        raise FileNotFoundError(f"missing model config file: {src_config_json}")

    ensure_empty_or_create_dir(dst_dir, force=bool(args.force))

    skip_names = {
        "config.json",
        "model.safetensors.index.json",
        "modeling_deepseek.py",
        "PATCH_INFO.txt",
    }
    symlink_model_dir_contents(
        src_model_dir,
        dst_dir,
        skip_names=skip_names,
    )

    rewrite_config_json(
        src_config_json,
        dst_dir / "config.json",
        resident_expert_ids=resident_expert_ids,
    )

    stats = rewrite_weight_index_json(
        src_index_path,
        dst_dir / "model.safetensors.index.json",
        resident_expert_ids=resident_expert_ids,
    )

    copy_patched_modeling_file(
        patched_modeling_path,
        dst_dir,
    )

    write_patch_info(
        dst_dir,
        mode=args.mode,
        src_model_dir=src_model_dir,
        config_path=config_path,
        resident_expert_ids=resident_expert_ids,
        stats=stats,
        patched_modeling_path=patched_modeling_path,
    )

    print("[partial-ref] done")
    print(f"[partial-ref] mode          = {args.mode}")
    print(f"[partial-ref] src_model_dir = {src_model_dir}")
    print(f"[partial-ref] dst_dir       = {dst_dir}")
    if resident_expert_ids is None:
        print("[partial-ref] resident_expert_ids = <all experts>")
    else:
        print(f"[partial-ref] resident_expert_ids = {resident_expert_ids}")
    print(f"[partial-ref] old_weight_count    = {stats['old_weight_count']}")
    print(f"[partial-ref] new_weight_count    = {stats['new_weight_count']}")
    print(f"[partial-ref] removed_weight_count= {stats['removed_weight_count']}")
    print()
    print("[partial-ref] next checks:")
    print(f"  ls -l {dst_dir}")
    print(f"  grep -n 'resident_expert_ids' {dst_dir / 'config.json'}")
    print(f"  grep -n 'model.layers.3.mlp.experts.9.' {dst_dir / 'model.safetensors.index.json'}")
    print(f"  grep -n 'model.layers.3.mlp.experts.0.' {dst_dir / 'model.safetensors.index.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
