from __future__ import annotations

import os
import shutil
from pathlib import Path


SRC = Path("/model/ModelScope/deepseek-ai/DeepSeek-V3.1")
DST = Path("tmp/deepseek_restricted_ref")
SKIP = {"modeling_deepseek_v3.py"}


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)

    for src_path in SRC.iterdir():
        dst_path = DST / src_path.name

        if src_path.name in SKIP:
            continue

        if dst_path.exists() or dst_path.is_symlink():
            continue

        os.symlink(src_path.resolve(), dst_path)

    patch_info = DST / "PATCH_INFO.txt"
    patch_info.write_text(
        f"source={SRC}\npatched_file=modeling_deepseek_v3.py\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
