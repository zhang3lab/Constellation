#!/usr/bin/env bash
set -euo pipefail

if ! command -v python >/dev/null 2>&1; then
  echo "python not found"
  exit 1
fi

GENCODE_FLAGS="$(
python - <<'PY'
import sys
try:
    import torch
except Exception as e:
    print(f"failed to import torch: {e}", file=sys.stderr)
    sys.exit(1)

n = torch.cuda.device_count()
if n <= 0:
    print("no cuda devices found", file=sys.stderr)
    sys.exit(1)

caps = sorted({torch.cuda.get_device_capability(i) for i in range(n)})
for major, minor in caps:
    print(f"-gencode arch=compute_{major}{minor},code=sm_{major}{minor}")
PY
)"

echo "Using gencode flags:"
echo "$GENCODE_FLAGS"

nvcc -std=c++17 \
  $GENCODE_FLAGS \
  "$@"
