#!/usr/bin/env bash
# 用固定 NexTask v3 runtime 校验题包并导出 slime JSONL。

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
NEXTASK_RUNTIME_ROOT=${NEXTASK_RUNTIME_ROOT:-/gpfs/users/i-zhangguoqiang/terminal-code/NexTask-rfc0012-v3-runtime}
NEXTASK_RUNTIME_PYTHON=${NEXTASK_RUNTIME_PYTHON:-/gpfs/users/i-zhangguoqiang/terminal-code/NexTask-rfc0012/.venv/bin/python}
COLLECTION=${COLLECTION:-/gpfs/shared_data/zhangguoqiang/terminal-code/NexTask/nexbench2408-rfc0012-nostage-verified}
OUTPUT=${OUTPUT:-${COLLECTION}/slime_train.jsonl}

[[ -x "$NEXTASK_RUNTIME_PYTHON" ]] || {
    echo "[ERROR] NexTask Python 不可执行: $NEXTASK_RUNTIME_PYTHON" >&2
    exit 1
}
[[ -d "$NEXTASK_RUNTIME_ROOT/src/pipeline" ]] || {
    echo "[ERROR] NexTask runtime 不完整: $NEXTASK_RUNTIME_ROOT" >&2
    exit 1
}

export PYTHONPATH="$NEXTASK_RUNTIME_ROOT/src:$NEXTASK_RUNTIME_ROOT/nexau${PYTHONPATH:+:$PYTHONPATH}"
exec "$NEXTASK_RUNTIME_PYTHON" "$SCRIPT_DIR/bridge.py" export \
    --collection "$COLLECTION" \
    --output "$OUTPUT"
