#!/usr/bin/env bash
# 在专用 >=2×H100 Pod 中顺序执行 SGLang 与 Slime rollout 冒烟。

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
JOB_NAME=${JOB_NAME:?必须设置 JOB_NAME}
RUN_BASE=${RUN_BASE:-/gpfs/users/i-zhangguoqiang/terminal-code/RL/RUN/$JOB_NAME}
SMOKE_CONTEXT_LEN=${SMOKE_CONTEXT_LEN:-262144}

receipt_ok() {
    python - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path
path=Path(sys.argv[1])
expected_context=int(sys.argv[2])
try:
    payload=json.loads(path.read_text(encoding="utf-8"))
except (OSError,json.JSONDecodeError):
    raise SystemExit(1)
valid=payload.get("ok") is True and payload.get("context_length")==expected_context
raise SystemExit(0 if valid else 1)
PY
}

if receipt_ok "$RUN_BASE/sglang-smoke/receipt.json" "$SMOKE_CONTEXT_LEN"; then
    echo "[SKIP] SGLang smoke 已有成功 receipt"
else
    RUN_ROOT="$RUN_BASE/sglang-smoke" SMOKE_CONTEXT_LEN="$SMOKE_CONTEXT_LEN" bash "$SCRIPT_DIR/smoke_sglang_nex_n25_mini.sh"
fi

if receipt_ok "$RUN_BASE/slime-rollout-smoke/receipt.json" "$SMOKE_CONTEXT_LEN"; then
    echo "[SKIP] Slime rollout smoke 已有成功 receipt"
else
    RUN_ROOT="$RUN_BASE/slime-rollout-smoke" SMOKE_CONTEXT_LEN="$SMOKE_CONTEXT_LEN" bash "$SCRIPT_DIR/smoke_slime_rollout_nex_n25_mini.sh"
fi

python - <<PY
import json
from pathlib import Path
root=Path(${RUN_BASE@Q})
sg=json.loads((root/'sglang-smoke/receipt.json').read_text())
sl=json.loads((root/'slime-rollout-smoke/receipt.json').read_text())
assert sg.get('ok') is True and sl.get('ok') is True
out={'ok':True,'sglang':sg,'slime_rollout':sl}
tmp=root/'gpu-smokes-receipt.tmp'
tmp.write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
tmp.replace(root/'gpu-smokes-receipt.json')
print(json.dumps(out,ensure_ascii=False,sort_keys=True))
PY

echo "[OK] GPU smoke receipt: $RUN_BASE/gpu-smokes-receipt.json"
