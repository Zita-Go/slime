#!/usr/bin/env bash
# 2×H100：用 slime debug-rollout-only 跑一条真实 NexTask/E2B Agent 轨迹。

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SLIME_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
HF_CHECKPOINT=${HF_CHECKPOINT:-/gpfs/shared_data/zhangguoqiang/models/huggingface.co/nex-agi/Nex-N2.5-mini}
PROMPT_DATA=${PROMPT_DATA:-/gpfs/shared_data/zhangguoqiang/terminal-code/NexTask/nexbench2408-rfc0012-nostage-verified/slime_train.jsonl}
RUN_ROOT=${RUN_ROOT:-/gpfs/users/i-zhangguoqiang/terminal-code/RL/RUN/nex-n25-slime-rollout-smoke}
NEXTASK_RUNTIME_ROOT=${NEXTASK_RUNTIME_ROOT:-/gpfs/users/i-zhangguoqiang/terminal-code/NexTask-rfc0012-v3-runtime}
NEXTASK_RUNTIME_PYTHON=${NEXTASK_RUNTIME_PYTHON:-/gpfs/users/i-zhangguoqiang/terminal-code/NexTask-rfc0012/.venv/bin/python}
TERMINAL_RL_E2B_CONFIG_FILE=${TERMINAL_RL_E2B_CONFIG_FILE:-/gpfs/users/i-zhangguoqiang/terminal-code/RL/.secrets/nextask-e2b.json}
E2B_VENDOR_DIR=${E2B_VENDOR_DIR:-/gpfs/users/i-zhangguoqiang/terminal-code/RL/vendor/e2b-2.21.0}
SMOKE_DATA="$RUN_ROOT/smoke.jsonl"
DEBUG_DUMP="$RUN_ROOT/rollout_{rollout_id}.pt"
RECEIPT="$RUN_ROOT/receipt.json"
NUM_GPUS=${NUM_GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())')}

[[ -f "$HF_CHECKPOINT/model.safetensors.index.json" ]] || { echo "[ERROR] 模型不完整" >&2; exit 1; }
[[ -f "$PROMPT_DATA" ]] || { echo "[ERROR] 训练 JSONL 不存在" >&2; exit 1; }
[[ -f "$TERMINAL_RL_E2B_CONFIG_FILE" ]] || { echo "[ERROR] E2B config 不存在" >&2; exit 1; }
[[ -f "$E2B_VENDOR_DIR/e2b/__init__.py" ]] || { echo "[ERROR] vendored e2b 不存在" >&2; exit 1; }
[[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] && (( NUM_GPUS >= 2 )) || { echo "[ERROR] Slime rollout smoke 至少需要 2 张 GPU" >&2; exit 1; }
mkdir -p "$RUN_ROOT"

PROMPT_DATA="$PROMPT_DATA" SMOKE_DATA="$SMOKE_DATA" python - <<'PY'
import os
from pathlib import Path
source=Path(os.environ["PROMPT_DATA"])
target=Path(os.environ["SMOKE_DATA"])
first=next(line for line in source.read_text(encoding="utf-8").splitlines() if line.strip())
target.write_text(first+"\n",encoding="utf-8")
PY

cd "$SLIME_DIR"
source scripts/models/qwen3.5-35B-A3B.sh

ray stop --force >/dev/null 2>&1 || true
ray start --head --node-ip-address 127.0.0.1 --num-gpus "$NUM_GPUS" --disable-usage-stats --dashboard-host=127.0.0.1 --dashboard-port=8265
cleanup() {
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

export NEXTASK_RUNTIME_ROOT NEXTASK_RUNTIME_PYTHON TERMINAL_RL_E2B_CONFIG_FILE E2B_VENDOR_DIR
export TERMINAL_RL_VERIFIER_WORK_ROOT="$RUN_ROOT/verifier-work"
export TERMINAL_RL_AGENT_TIME_BUDGET_SEC=${TERMINAL_RL_AGENT_TIME_BUDGET_SEC:-300}
export TERMINAL_RL_WORKSPACE_TIMEOUT_SEC=${TERMINAL_RL_WORKSPACE_TIMEOUT_SEC:-600}
export TERMINAL_RL_GRADE_TIMEOUT_SEC=${TERMINAL_RL_GRADE_TIMEOUT_SEC:-2400}
export TERMINAL_RL_BOOT_CONCURRENCY=1
export TERMINAL_RL_MAX_TURNS=${TERMINAL_RL_MAX_TURNS:-3}
export TERMINAL_RL_REASONING_EFFORT=medium
export ADAPTER_BIND_HOST=127.0.0.1
export ADAPTER_PORT=0
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

RUNTIME_ENV_JSON=$(python - <<'PY'
import json, os
keys=(
    "NEXTASK_RUNTIME_ROOT","NEXTASK_RUNTIME_PYTHON","TERMINAL_RL_E2B_CONFIG_FILE","E2B_VENDOR_DIR",
    "TERMINAL_RL_VERIFIER_WORK_ROOT","TERMINAL_RL_AGENT_TIME_BUDGET_SEC","TERMINAL_RL_WORKSPACE_TIMEOUT_SEC",
    "TERMINAL_RL_GRADE_TIMEOUT_SEC","TERMINAL_RL_BOOT_CONCURRENCY","TERMINAL_RL_MAX_TURNS",
    "TERMINAL_RL_REASONING_EFFORT","ADAPTER_BIND_HOST","ADAPTER_PORT","no_proxy","NO_PROXY",
)
env={key:os.environ[key] for key in keys}
env.update(
    PYTHONPATH=f"{os.environ['E2B_VENDOR_DIR']}:/root/Megatron-LM:{os.getcwd()}",
    CUDA_DEVICE_MAX_CONNECTIONS="1",
    NCCL_NVLS_ENABLE="0",
)
print(json.dumps({"env_vars":env}))
PY
)

ray job submit --address=http://127.0.0.1:8265 \
    --runtime-env-json="$RUNTIME_ENV_JSON" \
    -- python3 -u train.py \
    --debug-rollout-only \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "$NUM_GPUS" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$HF_CHECKPOINT" \
    --ref-load "$HF_CHECKPOINT" \
    --save "$RUN_ROOT/checkpoints" \
    --custom-generate-function-path examples.terminal_agent_rl.generate.generate \
    --prompt-data "$SMOKE_DATA" \
    --input-key prompt \
    --label-key label \
    --metadata-key metadata \
    --num-rollout 1 \
    --rollout-batch-size 1 \
    --n-samples-per-prompt 1 \
    --rollout-max-prompt-len 32768 \
    --rollout-max-context-len 32768 \
    --rollout-max-response-len 1024 \
    --rollout-temperature 0.7 \
    --rollout-top-p 0.95 \
    --rollout-top-k 40 \
    --rollout-stop-token-ids 248046 248044 \
    --global-batch-size 1 \
    --micro-batch-size 1 \
    --optimizer adam \
    --lr 1e-6 \
    --lr-decay-style constant \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --rollout-num-gpus 2 \
    --rollout-num-gpus-per-engine 2 \
    --sglang-mem-fraction-static 0.75 \
    --sglang-context-length 32768 \
    --sglang-tool-call-parser qwen3_coder \
    --sglang-reasoning-parser qwen3 \
    --sglang-chat-template "$HF_CHECKPOINT/chat_template.jinja" \
    --sglang-mamba-scheduler-strategy extra_buffer \
    --sglang-server-concurrency 1 \
    --save-debug-rollout-data "$DEBUG_DUMP" \
    2>&1 | tee "$RUN_ROOT/run.log"

RUN_ROOT="$RUN_ROOT" RECEIPT="$RECEIPT" python - <<'PY'
from __future__ import annotations
import json, os
from pathlib import Path
import torch

root=Path(os.environ["RUN_ROOT"])
dumps=sorted(root.glob("rollout_*.pt"))
assert len(dumps)==1,dumps
payload=torch.load(dumps[0],map_location="cpu",weights_only=False)
samples=payload.get("samples")
assert isinstance(samples,list) and samples,samples
for sample in samples:
    assert sample["status"]=="completed",sample.get("status")
    assert sample.get("remove_sample") is False
    assert sample.get("reward") is not None
    tokens=sample.get("tokens") or []
    loss_mask=sample.get("loss_mask") or []
    log_probs=sample.get("rollout_log_probs") or []
    assert tokens and len(loss_mask)==len(log_probs)==sample["response_length"]
    assert sum(loss_mask)>0
    metadata=sample.get("metadata") or {}
    assert metadata.get("item_id")
    assert metadata.get("verifier_verdict") in {"EQUAL","NON_EQUAL"}
receipt={
    "ok":True,
    "rollout_dump":str(dumps[0]),
    "rollout_id":payload.get("rollout_id"),
    "sample_count":len(samples),
    "statuses":[sample["status"] for sample in samples],
    "rewards":[sample["reward"] for sample in samples],
    "response_lengths":[sample["response_length"] for sample in samples],
    "trained_tokens":[sum(sample.get("loss_mask") or []) for sample in samples],
    "verdicts":[(sample.get("metadata") or {}).get("verifier_verdict") for sample in samples],
}
path=Path(os.environ["RECEIPT"])
tmp=path.with_suffix(".tmp")
tmp.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
tmp.replace(path)
print(json.dumps(receipt,ensure_ascii=False,sort_keys=True))
PY

echo "[OK] Slime rollout smoke receipt: $RECEIPT"
