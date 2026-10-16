#!/usr/bin/env bash
# 当前恢复三步通过后，在同一空闲64卡上一次只执行一个提速候选；不申请卡或重启Ray。
set -euo pipefail
VARIANT=${1:?用法: run_performance_trial.sh rollout8|pack16k|deepep}
SLIME_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${LOAD_PATH:?必须提供已经完整验收的同一来源checkpoint root}"
: "${PERF_ACCEPTANCE_RECEIPT:?必须提供三步恢复成功及来源checkpoint验收回执}"
: "${RUN_ROOT:?必须提供本候选独立且尚不存在的RUN_ROOT}"
: "${PROMPT_DATA:?必须显式提供冻结138题JSONL}"
CONCURRENCY=4
CAP=8192
DISPATCHER=alltoall
case "$VARIANT" in
    rollout8) CONCURRENCY=8 ;;
    pack16k) CAP=16384 ;;
    deepep) DISPATCHER=deepep ;;
    *) echo "未知候选: $VARIANT" >&2; exit 1 ;;
esac
NUM_ROLLOUT=$(python3 - "$LOAD_PATH" "$RUN_ROOT" "$PERF_ACCEPTANCE_RECEIPT" <<'PY'
import json
import shutil
import sys
from pathlib import Path
load, run, receipt = map(Path, sys.argv[1:])
proof = json.loads(receipt.read_text())
assert proof['ok'] is True and proof['additional_optimizer_updates_verified'] >= 3
assert Path(proof['checkpoint_root']).resolve() == load.resolve(), '验收必须覆盖实际LOAD_PATH'
assert not run.exists(), '禁止覆盖既有RUN'
assert shutil.disk_usage(run.parent).free >= 700_000_000_000, '单步完整checkpoint空间不足'
last = int((load / 'latest_checkpointed_iteration.txt').read_text().strip())
assert all((load / f'iter_{last:07d}' / name).is_file() for name in ('.metadata', 'common.pt'))
import ray
from ray.job_submission import JobSubmissionClient
ray.init(address='auto', ignore_reinit_error=True, logging_level='ERROR', log_to_driver=False)
assert sum(n['Alive'] for n in ray.nodes()) == 8
assert ray.cluster_resources()['GPU'] == ray.available_resources()['GPU'] == 64
assert not [j for j in JobSubmissionClient('http://127.0.0.1:8265').list_jobs()
            if j.submission_id and str(j.status) in ('RUNNING', 'PENDING')], '另有训练，拒绝叠加'
print(last + 2)  # start=last+1，只新增一次完整optimizer更新。
PY
)
cd "$SLIME_DIR"
exec env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    SLIME_ACTOR_NODES=4 SLIME_ACTOR_GPUS_PER_NODE=8 SLIME_ROLLOUT_GPUS=32 \
    TP_SIZE=8 CP_SIZE=4 PP_SIZE=1 EP_SIZE=32 ETP_SIZE=1 \
    SGLANG_SERVER_CONCURRENCY="$CONCURRENCY" MAX_TOKENS_PER_GPU="$CAP" MOE_DISPATCHER="$DISPATCHER" \
    LOG_PROBS_CHUNK_SIZE=1024 TRAINING_PHASE=smoke NUM_ROLLOUT="$NUM_ROLLOUT" SAVE_INTERVAL=1 \
    USE_TENSORBOARD=1 ENABLE_TB21_EVAL=0 ROLLOUT_BATCH_SIZE=8 N_SAMPLES_PER_PROMPT=16 GLOBAL_BATCH_SIZE=128 \
    LOAD_PATH="$LOAD_PATH" RUN_ROOT="$RUN_ROOT" PROMPT_DATA="$PROMPT_DATA" \
    GLOO_SOCKET_IFNAME=bond0 NCCL_SOCKET_IFNAME=bond0 \
    bash examples/terminal_agent_rl/run_nex_n25_mini.sh
