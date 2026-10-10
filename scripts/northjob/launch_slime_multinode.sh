#!/usr/bin/env bash
# 使用 NorthJob raw launcher 申请多节点 slime/Ray 集群。

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
DEFAULT_WORKSPACE=$(dirname -- "$REPO_ROOT")

NORTHJOB_BIN=${NORTHJOB_BIN:-/gpfs/users/i-zhangguoqiang/envs/northjob-3.3.11/bin/northjob}
NORTHLAUNCH_BIN=${NORTHLAUNCH_BIN:-$(dirname -- "$NORTHJOB_BIN")/northlaunch}
JOB_NAME=${JOB_NAME:-slime-ray-8n8g-$(date -u +%Y%m%d-%H%M%S)}
WORKSPACE=${WORKSPACE:-$DEFAULT_WORKSPACE}

K8S_QUEUE=${K8S_QUEUE:-bg-agentic-coding}
K8S_NAMESPACE=${K8S_NAMESPACE:-bg-agentic-coding}
K8S_PRIORITY=${K8S_PRIORITY:-3}
K8S_IMAGE=${K8S_IMAGE:-harbor.local.clusters/bp/slime@sha256:5f3b0f305853e986f5c79f45d13f413f2f39d3661bb94819e03140d2c72ef0c8}

GPU_RESOURCE=${GPU_RESOURCE:-h100}
NNODES=${NNODES:-8}
GPU_PER_NODE=${GPU_PER_NODE:-8}
MEM_PER_POD=${MEM_PER_POD:-640}
ACTOR_NODES=${ACTOR_NODES:-4}
ROLLOUT_NODES=${ROLLOUT_NODES:-4}
RAY_OBJECT_STORE_MEMORY_BYTES=${RAY_OBJECT_STORE_MEMORY_BYTES:-34359738368}
MODE=${MODE:-interactive}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-}
DRY_RUN=${DRY_RUN:-true}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

require_positive_int() {
    local name=$1
    local value=${!name}
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$name 必须是正整数，当前值: $value"
}

[[ -x "$NORTHJOB_BIN" ]] || die "NorthJob 不可执行: $NORTHJOB_BIN"
[[ "$NORTHLAUNCH_BIN" == /* && -x "$NORTHLAUNCH_BIN" ]] || die "GPFS northlaunch 不可执行: $NORTHLAUNCH_BIN"
[[ -d "$REPO_ROOT/.git" ]] || die "slime 仓库不存在: $REPO_ROOT"
[[ -d "$WORKSPACE" && -w "$WORKSPACE" ]] || die "workspace 不存在或不可写: $WORKSPACE"
[[ "$JOB_NAME" =~ ^[a-z0-9][a-z0-9-]{0,62}$ ]] || \
    die "JOB_NAME 必须是 1-63 位小写字母、数字或连字符: $JOB_NAME"

for name in NNODES GPU_PER_NODE MEM_PER_POD K8S_PRIORITY ACTOR_NODES \
    ROLLOUT_NODES RAY_OBJECT_STORE_MEMORY_BYTES; do
    require_positive_int "$name"
done

(( NNODES >= 2 )) || die "多节点任务要求 NNODES >= 2"
(( GPU_PER_NODE <= 8 )) || die "单节点最多申请 8 张 GPU"
(( K8S_PRIORITY <= 3 )) || die "K8S_PRIORITY 必须在 1 到 3 之间"
(( ACTOR_NODES + ROLLOUT_NODES == NNODES )) || \
    die "ACTOR_NODES + ROLLOUT_NODES 必须等于 NNODES"
[[ "$GPU_RESOURCE" == "h100" || "$GPU_RESOURCE" == "b300" ]] || \
    die "GPU_RESOURCE 只能是 h100 或 b300"
[[ "$MODE" == "interactive" || "$MODE" == "train" ]] || \
    die "MODE 只能是 interactive 或 train"
[[ "$DRY_RUN" == "true" || "$DRY_RUN" == "false" ]] || \
    die "DRY_RUN 只能是 true 或 false"

if [[ "$MODE" == "train" ]]; then
    [[ -n "$TRAIN_SCRIPT" ]] || die "MODE=train 时必须指定 TRAIN_SCRIPT"
    if [[ "$TRAIN_SCRIPT" != /* ]]; then
        TRAIN_SCRIPT="$REPO_ROOT/$TRAIN_SCRIPT"
    fi
    [[ -r "$TRAIN_SCRIPT" ]] || die "训练脚本不可读: $TRAIN_SCRIPT"
else
    TRAIN_SCRIPT=-
fi

RUN_DIR="$WORKSPACE/RUN/$JOB_NAME"
[[ ! -e "$RUN_DIR/cluster.ready" && ! -e "$RUN_DIR/cluster.done" ]] || \
    die "检测到同名任务的旧状态文件，请更换 JOB_NAME: $RUN_DIR"

printf -v NODE_CMD 'bash %q %q %q %q %q %q %q %q' \
    "$SCRIPT_DIR/bootstrap_slime_ray_node.sh" \
    "$JOB_NAME" "$NNODES" "$GPU_PER_NODE" "$ACTOR_NODES" "$ROLLOUT_NODES" \
    "$RAY_OBJECT_STORE_MEMORY_BYTES" "$MODE:$TRAIN_SCRIPT"
printf -v PREPARE_CMD 'source %q %q %q' \
    "$SCRIPT_DIR/prepare_northlaunch.sh" "$JOB_NAME" "$NORTHLAUNCH_BIN"

NORTHJOB_ARGS=(
    create
    --job-type train
    --train-launcher raw
    --user-env-path "$PREPARE_CMD"
    --job-name "$JOB_NAME"
    --workspace "$WORKSPACE"
    --nnodes "$NNODES"
    --nproc-per-node 1
    --gpu-per-node "$GPU_PER_NODE"
    --gpu-resource "$GPU_RESOURCE"
    --mem-per-pod "$MEM_PER_POD"
    --k8s-priority "$K8S_PRIORITY"
    --k8s-queue "$K8S_QUEUE"
    --k8s-namespace "$K8S_NAMESPACE"
    --k8s-images "$K8S_IMAGE"
    "$NODE_CMD"
)

echo "[INFO] job=$JOB_NAME mode=$MODE"
echo "[INFO] resources=$NNODES nodes x $GPU_PER_NODE $GPU_RESOURCE GPU(s) = $((NNODES * GPU_PER_NODE)) GPUs"
echo "[INFO] slime topology=actor ${ACTOR_NODES}n/$((ACTOR_NODES * GPU_PER_NODE))g + rollout ${ROLLOUT_NODES}n/$((ROLLOUT_NODES * GPU_PER_NODE))g"
echo "[INFO] image=$K8S_IMAGE"
echo "[INFO] workspace=$WORKSPACE source=$REPO_ROOT ($(git -C "$REPO_ROOT" rev-parse --short HEAD))"
[[ "$MODE" == "interactive" ]] || echo "[INFO] train_script=$TRAIN_SCRIPT"
echo "[WARN] NorthJob 3.3.11 会强制任务可回收；正式训练仍可能被平台抢占。"

if [[ "$DRY_RUN" == "true" ]]; then
    printf 'NORTHJOB_NO_UPDATE_CHECK=1'
    printf ' %q' "$NORTHJOB_BIN" "${NORTHJOB_ARGS[@]}"
    printf '\n'
    echo "[DRY-RUN] 未提交。使用 DRY_RUN=false 重新执行。"
    exit 0
fi

NORTHJOB_NO_UPDATE_CHECK=1 "$NORTHJOB_BIN" "${NORTHJOB_ARGS[@]}"

echo "[OK] NorthJob 已接受多节点任务: $JOB_NAME"
echo "[INFO] 集群状态目录: $RUN_DIR"
printf '[INFO] 启动结果检查: python3 %q --job-name %q --namespace %q --nodes %q --gpus-per-node %q --run-dir %q\n' \
    "$SCRIPT_DIR/check_slime_startup.py" "$JOB_NAME" "$K8S_NAMESPACE" "$NNODES" "$GPU_PER_NODE" "$RUN_DIR"
printf '[INFO] interactive 模式在 cluster.ready 出现后进入 master：%q %q\n' \
    "$SCRIPT_DIR/enter_slime_dev_pod.sh" "$JOB_NAME"
