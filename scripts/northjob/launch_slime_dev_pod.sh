#!/usr/bin/env bash
# 申请一个可交互的单节点 slime GPU 开发 Pod。

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
DEFAULT_WORKSPACE=$(dirname -- "$REPO_ROOT")

NORTHJOB_BIN=${NORTHJOB_BIN:-/gpfs/users/i-zhangguoqiang/envs/northjob-3.3.11/bin/northjob}
JOB_NAME=${JOB_NAME:-slime-dev-h100x8-$(date -u +%Y%m%d-%H%M%S)}
WORKSPACE=${WORKSPACE:-$DEFAULT_WORKSPACE}

K8S_QUEUE=${K8S_QUEUE:-bg-agentic-coding}
K8S_NAMESPACE=${K8S_NAMESPACE:-bg-agentic-coding}
K8S_PRIORITY=${K8S_PRIORITY:-3}
# 该镜像由集群内部 registry 提供，包含 SGLang v0.5.15.post1/cu129；固定 digest 避免 latest 漂移。
K8S_IMAGE=${K8S_IMAGE:-harbor.local.clusters/bp/slime@sha256:5f3b0f305853e986f5c79f45d13f413f2f39d3661bb94819e03140d2c72ef0c8}

GPU_RESOURCE=${GPU_RESOURCE:-h100}
GPU_PER_NODE=${GPU_PER_NODE:-8}
MEM_PER_POD=${MEM_PER_POD:-640}
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
[[ -d "$REPO_ROOT/.git" ]] || die "slime 仓库不存在: $REPO_ROOT"
[[ -d "$WORKSPACE" && -w "$WORKSPACE" ]] || die "workspace 不存在或不可写: $WORKSPACE"
[[ "$JOB_NAME" =~ ^[a-z0-9][a-z0-9-]{0,62}$ ]] || \
    die "JOB_NAME 必须是 1-63 位小写字母、数字或连字符: $JOB_NAME"

for name in GPU_PER_NODE MEM_PER_POD K8S_PRIORITY; do
    require_positive_int "$name"
done

(( GPU_PER_NODE <= 8 )) || die "单节点最多申请 8 张 GPU，当前值: $GPU_PER_NODE"
(( K8S_PRIORITY <= 3 )) || die "K8S_PRIORITY 必须在 1 到 3 之间"
[[ "$GPU_RESOURCE" == "h100" || "$GPU_RESOURCE" == "b300" ]] || \
    die "GPU_RESOURCE 只能是 h100 或 b300"
[[ "$DRY_RUN" == "true" || "$DRY_RUN" == "false" ]] || \
    die "DRY_RUN 只能是 true 或 false"

RUN_DIR="$WORKSPACE/RUN/$JOB_NAME"

printf -v BOOTSTRAP_CMD 'bash %q %q %q' \
    "$SCRIPT_DIR/bootstrap_slime_dev_pod.sh" "$JOB_NAME" "$GPU_PER_NODE"

NORTHJOB_ARGS=(
    create
    --dev-pod
    --job-name "$JOB_NAME"
    --workspace "$WORKSPACE"
    --nnodes 1
    --nproc-per-node 1
    --gpu-per-node "$GPU_PER_NODE"
    --gpu-resource "$GPU_RESOURCE"
    --mem-per-pod "$MEM_PER_POD"
    --k8s-priority "$K8S_PRIORITY"
    --k8s-queue "$K8S_QUEUE"
    --k8s-namespace "$K8S_NAMESPACE"
    --k8s-images "$K8S_IMAGE"
    "$BOOTSTRAP_CMD"
)

echo "[INFO] job=$JOB_NAME"
echo "[INFO] resources=1 node x $GPU_PER_NODE $GPU_RESOURCE GPU(s), memory=${MEM_PER_POD}Gi"
echo "[INFO] image=$K8S_IMAGE"
echo "[INFO] workspace=$WORKSPACE"
echo "[INFO] source=$REPO_ROOT ($(git -C "$REPO_ROOT" rev-parse --short HEAD))"
echo "[WARN] NorthJob 3.3.11 会强制任务可回收；开发 Pod 仍可能被抢占。"

if [[ "$DRY_RUN" == "true" ]]; then
    printf 'NORTHJOB_NO_UPDATE_CHECK=1'
    printf ' %q' "$NORTHJOB_BIN" "${NORTHJOB_ARGS[@]}"
    printf '\n'
    echo "[DRY-RUN] 未提交。使用 DRY_RUN=false 重新执行以申请 GPU。"
    exit 0
fi

NORTHJOB_NO_UPDATE_CHECK=1 "$NORTHJOB_BIN" "${NORTHJOB_ARGS[@]}"

echo "[OK] NorthJob 已接受任务: $JOB_NAME"
printf '[INFO] 等待 Pod Ready 后进入：%q %q\n' \
    "$SCRIPT_DIR/enter_slime_dev_pod.sh" "$JOB_NAME"
echo "[INFO] bootstrap 日志: $RUN_DIR/bootstrap.log"
