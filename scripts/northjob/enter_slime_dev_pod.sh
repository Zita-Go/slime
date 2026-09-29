#!/usr/bin/env bash
# 按 NorthJob JobAlias 查找真实 Volcano Job/Pod 名并进入容器。

set -euo pipefail

JOB_NAME=${1:-}
K8S_NAMESPACE=${K8S_NAMESPACE:-bg-agentic-coding}
KUBECTL_BIN=${KUBECTL_BIN:-kubectl}
DRY_RUN=${DRY_RUN:-false}

if [[ -z "$JOB_NAME" ]]; then
    echo "用法: $0 <job-name>" >&2
    exit 2
fi

command -v "$KUBECTL_BIN" >/dev/null 2>&1 || {
    echo "[ERROR] 找不到 kubectl: $KUBECTL_BIN" >&2
    exit 1
}
[[ "$DRY_RUN" == "true" || "$DRY_RUN" == "false" ]] || {
    echo "[ERROR] DRY_RUN 只能是 true 或 false" >&2
    exit 1
}

JOB_JSON=$($KUBECTL_BIN get jobs.batch.volcano.sh -n "$K8S_NAMESPACE" -o json)
ACTUAL_JOB=$(python3 -c '
import json
import sys

alias = sys.argv[1]
items = json.load(sys.stdin).get("items", [])
matches = [
    item
    for item in items
    if item.get("metadata", {}).get("annotations", {}).get("JobAlias") == alias
]
matches.sort(key=lambda item: item.get("metadata", {}).get("creationTimestamp", ""))
if matches:
    print(matches[-1]["metadata"]["name"])
' "$JOB_NAME" <<<"$JOB_JSON")

if [[ -z "$ACTUAL_JOB" ]]; then
    echo "[ERROR] namespace=$K8S_NAMESPACE 中找不到 JobAlias=$JOB_NAME" >&2
    exit 1
fi

POD_JSON=$($KUBECTL_BIN get pods -n "$K8S_NAMESPACE" \
    -l "volcano.sh/job-name=$ACTUAL_JOB" -o json)
read -r POD_NAME CONTAINER_NAME POD_PHASE < <(python3 -c '
import json
import sys

items = json.load(sys.stdin).get("items", [])
items.sort(
    key=lambda item: (
        item.get("metadata", {}).get("labels", {}).get("volcano.sh/task-spec")
        != "master",
        item.get("metadata", {}).get("creationTimestamp", ""),
    )
)
running = [item for item in items if item.get("status", {}).get("phase") == "Running"]
selected = running[0] if running else (items[0] if items else None)
if selected:
    containers = selected.get("spec", {}).get("containers", [])
    container = containers[0]["name"] if containers else "master"
    print(
        selected["metadata"]["name"],
        container,
        selected.get("status", {}).get("phase", "Unknown"),
    )
' <<<"$POD_JSON")

if [[ -z "${POD_NAME:-}" ]]; then
    echo "[ERROR] Volcano Job=$ACTUAL_JOB 尚未创建 Pod" >&2
    exit 1
fi

echo "[INFO] alias=$JOB_NAME actual_job=$ACTUAL_JOB pod=$POD_NAME phase=$POD_PHASE"
if [[ "$POD_PHASE" != "Running" ]]; then
    echo "[ERROR] Pod 尚未 Running；请稍后重试。" >&2
    $KUBECTL_BIN get pod "$POD_NAME" -n "$K8S_NAMESPACE" -o wide
    exit 1
fi

if [[ "$DRY_RUN" == "true" ]]; then
    printf '%q exec -it -n %q %q -c %q -- bash\n' \
        "$KUBECTL_BIN" "$K8S_NAMESPACE" "$POD_NAME" "$CONTAINER_NAME"
    exit 0
fi

exec "$KUBECTL_BIN" exec -it -n "$K8S_NAMESPACE" \
    "$POD_NAME" -c "$CONTAINER_NAME" -- bash
