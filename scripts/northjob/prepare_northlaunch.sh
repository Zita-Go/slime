#!/usr/bin/env bash
# 由 NorthJob --user-env-path source；仅提供 northlaunch，不切换训练环境。

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
JOB_NAME=${1:?缺少 job-name}
NORTHLAUNCH_TARGET=${2:-/gpfs/users/i-zhangguoqiang/envs/northjob-3.3.11/bin/northlaunch}
LINK_DIR=${NORTHLAUNCH_LINK_DIR:-/root/.local/bin}
RUN_DIR=${NORTHLAUNCH_RUN_DIR:-$(dirname -- "$REPO_ROOT")/RUN/$JOB_NAME}
NODE_ID=${RANK:-${HOSTNAME:-unknown}}
NODE_ID=${NODE_ID//[^a-zA-Z0-9_.-]/_}

[[ "$JOB_NAME" =~ ^[a-z0-9][a-z0-9-]{0,62}$ ]] || { echo "[ERROR] job-name 格式错误" >&2; return 1 2>/dev/null || exit 1; }
umask 077
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/launcher-${NODE_ID}.log") 2>&1
trap 'launcher_rc=$?; printf "%s\n" "$launcher_rc" > "$RUN_DIR/launcher-${NODE_ID}.exit_code"' EXIT

echo "[INFO] preparing northlaunch host=$(hostname) rank=${RANK:-unset}"
[[ "$NORTHLAUNCH_TARGET" == /* && -x "$NORTHLAUNCH_TARGET" ]] || {
    echo "[ERROR] GPFS northlaunch 不可执行: $NORTHLAUNCH_TARGET" >&2
    return 1 2>/dev/null || exit 1
}
[[ "$LINK_DIR" == /* ]] || { echo "[ERROR] 入口目录必须是绝对路径" >&2; return 1 2>/dev/null || exit 1; }
TRAIN_PYTHON_BEFORE=$(command -v python) || { echo "[ERROR] 镜像 Python 不存在" >&2; return 1 2>/dev/null || exit 1; }
export SLIME_IMAGE_PYTHON="$TRAIN_PYTHON_BEFORE"
mkdir -p "$LINK_DIR"
NORTHLAUNCH_LINK="$LINK_DIR/northlaunch"
if [[ -e "$NORTHLAUNCH_LINK" || -L "$NORTHLAUNCH_LINK" ]]; then
    [[ -L "$NORTHLAUNCH_LINK" && $(readlink -f "$NORTHLAUNCH_LINK") == $(readlink -f "$NORTHLAUNCH_TARGET") ]] || {
        echo "[ERROR] 拒绝覆盖其他 northlaunch 入口: $NORTHLAUNCH_LINK" >&2
        return 1 2>/dev/null || exit 1
    }
else
    ln -s -- "$NORTHLAUNCH_TARGET" "$NORTHLAUNCH_LINK"
fi
export PATH="$LINK_DIR:$PATH"
[[ $(command -v python) == "$TRAIN_PYTHON_BEFORE" ]] || {
    echo "[ERROR] 准备入口改变了训练 Python，拒绝启动" >&2
    return 1 2>/dev/null || exit 1
}
if northlaunch --help; then
    :
else
    help_rc=$?
    echo "[ERROR] northlaunch --help 失败: exit=$help_rc" >&2
    return "$help_rc" 2>/dev/null || exit "$help_rc"
fi
echo "[READY] northlaunch=$(command -v northlaunch) training_python=$TRAIN_PYTHON_BEFORE"
