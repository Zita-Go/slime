#!/usr/bin/env bash
# 仅在额外申请且已独立验收的8GPU评测Pod本地，通过monitor托管本前台进程。
set -euo pipefail
SLIME_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${EVAL_EXPECTED_POD:?必须传入已核验的独立评测Pod全名，禁止在训练master误启动}"
[[ "$(hostname)" == "$EVAL_EXPECTED_POD" ]] || { echo "不是目标评测Pod" >&2; exit 1; }
export TB21_EVAL_QUEUE=${TB21_EVAL_QUEUE:-/gpfs/shared_data/zhangguoqiang/terminal-code/RL/tb21-eval-queue}
export TB21_RUNTIME_PYTHON=${TB21_RUNTIME_PYTHON:-/gpfs/users/i-zhangguoqiang/terminal-code/Harbor-nexau/.venv/bin/python}
export TERMINAL_RL_E2B_CONFIG_FILE=${TERMINAL_RL_E2B_CONFIG_FILE:-/gpfs/users/i-zhangguoqiang/terminal-code/RL/.secrets/nextask-e2b.json}
export TERMINAL_RL_E2B_PROXY=${TERMINAL_RL_E2B_PROXY:-http://10.51.6.27:5023}
[[ -f "$TERMINAL_RL_E2B_CONFIG_FILE" && ! -L "$TERMINAL_RL_E2B_CONFIG_FILE" ]] || exit 1
MODE=$(stat -c %a "$TERMINAL_RL_E2B_CONFIG_FILE")
(( (8#$MODE & 8#077) == 0 )) || { echo "E2B config权限过宽" >&2; exit 1; }
export PYTHONPATH="$SLIME_DIR:/root/Megatron-LM"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
PYTHONPATH="$SLIME_DIR" "$TB21_RUNTIME_PYTHON" -c 'import importlib.metadata as m; from examples.terminal_agent_rl.tb21_harbor import SlimeTerminalAgent; assert m.version("harbor") == "0.21.0"'
cd "$SLIME_DIR"
exec env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    python3 -u -m examples.terminal_agent_rl.tb21_worker --queue "$TB21_EVAL_QUEUE" "$@"
