#!/usr/bin/env bash
# 在 scripts/northjob/bootstrap_slime_ray_node.sh 已创建的 Ray 集群上训练。

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SLIME_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)

HF_CHECKPOINT=${HF_CHECKPOINT:-/gpfs/shared_data/zhangguoqiang/models/huggingface.co/nex-agi/Nex-N2.5-mini}
REF_MODEL_PATH=${REF_MODEL_PATH:-${HF_CHECKPOINT}_torch_dist}
PROMPT_DATA=${PROMPT_DATA:-/gpfs/shared_data/zhangguoqiang/terminal-code/NexTask/nexbench2408-rfc0012-nostage-verified/slime_train.jsonl}
NEXTASK_RUNTIME_ROOT=${NEXTASK_RUNTIME_ROOT:-/gpfs/users/i-zhangguoqiang/terminal-code/NexTask-rfc0012-v3-runtime}
NEXTASK_RUNTIME_PYTHON=${NEXTASK_RUNTIME_PYTHON:-/gpfs/users/i-zhangguoqiang/terminal-code/RL/vendor/nextask-runtime-py313/bin/python}
TERMINAL_RL_E2B_CONFIG_FILE=${TERMINAL_RL_E2B_CONFIG_FILE:-/gpfs/users/i-zhangguoqiang/terminal-code/RL/.secrets/nextask-e2b.json}
E2B_VENDOR_DIR=${E2B_VENDOR_DIR:-/gpfs/users/i-zhangguoqiang/terminal-code/RL/vendor/e2b-2.21.0}
TERMINAL_RL_E2B_PROXY=${TERMINAL_RL_E2B_PROXY-http://10.51.6.27:5023}
TERMINAL_RL_AGENT=${TERMINAL_RL_AGENT:-claude_code}
TERMINAL_RL_CC_TRANSPORT=${TERMINAL_RL_CC_TRANSPORT:-e2b_pull}
SLIME_AGENT_NODE_TARBALL=${SLIME_AGENT_NODE_TARBALL:-/gpfs/shared_data/zhangguoqiang/terminal-code/RL/vendor/claude-code-2.1.193/node-v20.19.5-linux-x64.tar.gz}
SLIME_AGENT_CC_TARBALL=${SLIME_AGENT_CC_TARBALL:-/gpfs/shared_data/zhangguoqiang/terminal-code/RL/vendor/claude-code-2.1.193/anthropic-ai-claude-code-2.1.193.tgz}
EXPECTED_NODE_TARBALL_SHA256=${EXPECTED_NODE_TARBALL_SHA256:-4eba5fbe1fb10753bc06e42f001a91c5cec16798b7764a3e9257adc59af47fe1}
EXPECTED_CC_TARBALL_SHA256=${EXPECTED_CC_TARBALL_SHA256:-fc0cdf116ef64ee80259a6a71bc96070e116e2583abf6b129cf38fa61fb4967a}

EXPECTED_NEXTASK_COMMIT=${EXPECTED_NEXTASK_COMMIT:-7f746b9832fcedbab4904fdd39d200aab3120ec7}
EXPECTED_NEXAU_COMMIT=${EXPECTED_NEXAU_COMMIT:-4e95828aa9a8c33e5b478e465557351aecfa3d88}

ACTOR_NODES=${SLIME_ACTOR_NODES:-8}
ACTOR_GPUS_PER_NODE=${SLIME_ACTOR_GPUS_PER_NODE:-8}
ROLLOUT_GPUS=${SLIME_ROLLOUT_GPUS:-32}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-2}

MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN:-262144}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-65536}
MAX_GEN_LEN=${MAX_GEN_LEN:-32768}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-16}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
TRAINING_PHASE=${TRAINING_PHASE:-long}
case "$TRAINING_PHASE" in
    smoke) DEFAULT_NUM_ROLLOUT=3; DEFAULT_SAVE_INTERVAL=1 ;;
    long) DEFAULT_NUM_ROLLOUT=100; DEFAULT_SAVE_INTERVAL=5 ;;
    *) echo "[ERROR] TRAINING_PHASE 必须为 smoke 或 long" >&2; exit 1 ;;
esac
NUM_ROLLOUT=${NUM_ROLLOUT:-$DEFAULT_NUM_ROLLOUT}
SAVE_INTERVAL=${SAVE_INTERVAL:-$DEFAULT_SAVE_INTERVAL}
EVAL_INTERVAL=${EVAL_INTERVAL:-5}
ENABLE_TB21_EVAL=${ENABLE_TB21_EVAL:-1}
EVAL_CONFIG=${EVAL_CONFIG:-$SCRIPT_DIR/tb21_eval.yaml}
TB21_RUNTIME_PYTHON=${TB21_RUNTIME_PYTHON:-/gpfs/users/i-zhangguoqiang/terminal-code/Harbor-nexau/.venv/bin/python}
TB21_CONCURRENCY=${TB21_CONCURRENCY:-8}
TB21_EVAL_MODE=${TB21_EVAL_MODE:-async}
export TB21_EVAL_QUEUE=${TB21_EVAL_QUEUE:-/gpfs/shared_data/zhangguoqiang/terminal-code/RL/tb21-eval-queue}
USE_TENSORBOARD=${USE_TENSORBOARD:-1}
SGLANG_SERVER_CONCURRENCY=${SGLANG_SERVER_CONCURRENCY:-4}
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-1024}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}
MOE_DISPATCHER=${MOE_DISPATCHER:-alltoall}

TP_SIZE=${TP_SIZE:-2}
PP_SIZE=${PP_SIZE:-1}
CP_SIZE=${CP_SIZE:-32}
EP_SIZE=${EP_SIZE:-32}
ETP_SIZE=${ETP_SIZE:-1}

EXP_TAG=${EXP_TAG:-nex-n25-mini-nextask-terminal-rl}
STAMP=$(date -u +%Y%m%d-%H%M%S)
RUN_ROOT=${RUN_ROOT:-$SLIME_DIR/runs/${EXP_TAG}_${STAMP}}
SAVE_PATH=${SAVE_PATH:-$RUN_ROOT/checkpoints}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-$RUN_ROOT/tensorboard}
export TB21_EVAL_OUTPUT=${TB21_EVAL_OUTPUT:-$RUN_ROOT/tb21-eval}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

[[ "$LOG_PROBS_CHUNK_SIZE" =~ ^[1-9][0-9]*$ ]] || die "LOG_PROBS_CHUNK_SIZE 必须是正整数"
[[ "$MAX_TOKENS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || die "MAX_TOKENS_PER_GPU 必须是正整数"
[[ "$SAVE_INTERVAL" =~ ^[1-9][0-9]*$ ]] || die "SAVE_INTERVAL 必须是正整数"
[[ "$EVAL_INTERVAL" =~ ^[1-9][0-9]*$ ]] || die "EVAL_INTERVAL 必须是正整数"
[[ "$TB21_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || die "TB21_CONCURRENCY 必须是正整数"
[[ "$USE_TENSORBOARD" =~ ^[01]$ && "$ENABLE_TB21_EVAL" =~ ^[01]$ ]] || die "监测/评测开关必须是0或1"
[[ "$TB21_EVAL_MODE" == async || "$TB21_EVAL_MODE" == sync ]] || die "TB21_EVAL_MODE 必须为async/sync"
[[ "$MOE_DISPATCHER" == alltoall || "$MOE_DISPATCHER" == deepep ]] || die "MOE_DISPATCHER 必须为alltoall/deepep"
[[ "$TERMINAL_RL_AGENT" == claude_code || "$TERMINAL_RL_AGENT" == terminal_tool ]] || die "TERMINAL_RL_AGENT 必须为claude_code/terminal_tool"
[[ "$TERMINAL_RL_CC_TRANSPORT" == direct || "$TERMINAL_RL_CC_TRANSPORT" == e2b_pull ]] || die "TERMINAL_RL_CC_TRANSPORT 必须为direct/e2b_pull"
if [[ "$TERMINAL_RL_AGENT" == claude_code ]]; then
    [[ -f "$SLIME_AGENT_NODE_TARBALL" && ! -L "$SLIME_AGENT_NODE_TARBALL" ]] || die "固定Node包不存在或为符号链接"
    [[ -f "$SLIME_AGENT_CC_TARBALL" && ! -L "$SLIME_AGENT_CC_TARBALL" ]] || die "固定Claude Code包不存在或为符号链接"
    [[ $(sha256sum "$SLIME_AGENT_NODE_TARBALL" | awk '{print $1}') == "$EXPECTED_NODE_TARBALL_SHA256" ]] || die "固定Node包SHA漂移"
    [[ $(sha256sum "$SLIME_AGENT_CC_TARBALL" | awk '{print $1}') == "$EXPECTED_CC_TARBALL_SHA256" ]] || die "固定Claude Code包SHA漂移"
fi

TRACKING_ARGS=()
if [[ "$USE_TENSORBOARD" == 1 ]]; then
    python3 -c 'from torch.utils.tensorboard import SummaryWriter' || die "镜像缺少TensorBoard依赖"
    TRACKING_ARGS+=(--use-tensorboard --tb-project-name terminal-agent-rl --tb-experiment-name "$EXP_TAG")
fi
EVAL_ARGS=()
if [[ "$ENABLE_TB21_EVAL" == 1 ]]; then
    if [[ "$TB21_EVAL_MODE" == async ]]; then
        (( EVAL_INTERVAL % SAVE_INTERVAL == 0 )) || die "后台评测要求SAVE_INTERVAL整除EVAL_INTERVAL，确保目标checkpoint存在"
        EVAL_FUNCTION=examples.terminal_agent_rl.tb21_queue.enqueue
    else
        EVAL_FUNCTION=examples.terminal_agent_rl.tb21_eval.evaluate
    fi
    [[ -x "$TB21_RUNTIME_PYTHON" ]] || die "TB2.1 Harbor Python不可执行"
    PYTHONPATH="$SLIME_DIR" "$TB21_RUNTIME_PYTHON" -c \
        'import importlib.metadata as m; from examples.terminal_agent_rl.tb21_harbor import SlimeTerminalAgent, AuditedE2BEnvironment; assert m.version("harbor") == "0.21.0"'
    PYTHONPATH="$SLIME_DIR" python3 - "$EVAL_CONFIG" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf
from examples.terminal_agent_rl.tb21_dataset import validate_dataset
config = OmegaConf.to_container(OmegaConf.load(sys.argv[1]), resolve=True)["eval"]
assert len(config["datasets"]) == 1 and config["datasets"][0]["name"] == "tb21"
validate_dataset(Path(config["datasets"][0]["path"]))
print("[OK] pinned full TB2.1 eval dataset, 89 tasks")
PY
    EVAL_ARGS+=(--eval-config "$EVAL_CONFIG" --eval-interval "$EVAL_INTERVAL" --skip-eval-before-train
        --eval-function-path "$EVAL_FUNCTION" --n-samples-per-eval-prompt 1)
fi

[[ -f "$HF_CHECKPOINT/model.safetensors.index.json" ]] || die "HF checkpoint 不完整: $HF_CHECKPOINT"
[[ -f "$HF_CHECKPOINT/chat_template.jinja" ]] || die "缺少 chat template: $HF_CHECKPOINT/chat_template.jinja"
[[ -f "$REF_MODEL_PATH/latest_checkpointed_iteration.txt" ]] || die "缺少 torch_dist checkpoint: $REF_MODEL_PATH"
[[ -f "$PROMPT_DATA" ]] || die "训练 JSONL 不存在: $PROMPT_DATA"
PYTHONPATH="$SLIME_DIR${PYTHONPATH:+:$PYTHONPATH}" python3 -m examples.terminal_agent_rl.validate_dataset --prompt-data "$PROMPT_DATA"
[[ -x "$NEXTASK_RUNTIME_PYTHON" ]] || die "NexTask Python 不可执行: $NEXTASK_RUNTIME_PYTHON"
[[ $(git -C "$NEXTASK_RUNTIME_ROOT" rev-parse HEAD) == "$EXPECTED_NEXTASK_COMMIT" ]] || die "NexTask runtime commit 漂移"
[[ $(git -C "$NEXTASK_RUNTIME_ROOT/nexau" rev-parse HEAD) == "$EXPECTED_NEXAU_COMMIT" ]] || die "NexAU runtime commit 漂移"
[[ -f "$TERMINAL_RL_E2B_CONFIG_FILE" && ! -L "$TERMINAL_RL_E2B_CONFIG_FILE" ]] || die "E2B config 不存在或为符号链接"
[[ -f "$E2B_VENDOR_DIR/e2b/__init__.py" ]] || die "缺少兼容内部网关的 vendored e2b 2.21.0: $E2B_VENDOR_DIR"
E2B_MODE=$(stat -c %a "$TERMINAL_RL_E2B_CONFIG_FILE")
(( (8#$E2B_MODE & 8#077) == 0 )) || die "E2B config 权限必须不宽于 0600，当前为 $E2B_MODE"

TOTAL_ACTOR_GPUS=$((ACTOR_NODES * ACTOR_GPUS_PER_NODE))
(( TOTAL_ACTOR_GPUS % (TP_SIZE * PP_SIZE * CP_SIZE) == 0 )) || die "actor GPU 数不能被 TP×PP×CP 整除"
# Megatron的attention与expert分组独立；expert组使用ETP而不是attention TP。
(( TOTAL_ACTOR_GPUS % (ETP_SIZE * EP_SIZE * PP_SIZE) == 0 )) || die "actor GPU 数不能被 ETP×EP×PP 整除"
(( ROLLOUT_GPUS % ROLLOUT_TP_SIZE == 0 )) || die "rollout GPU 数不能被 rollout TP 整除"
(( MAX_CONTEXT_LEN % CP_SIZE == 0 )) || die "MAX_CONTEXT_LEN 必须能被 CP_SIZE 整除"
(( MAX_PROMPT_LEN < MAX_CONTEXT_LEN )) || die "MAX_PROMPT_LEN 必须小于 MAX_CONTEXT_LEN"
(( GLOBAL_BATCH_SIZE == ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT )) || die "每批更新一次时 GLOBAL_BATCH_SIZE 必须等于题数×n"

E2B_VENDOR_DIR="$E2B_VENDOR_DIR" HF_CHECKPOINT="$HF_CHECKPOINT" \
PYTHONPATH="$E2B_VENDOR_DIR:$SLIME_DIR" python3 - <<'PY'
import importlib.metadata as metadata
import json
import os
from pathlib import Path

assert metadata.version("e2b") == "2.21.0", metadata.version("e2b")
config = json.loads((Path(os.environ["HF_CHECKPOINT"]) / "config.json").read_text())
text = config["text_config"]
assert config["model_type"] == "qwen3_5_moe"
assert text["num_hidden_layers"] == 40
assert text["hidden_size"] == 2048
assert text["num_experts"] == 256
assert text["num_experts_per_tok"] == 8
print("[OK] model config and vendored e2b preflight")
PY

mkdir -p "$RUN_ROOT/rollout_dumps" "$SAVE_PATH"
cd "$SLIME_DIR"
source scripts/models/qwen3.5-35B-A3B.sh

CKPT_ARGS=(
    --hf-checkpoint "$HF_CHECKPOINT"
    --ref-load "$REF_MODEL_PATH"
    --save "$SAVE_PATH"
    --save-interval "$SAVE_INTERVAL"
)
if [[ -n "${LOAD_PATH:-}" ]]; then
    # 完整恢复时沿用已保存的scheduler及进度，允许本次NUM_ROLLOUT与原任务不同。
    CKPT_ARGS+=(--load "$LOAD_PATH" --use-checkpoint-opt-param-scheduler)
fi

ROLLOUT_ARGS=(
    --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
    --custom-generate-function-path examples.terminal_agent_rl.generate.generate
    --prompt-data "$PROMPT_DATA"
    --input-key prompt
    --label-key label
    --metadata-key metadata
    --rollout-shuffle
    --num-rollout "$NUM_ROLLOUT"
    --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
    --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT"
    --rollout-max-prompt-len "$MAX_PROMPT_LEN"
    --rollout-max-context-len "$MAX_CONTEXT_LEN"
    --rollout-max-response-len "$MAX_GEN_LEN"
    --rollout-temperature 1.0
    --rollout-top-p 0.95
    --rollout-top-k 40
    --rollout-stop-token-ids 248046 248044
    --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
    --num-steps-per-rollout 1
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --micro-batch-size 1
    --balance-data
    --save-debug-rollout-data "$RUN_ROOT/rollout_dumps/rollout_{rollout_id}.pt"
)

PERF_ARGS=(
    --tensor-model-parallel-size "$TP_SIZE"
    --sequence-parallel
    --pipeline-model-parallel-size "$PP_SIZE"
    --context-parallel-size "$CP_SIZE"
    --expert-model-parallel-size "$EP_SIZE"
    --expert-tensor-parallel-size "$ETP_SIZE"
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU"
    --log-probs-chunk-size "$LOG_PROBS_CHUNK_SIZE"
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.0
    --kl-loss-type low_var_kl
    --kl-coef 0.0
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
)

SGLANG_ARGS=(
    --rollout-num-gpus "$ROLLOUT_GPUS"
    --rollout-num-gpus-per-engine "$ROLLOUT_TP_SIZE"
    --sglang-mem-fraction-static 0.75
    --sglang-context-length "$MAX_CONTEXT_LEN"
    --sglang-tool-call-parser qwen3_coder
    --sglang-reasoning-parser qwen3
    --sglang-chat-template "$HF_CHECKPOINT/chat_template.jinja"
    --sglang-mamba-scheduler-strategy extra_buffer
    --sglang-server-concurrency "$SGLANG_SERVER_CONCURRENCY"
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --moe-token-dispatcher-type alltoall
)
if [[ "$MOE_DISPATCHER" == deepep ]]; then
    # 模型脚本中已有alltoall；最后一个同名参数由argparse覆盖，不同时启用两种dispatcher。
    MISC_ARGS+=(--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep --moe-enable-deepep)
fi

export NEXTASK_RUNTIME_ROOT NEXTASK_RUNTIME_PYTHON TERMINAL_RL_E2B_CONFIG_FILE E2B_VENDOR_DIR
export TERMINAL_RL_E2B_PROXY
export TERMINAL_RL_VERIFIER_WORK_ROOT=${TERMINAL_RL_VERIFIER_WORK_ROOT:-/tmp/slime-terminal-agent}
export TERMINAL_RL_KEEP_VERIFIER_ARTIFACTS=${TERMINAL_RL_KEEP_VERIFIER_ARTIFACTS:-0}
[[ "$TERMINAL_RL_KEEP_VERIFIER_ARTIFACTS" =~ ^[01]$ ]] || die "TERMINAL_RL_KEEP_VERIFIER_ARTIFACTS 必须为0或1"
export TERMINAL_RL_AGENT_TIME_BUDGET_SEC=${TERMINAL_RL_AGENT_TIME_BUDGET_SEC:-14400}
export TERMINAL_RL_WORKSPACE_TIMEOUT_SEC=${TERMINAL_RL_WORKSPACE_TIMEOUT_SEC:-600}
export TERMINAL_RL_GRADE_TIMEOUT_SEC=${TERMINAL_RL_GRADE_TIMEOUT_SEC:-2400}
export TERMINAL_RL_BOOT_CONCURRENCY=${TERMINAL_RL_BOOT_CONCURRENCY:-16}
export TERMINAL_RL_MAX_TURNS=${TERMINAL_RL_MAX_TURNS:-400}
export TERMINAL_RL_REASONING_EFFORT=${TERMINAL_RL_REASONING_EFFORT:-high}
export TERMINAL_RL_AGENT TERMINAL_RL_CC_TRANSPORT SLIME_AGENT_NODE_TARBALL SLIME_AGENT_CC_TARBALL
CC_SETTINGS_JSON='{"permissions":{"defaultMode":"bypassPermissions"},"autoCompactEnabled":true,"autoCompactWindow":196608}'
CC_AGENTS_JSON='{"worker":{"description":"Handle a delegated terminal subtask and report concrete results","prompt":"Work on the delegated subtask inside the current sandbox. Inspect and modify the environment as needed, run relevant checks, and return a concise evidence-based result.","tools":["Read","Edit","Write","Grep","Glob","Bash"]}}'
export SLIME_AGENT_CC_EXTRA_ARGS=${SLIME_AGENT_CC_EXTRA_ARGS:-"--settings '$CC_SETTINGS_JSON' --disable-slash-commands --agents '$CC_AGENTS_JSON' --disallowedTools WebFetch WebSearch"}
export SLIME_AGENT_CC_EXTRA_ENVS=${SLIME_AGENT_CC_EXTRA_ENVS:-'{"MAX_THINKING_TOKENS":"64000"}'}
export TB21_AGENT_TIME_BUDGET_SEC=${TB21_AGENT_TIME_BUDGET_SEC:-21600}
[[ "$TB21_AGENT_TIME_BUDGET_SEC" =~ ^[1-9][0-9]*$ ]] || { echo "TB21_AGENT_TIME_BUDGET_SEC必须为正整数" >&2; exit 2; }
export TB21_TRIAL_TIMEOUT_SEC=${TB21_TRIAL_TIMEOUT_SEC:-$((TB21_AGENT_TIME_BUDGET_SEC + 10800))}
[[ "$TB21_TRIAL_TIMEOUT_SEC" =~ ^[1-9][0-9]*$ ]] || { echo "TB21_TRIAL_TIMEOUT_SEC必须为正整数" >&2; exit 2; }
(( TB21_TRIAL_TIMEOUT_SEC > TB21_AGENT_TIME_BUDGET_SEC + 60 )) || {
    echo "TB21_TRIAL_TIMEOUT_SEC必须大于TB21 Agent预算加60秒" >&2
    exit 2
}
export TB21_MAX_TURNS=${TB21_MAX_TURNS:-100000}
[[ "$TB21_MAX_TURNS" =~ ^[1-9][0-9]*$ ]] || { echo "TB21_MAX_TURNS必须为正整数" >&2; exit 2; }
export TB21_REASONING_EFFORT=${TB21_REASONING_EFFORT:-high}
[[ "$TB21_REASONING_EFFORT" == high ]] || { echo "TB21_REASONING_EFFORT必须为high" >&2; exit 2; }
export ADAPTER_PORT=0
export TB21_RUNTIME_PYTHON TB21_CONCURRENCY

export MASTER_ADDR=${MASTER_ADDR:?NorthJob 未设置 MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT:-6379}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}
if [[ "$TERMINAL_RL_AGENT" == claude_code && "$TERMINAL_RL_CC_TRANSPORT" == direct ]]; then
    export ADAPTER_BIND_HOST=${ADAPTER_BIND_HOST:-0.0.0.0}
    export ADAPTER_PUBLIC_HOST=${ADAPTER_PUBLIC_HOST:-$(hostname -I | awk '{print $1}')}
    [[ -n "$ADAPTER_PUBLIC_HOST" && "$ADAPTER_PUBLIC_HOST" != 127.0.0.1 ]] || die "Claude Code需要E2B可达的ADAPTER_PUBLIC_HOST"
else
    export ADAPTER_BIND_HOST=${ADAPTER_BIND_HOST:-127.0.0.1}
fi
export no_proxy="127.0.0.1,${MASTER_ADDR}${ADAPTER_PUBLIC_HOST:+,$ADAPTER_PUBLIC_HOST}${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"

RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os

keys = (
    "NEXTASK_RUNTIME_ROOT",
    "NEXTASK_RUNTIME_PYTHON",
    "TERMINAL_RL_E2B_CONFIG_FILE",
    "E2B_VENDOR_DIR",
    "TERMINAL_RL_E2B_PROXY",
    "TERMINAL_RL_AGENT",
    "TERMINAL_RL_CC_TRANSPORT",
    "SLIME_AGENT_NODE_TARBALL",
    "SLIME_AGENT_CC_TARBALL",
    "SLIME_AGENT_CC_EXTRA_ARGS",
    "SLIME_AGENT_CC_EXTRA_ENVS",
    "TERMINAL_RL_VERIFIER_WORK_ROOT",
    "TERMINAL_RL_KEEP_VERIFIER_ARTIFACTS",
    "TERMINAL_RL_AGENT_TIME_BUDGET_SEC",
    "TERMINAL_RL_WORKSPACE_TIMEOUT_SEC",
    "TERMINAL_RL_GRADE_TIMEOUT_SEC",
    "TERMINAL_RL_BOOT_CONCURRENCY",
    "TERMINAL_RL_MAX_TURNS",
    "TERMINAL_RL_REASONING_EFFORT",
    "TENSORBOARD_DIR",
    "TB21_RUNTIME_PYTHON",
    "TB21_CONCURRENCY",
    "TB21_AGENT_TIME_BUDGET_SEC",
    "TB21_TRIAL_TIMEOUT_SEC",
    "TB21_MAX_TURNS",
    "TB21_REASONING_EFFORT",
    "TB21_EVAL_OUTPUT",
    "TB21_EVAL_QUEUE",
    "ADAPTER_BIND_HOST",
    "ADAPTER_PUBLIC_HOST",
    "ADAPTER_PORT",
    "MASTER_ADDR",
    "MASTER_PORT",
    "GLOO_SOCKET_IFNAME",
    "NCCL_SOCKET_IFNAME",
    "no_proxy",
    "NO_PROXY",
)
env = {key: os.environ[key] for key in keys if key in os.environ}
env.update(
    PYTHONPATH=f"{os.environ['E2B_VENDOR_DIR']}:/root/Megatron-LM:{os.getcwd()}",
    CUDA_DEVICE_MAX_CONNECTIONS="1",
    NCCL_NVLS_ENABLE="0",
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
)
print(json.dumps({"env_vars": env}))
PY
)

ray status >/dev/null || die "Ray 集群尚未就绪"
echo "[INFO] run_root=$RUN_ROOT actor=${ACTOR_NODES}n/${TOTAL_ACTOR_GPUS}g rollout=${ROLLOUT_GPUS}g"
echo "[INFO] model=$HF_CHECKPOINT data=$PROMPT_DATA"

ray job submit --address=http://127.0.0.1:8265 \
    --runtime-env-json="$RUNTIME_ENV_JSON" \
    -- python3 -u "$SLIME_DIR/train.py" \
    --actor-num-nodes "$ACTOR_NODES" \
    --actor-num-gpus-per-node "$ACTOR_GPUS_PER_NODE" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${TRACKING_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    2>&1 | tee "$RUN_ROOT/run.log"
