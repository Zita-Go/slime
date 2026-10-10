#!/usr/bin/env bash
# NorthJob raw launcher 在每个节点执行本脚本，组成一个 Ray 集群。

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
WORKSPACE=$(dirname -- "$REPO_ROOT")

JOB_NAME=${1:?缺少 job-name}
EXPECTED_NODES=${2:?缺少节点数}
EXPECTED_GPUS=${3:?缺少单节点 GPU 数}
SLIME_ACTOR_NODES=${4:?缺少 actor 节点数}
SLIME_ROLLOUT_NODES=${5:?缺少 rollout 节点数}
RAY_OBJECT_STORE_MEMORY_BYTES=${6:?缺少 Ray object store 大小}
MODE_AND_SCRIPT=${7:?缺少运行模式}
MODE=${MODE_AND_SCRIPT%%:*}
TRAIN_SCRIPT=${MODE_AND_SCRIPT#*:}

NODE_RANK=${RANK:-unknown}
NODE_ROLE=${ROLE:-unknown}
RUN_DIR="$WORKSPACE/RUN/$JOB_NAME"
NODE_LOG="$RUN_DIR/node-${NODE_RANK}-${NODE_ROLE}.log"
DONE_FILE="$RUN_DIR/cluster.done"
STOP_FILE="$RUN_DIR/cluster.stop"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$NODE_LOG") 2>&1
trap 'node_rc=$?; printf "%s\n" "$node_rc" > "$RUN_DIR/node-${NODE_RANK}-${NODE_ROLE}.exit_code"' EXIT
echo "[INFO] bootstrap entered host=$(hostname) rank=$NODE_RANK role=$NODE_ROLE"

: "${ROLE:?NorthJob 未设置 ROLE}"
: "${MASTER_ADDR:?NorthJob 未设置 MASTER_ADDR}"
: "${NODE_IP:?NorthJob 未设置 NODE_IP}"
: "${WORLD_SIZE:?NorthJob 未设置 WORLD_SIZE}"

[[ "$WORLD_SIZE" == "$EXPECTED_NODES" ]] || {
    echo "[ERROR] WORLD_SIZE=$WORLD_SIZE, expected=$EXPECTED_NODES" >&2
    exit 1
}

export PYTHONPATH="$REPO_ROOT:/root/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export SLIME_ACTOR_NODES SLIME_ROLLOUT_NODES EXPECTED_GPUS
export SLIME_ACTOR_GPUS_PER_NODE="$EXPECTED_GPUS"
export SLIME_ROLLOUT_GPUS=$((SLIME_ROLLOUT_NODES * EXPECTED_GPUS))

echo "[INFO] host=$(hostname) rank=$NODE_RANK role=$ROLE node_ip=$NODE_IP master=$MASTER_ADDR"
echo "[INFO] source=$REPO_ROOT actor_nodes=$SLIME_ACTOR_NODES rollout_nodes=$SLIME_ROLLOUT_NODES"
if (( EXPECTED_GPUS == 0 )); then
    [[ "${SLIME_CPU_STARTUP_SMOKE:-0}" == "1" && "$MODE" == "interactive" ]] || {
        echo "[ERROR] 零GPU只允许显式CPU启动验收" >&2
        exit 1
    }
fi

python - <<'PY'
import os

import slime
import torch

expected = int(os.environ["EXPECTED_GPUS"])
if expected > 0:
    assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() == expected, (
    f"expected {expected} GPUs, got {torch.cuda.device_count()}"
)
assert slime.__file__.startswith(os.environ["PYTHONPATH"].split(":", 1)[0]), slime.__file__
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", expected)
print("slime", slime.__file__)
PY

ray stop --force >/dev/null 2>&1 || true

if [[ "$ROLE" == "driver" ]]; then
    ray start --head \
        --node-ip-address "$NODE_IP" \
        --port 6379 \
        --dashboard-host 0.0.0.0 \
        --dashboard-port 8265 \
        --num-gpus "$EXPECTED_GPUS" \
        --object-store-memory "$RAY_OBJECT_STORE_MEMORY_BYTES" \
        --disable-usage-stats
else
    MASTER_ADDR="$MASTER_ADDR" python - <<'PY'
import os
import socket
import time

host = os.environ["MASTER_ADDR"]
deadline = time.monotonic() + 600
while time.monotonic() < deadline:
    try:
        with socket.create_connection((host, 6379), timeout=3):
            break
    except OSError:
        time.sleep(2)
else:
    raise SystemExit(f"Ray head {host}:6379 was not reachable within 600s")
PY
    ray start \
        --address "$MASTER_ADDR:6379" \
        --node-ip-address "$NODE_IP" \
        --num-gpus "$EXPECTED_GPUS" \
        --object-store-memory "$RAY_OBJECT_STORE_MEMORY_BYTES" \
        --disable-usage-stats
fi

touch "$RUN_DIR/node-${NODE_RANK}.ready"

if [[ "$ROLE" == "driver" ]]; then
    EXPECTED_NODES="$EXPECTED_NODES" EXPECTED_GPUS="$EXPECTED_GPUS" MASTER_ADDR="$MASTER_ADDR" python - <<'PY'
import os
import time

import ray

expected = int(os.environ["EXPECTED_NODES"])
expected_gpus = expected * int(os.environ["EXPECTED_GPUS"])
ray.init(address=f"{os.environ['MASTER_ADDR']}:6379", logging_level="ERROR")
deadline = time.monotonic() + 900
while time.monotonic() < deadline:
    alive = [node for node in ray.nodes() if node.get("Alive")]
    if len(alive) == expected and ray.cluster_resources().get("GPU", 0) == expected_gpus:
        print("[READY] Ray nodes:", len(alive), "resources:", ray.cluster_resources())
        break
    print(f"[WAIT] Ray nodes {len(alive)}/{expected}")
    time.sleep(5)
else:
    raise SystemExit(f"Ray cluster did not reach {expected} nodes within 900s")
ray.shutdown()
PY
    touch "$RUN_DIR/cluster.ready"
fi

if [[ "$MODE" == "train" && "$ROLE" == "driver" ]]; then
    [[ -r "$TRAIN_SCRIPT" ]] || {
        echo "[ERROR] 训练脚本不可读: $TRAIN_SCRIPT" >&2
        exit 1
    }
    echo "[INFO] launching training script: $TRAIN_SCRIPT"
    set +e
    bash "$TRAIN_SCRIPT" 2>&1 | tee -a "$RUN_DIR/training.log"
    train_rc=${PIPESTATUS[0]}
    set -e
    printf '%s\n' "$train_rc" > "$RUN_DIR/training.exit_code"
    touch "$DONE_FILE"
    ray stop --force >/dev/null 2>&1 || true
    exit "$train_rc"
fi

echo "[INFO] node ready; waiting for cluster.done or cluster.stop"
while [[ ! -e "$DONE_FILE" && ! -e "$STOP_FILE" ]]; do
    sleep 10
done

ray stop --force >/dev/null 2>&1 || true
echo "[INFO] node shutdown complete"
