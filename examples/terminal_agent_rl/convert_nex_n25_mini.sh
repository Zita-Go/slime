#!/usr/bin/env bash
# 在单台 8×H100 GPU Pod 中把 Nex-N2.5-mini 转为 Megatron torch_dist。

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SLIME_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
HF_CHECKPOINT=${HF_CHECKPOINT:-/gpfs/shared_data/zhangguoqiang/models/huggingface.co/nex-agi/Nex-N2.5-mini}
REF_MODEL_PATH=${REF_MODEL_PATH:-${HF_CHECKPOINT}_torch_dist}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

[[ -f "$HF_CHECKPOINT/model.safetensors.index.json" ]] || {
    echo "[ERROR] Hugging Face checkpoint 不完整: $HF_CHECKPOINT" >&2
    exit 1
}
[[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || {
    echo "[ERROR] NPROC_PER_NODE 必须是正整数" >&2
    exit 1
}
if [[ -f "$REF_MODEL_PATH/latest_checkpointed_iteration.txt" ]]; then
    echo "[OK] torch_dist 已存在: $REF_MODEL_PATH"
    exit 0
fi
[[ ! -e "$REF_MODEL_PATH" ]] || {
    echo "[ERROR] 输出目录已存在但不是完整 checkpoint，请人工检查后换路径: $REF_MODEL_PATH" >&2
    exit 1
}

cd "$SLIME_DIR"
source scripts/models/qwen3.5-35B-A3B.sh

export PYTHONPATH="/root/Megatron-LM:${SLIME_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nproc-per-node "$NPROC_PER_NODE" \
    tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$HF_CHECKPOINT" \
    --save "$REF_MODEL_PATH"

[[ -f "$REF_MODEL_PATH/latest_checkpointed_iteration.txt" ]] || {
    echo "[ERROR] 转换进程退出成功但 checkpoint tracker 缺失: $REF_MODEL_PATH" >&2
    exit 1
}
echo "[OK] Nex-N2.5-mini torch_dist: $REF_MODEL_PATH"
