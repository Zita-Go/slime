#!/usr/bin/env bash
# 在 NorthJob GPU Pod 内把 GPFS 上的 slime fork 覆盖为 editable 安装。

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
WORKSPACE=$(dirname -- "$REPO_ROOT")

JOB_NAME=${1:?缺少 job-name}
EXPECTED_GPUS=${2:?缺少期望 GPU 数}
RUN_DIR="$WORKSPACE/RUN/$JOB_NAME"

mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/bootstrap.log") 2>&1

echo "[INFO] bootstrap host=$(hostname) source=$REPO_ROOT"
cd "$REPO_ROOT"
# 镜像内已经包含 slime 的运行与构建依赖。禁用 PEP 517 build isolation，
# 避免无公网 Pod 为临时构建环境重新访问 PyPI 下载 packaging/setuptools。
python -m pip install -e . --no-deps --no-build-isolation

EXPECTED_GPUS="$EXPECTED_GPUS" python - <<'PY'
import os

import sglang
import slime
import torch

expected_gpus = int(os.environ["EXPECTED_GPUS"])
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() == expected_gpus, (
    f"expected {expected_gpus} GPUs, got {torch.cuda.device_count()}"
)

print(
    "torch",
    torch.__version__,
    "cuda",
    torch.version.cuda,
    "gpus",
    torch.cuda.device_count(),
)
print("sglang", getattr(sglang, "__version__", "unknown"), sglang.__file__)
print("slime", slime.__file__)
PY

touch "$RUN_DIR/ready"
echo "[READY] slime GPU dev pod is ready"
exec sleep infinity
