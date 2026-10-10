#!/usr/bin/env bash
# 由 NorthJob train/raw 调用；不使用GPU或启动Ray，验收真实入口与角色环境。

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
JOB_NAME=${1:?缺少 job-name}
RUN_DIR="$(dirname -- "$REPO_ROOT")/RUN/$JOB_NAME"
RUN_DIR="$RUN_DIR" python - <<'PY'
import ipaddress
import json
import os
import sys
from pathlib import Path

import ray
import torch

assert Path(sys.executable).resolve() == Path(os.environ["SLIME_IMAGE_PYTHON"]).resolve()
assert "/envs/northjob" not in sys.executable
assert torch.cuda.device_count() == 0
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert 0 <= rank < world_size
assert os.environ["ROLE"] == ("driver" if rank == 0 else "worker")
ipaddress.ip_address(os.environ["NODE_IP"])
ipaddress.ip_address(os.environ["MASTER_ADDR"])
root = Path(os.environ["RUN_DIR"])
receipt = {
    "ok": True,
    "rank": rank,
    "world_size": world_size,
    "role": os.environ["ROLE"],
    "training_python": sys.executable,
    "torch_version": torch.__version__,
    "ray_version": ray.__version__,
    "gpu_count": 0,
}
path = root / f"northlaunch-smoke-rank{rank}.json"
tmp = path.with_suffix(".tmp")
tmp.write_text(json.dumps(receipt, indent=2) + "\n")
tmp.replace(path)
print(json.dumps(receipt))
PY
