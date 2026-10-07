#!/usr/bin/env bash
# 2×H100：直接验证 Nex-N2.5-mini 的 SGLang 加载、思考模式与工具调用。

set -euo pipefail

HF_CHECKPOINT=${HF_CHECKPOINT:-/gpfs/shared_data/zhangguoqiang/models/huggingface.co/nex-agi/Nex-N2.5-mini}
RUN_ROOT=${RUN_ROOT:-/gpfs/users/i-zhangguoqiang/terminal-code/RL/RUN/nex-n25-sglang-smoke}
PORT=${PORT:-30000}
GPU_IDS=${GPU_IDS:-0,1}
SERVER_LOG="$RUN_ROOT/server.log"
RECEIPT="$RUN_ROOT/receipt.json"

[[ -f "$HF_CHECKPOINT/model.safetensors.index.json" ]] || {
    echo "[ERROR] 模型不完整: $HF_CHECKPOINT" >&2
    exit 1
}
[[ -f "$HF_CHECKPOINT/chat_template.jinja" ]] || {
    echo "[ERROR] chat template 不存在" >&2
    exit 1
}
mkdir -p "$RUN_ROOT"

PORT="$PORT" python - <<'PY'
import socket
import os
port=int(os.environ.get("PORT", "30000"))
with socket.socket() as sock:
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as exc:
        raise SystemExit(f"port {port} is already in use: {exc}")
PY

server_pid=""
cleanup() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM -- "-$server_pid" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$server_pid" 2>/dev/null; then
            kill -KILL -- "-$server_pid" 2>/dev/null || true
        fi
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

setsid env CUDA_VISIBLE_DEVICES="$GPU_IDS" python -u -m sglang.launch_server \
    --model-path "$HF_CHECKPOINT" \
    --served-model-name Nex-N2.5-mini \
    --tp-size 2 \
    --host 127.0.0.1 \
    --port "$PORT" \
    --trust-remote-code \
    --context-length 32768 \
    --mem-fraction-static 0.75 \
    --chunked-prefill-size 8192 \
    --max-running-requests 8 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --chat-template "$HF_CHECKPOINT/chat_template.jinja" \
    --mamba-scheduler-strategy extra_buffer \
    >"$SERVER_LOG" 2>&1 &
server_pid=$!

ready=false
for _ in $(seq 1 240); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "[ERROR] SGLang 提前退出" >&2
        tail -n 200 "$SERVER_LOG" >&2 || true
        exit 1
    fi
    if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health_generate" >/dev/null 2>&1; then
        ready=true
        break
    fi
    sleep 5
done
[[ "$ready" == true ]] || {
    echo "[ERROR] SGLang 20 分钟内未就绪" >&2
    tail -n 200 "$SERVER_LOG" >&2 || true
    exit 1
}

PORT="$PORT" RECEIPT="$RECEIPT" HF_CHECKPOINT="$HF_CHECKPOINT" python - <<'PY'
from __future__ import annotations

import json
import os
import platform
import urllib.request
from pathlib import Path

import sglang
import torch

port=int(os.environ["PORT"])
url=f"http://127.0.0.1:{port}/v1/chat/completions"

def request(payload):
    body=json.dumps(payload).encode()
    req=urllib.request.Request(url, data=body, headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=300) as response:
        assert response.status == 200
        return json.loads(response.read())

common={
    "model":"Nex-N2.5-mini",
    "temperature":0.7,
    "top_p":0.95,
    "top_k":40,
    "reasoning_effort":"medium",
}
plain=request({
    **common,
    "messages":[{"role":"user","content":"Reply with exactly SGLANG_SMOKE_OK."}],
    "max_tokens":128,
})
plain_message=plain["choices"][0]["message"]
assert (plain_message.get("content") or plain_message.get("reasoning_content")), plain

tool={
    "type":"function",
    "function":{
        "name":"bash",
        "description":"Run a shell command.",
        "parameters":{
            "type":"object",
            "properties":{"command":{"type":"string"}},
            "required":["command"],
        },
    },
}
tool_response=request({
    **common,
    "messages":[{"role":"user","content":"Call the bash tool exactly once with command: echo SGLANG_TOOL_OK"}],
    "tools":[tool],
    "tool_choice":{"type":"function","function":{"name":"bash"}},
    "max_tokens":256,
})
tool_message=tool_response["choices"][0]["message"]
calls=tool_message.get("tool_calls") or []
assert calls and calls[0]["function"]["name"] == "bash", tool_response
arguments=calls[0]["function"].get("arguments")
if isinstance(arguments, str):
    arguments=json.loads(arguments)
assert "SGLANG_TOOL_OK" in str(arguments.get("command")), arguments

receipt={
    "ok":True,
    "model_path":os.environ["HF_CHECKPOINT"],
    "torch_version":torch.__version__,
    "cuda_version":torch.version.cuda,
    "sglang_version":getattr(sglang,"__version__","unknown"),
    "python":platform.python_version(),
    "gpu_count_visible":torch.cuda.device_count(),
    "plain_finish_reason":plain["choices"][0].get("finish_reason"),
    "plain_has_content":bool(plain_message.get("content")),
    "plain_has_reasoning":bool(plain_message.get("reasoning_content")),
    "tool_finish_reason":tool_response["choices"][0].get("finish_reason"),
    "tool_name":calls[0]["function"]["name"],
    "tool_arguments":arguments,
}
path=Path(os.environ["RECEIPT"])
tmp=path.with_suffix(".tmp")
tmp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True)+"\n")
tmp.replace(path)
print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
PY

echo "[OK] SGLang smoke receipt: $RECEIPT"
