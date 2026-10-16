"""额外8卡上串行消费后台评测队列；4×TP2，总trial并发20。"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from .tb21_queue import BUDGET_KEYS, DEFAULT_QUEUE, atomic_json, checkpoint_published, record_result, validate_request

logger = logging.getLogger(__name__)


def server_command(model: Path, port: int, context: int) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tp-size",
        "2",
        "--trust-remote-code",
        "--context-length",
        str(context),
        "--mem-fraction-static",
        "0.75",
        "--chunked-prefill-size",
        "8192",
        "--max-running-requests",
        "5",
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
        "--chat-template",
        str(model / "chat_template.jinja"),
        "--mamba-scheduler-strategy",
        "extra_buffer",
    ]


def stop_servers(processes) -> None:
    for process in processes:
        # leader崩溃时也要清理本worker通过start_new_session创建的子进程组。
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 60
    for process in processes:
        try:
            process.wait(timeout=max(1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_evaluation(request: dict, output: Path) -> dict:
    from slime.agent.adapters import OpenAIAdapter
    from slime.agent.aiohttp_threaded import run_app_in_thread
    from slime.utils.eval_config import EvalDatasetConfig
    from slime.utils.processing_utils import load_tokenizer

    from .tb21_eval import _evaluate
    from .tb21_snapshot import export_snapshot

    validate_request(request)
    if not checkpoint_published(request):
        raise ValueError("checkpoint has not been published")
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    atomic_json(output / "request.json", request)
    for key in BUDGET_KEYS:
        if key in request["agent_env"]:
            os.environ[key] = request["agent_env"][key]
        else:
            os.environ.pop(key, None)
    os.environ["TB21_CONCURRENCY"] = "20"
    os.environ["TB21_EVAL_OUTPUT"] = str(output / "trials")
    model = output / "hf-snapshot"
    export_snapshot(Path(request["checkpoint_dir"]), Path(request["hf_checkpoint"]), model, request["optimizer_step"])
    converted_at = time.time()
    tokenizer = load_tokenizer(str(model), trust_remote_code=True)
    processes, logs, states = [], [], []
    ports = [31000 + i for i in range(4)]
    try:
        for index, port in enumerate(ports):
            log = (output / f"sglang-{index}.log").open("xb")
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=f"{index * 2},{index * 2 + 1}")
            processes.append(subprocess.Popen(server_command(model, port, request["context_length"]), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 1800
        while True:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("dedicated SGLang exited; inspect retained engine logs")
            ready = []
            for port in ports:
                try:
                    with client.open(f"http://127.0.0.1:{port}/health_generate", timeout=2) as response:
                        ready.append(response.status == 200)
                except OSError:
                    ready.append(False)
            if all(ready):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("4 dedicated inference engines did not become ready in 30min")
            time.sleep(5)
        for port in ports:
            fork = request["agent_env"].get("SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS")
            adapter = OpenAIAdapter(tokenizer=tokenizer, sglang_url=f"http://127.0.0.1:{port}", tool_parser=request["tool_parser"], reasoning_parser=request["reasoning_parser"], fork_threshold_tokens=int(fork) if fork else None)
            handle = run_app_in_thread(adapter.app, host="127.0.0.1", port=0, runner_kwargs={"handler_cancellation": True})
            states.append(SimpleNamespace(adapter=adapter, adapter_url=f"http://127.0.0.1:{handle.port}", max_context_len=request["context_length"], handle=handle))
        atomic_json(output / "inference-ready.json", {"ok": True, "optimizer_step": request["optimizer_step"], "engines": 4, "tp_size": 2, "total_concurrency": 20, "per_engine_concurrency": 5, "fixed_model_path": str(model), "pids": [p.pid for p in processes]})
        rewards = asyncio.run(_evaluate(None, request["rollout_id"], EvalDatasetConfig(**request["eval_config"]), states=states))
        if len(rewards) != 89:
            raise ValueError("incomplete full89 evaluation")
        completed = time.time()
        return {"ok": True, "optimizer_step": request["optimizer_step"], "checkpoint_dir": request["checkpoint_dir"], "task_count": 89, "success_rate": sum(rewards) / 89, "elapsed_sec": completed - started, "snapshot_export_sec": converted_at - started, "completed_at_unix": completed}
    finally:
        for state in states:
            state.handle.stop()
        stop_servers(processes)
        for log in logs:
            log.close()
        atomic_json(output / "inference-cleanup.json", {"owned_process_groups_signaled": [p.pid for p in processes], "leader_returncodes": [p.returncode for p in processes], "gpu_memory_released_independently_verified": False})


def consume_one(queue: Path) -> bool:
    """每个请求最多执行一次；失败/中断不覆盖，恢复需明确换新请求身份。"""
    requests = sorted((queue / "requests").glob("*.json"), key=lambda p: (p.stat().st_mtime_ns, p.name))
    for path in requests:
        output = queue / "jobs" / path.stem
        status = output / "status.json"
        if status.exists():
            continue
        request = json.loads(path.read_text())
        # 等待checkpoint发布不阻塞训练，也不让worker用不完整的文件启动模型。
        if not checkpoint_published(request):
            return False
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(status, {"status": "running", "optimizer_step": request["optimizer_step"], "started_at_unix": time.time()})
        atomic_json(queue / "worker.json", {"pid": os.getpid(), "status": "evaluating", "request": path.name, "optimizer_step": request["optimizer_step"], "gpu_count": 8, "trial_concurrency": 20}, replace=True)
        started = time.time()
        try:
            result = run_evaluation(request, output / "evaluation")
        except Exception as exc:
            logger.exception("background TB2.1 failed: %s", path)
            result = {"ok": False, "optimizer_step": request["optimizer_step"], "error": f"{type(exc).__name__}: {exc}", "elapsed_sec": time.time() - started, "completed_at_unix": time.time()}
        record_result(request, result, output)
        atomic_json(status, {"status": "completed" if result["ok"] else "failed", **result}, replace=True)
        atomic_json(queue / "worker.json", {"pid": os.getpid(), "status": "waiting_for_fixed_checkpoint", "last_request": path.name, "last_ok": result["ok"], "gpu_count": 8, "trial_concurrency": 20}, replace=True)
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=Path(DEFAULT_QUEUE))
    parser.add_argument("--once", action="store_true", help="只检查/消费一个已发布的请求，供验收；不是长期服务")
    args = parser.parse_args()
    import torch

    if torch.cuda.device_count() != 8:
        raise RuntimeError("dedicated eval worker requires exactly 8 visible GPUs")
    args.queue.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    # 与训练无共享Ray、NCCL进程组或模型服务；只锁独立评测队列。
    with (args.queue / ".worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(json.loads(path.read_text())["status"] == "running" for path in (args.queue / "jobs").glob("*/status.json")):
            raise RuntimeError("orphan running evaluation requires operator inspection; do not reuse GPUs or overwrite partial output")

        def interrupt(*_):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupt)
        atomic_json(args.queue / "worker.json", {"pid": os.getpid(), "status": "waiting_for_fixed_checkpoint", "gpu_count": 8, "engines": 4, "tp_size": 2, "trial_concurrency": 20, "inference_deployed": False}, replace=True)
        while True:
            consumed = consume_one(args.queue)
            if args.once:
                return
            if not consumed:
                time.sleep(30)


if __name__ == "__main__":
    main()
