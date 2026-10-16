"""在已同步的Slime推理权重上执行完整TB2.1，只返回评测指标。"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
from pathlib import Path

from .e2b_network import e2b_subprocess_env
from .tb21_dataset import COUNT, VERSION, validate_dataset
from .tb21_timeouts import trial_timeout_sec


def validate_receipt(receipt: dict) -> float:
    if receipt.get("ok") is not True or receipt.get("cleanup", {}).get("live_sandbox_cleanup_confirmed") is not True:
        raise RuntimeError("TB2.1 trial/cleanup did not pass; inspect retained trial logs")
    value = receipt.get("reward")
    if isinstance(value, bool) or value not in (0, 1):
        raise ValueError("TB2.1 eval reward must be binary")
    return float(value)


async def _trial(row, config, output, state):
    session_id = f"tb21-eval-{secrets.token_hex(16)}"
    defaults = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "max_new_tokens": config.max_response_len,
        "stop_token_ids": config.stop_token_ids,
    }
    if config.top_p is not None and config.top_p < 1.0:
        # Agent训练依赖逐token top-p支持集合；独立评测服务必须显式请求，
        # 不能在Adapter中放宽缺失数据校验或事后伪造候选集合。
        defaults["custom_params"] = {"return_top_p_token_ids": True}
    output.mkdir()
    receipt_path = output / "receipt.json"
    env = e2b_subprocess_env(dict(os.environ))
    # Harbor使用独立运行时自带SDK，不受训练专用vendored E2B/Python3.12模块污染。
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    process = None
    state.adapter.open_session(session_id, sampling_defaults=defaults, max_context_tokens=state.max_context_len)
    try:
        with (output / "runner.log").open("wb") as log:
            process = await asyncio.create_subprocess_exec(
                os.environ["TB21_RUNTIME_PYTHON"],
                "-m",
                "examples.terminal_agent_rl.tb21_harbor",
                "--task-dir",
                str(Path(config.path).parent / "tasks" / row["label"]),
                "--output",
                str(output),
                "--receipt",
                str(receipt_path),
                "--adapter-url",
                state.adapter_url,
                "--session-id",
                session_id,
                env=env,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
            )
            await asyncio.wait_for(process.wait(), timeout=trial_timeout_sec())
        if process.returncode:
            raise RuntimeError(f"TB2.1 trial failed ({row['label']}), see {output / 'runner.log'}")
        receipt = json.loads(receipt_path.read_text())
        return validate_receipt(receipt)
    finally:
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=120)
            except TimeoutError:
                process.kill()
                await process.wait()
        await state.adapter.drop_session(session_id, wait_timeout=30)


async def _evaluate(args, rollout_id, config, *, states=None):
    rows = validate_dataset(Path(config.path))
    concurrency = int(os.environ.get("TB21_CONCURRENCY", "8"))
    if concurrency <= 0:
        raise ValueError("TB21_CONCURRENCY must be positive")
    if states is None:
        from .generate import _AdapterService

        states = [_AdapterService(args)]
    if not states or concurrency % len(states):
        raise ValueError("trial concurrency must be divisible by the number of independent engines")
    output = Path(os.environ["TB21_EVAL_OUTPUT"]) / f"rollout-{rollout_id:06d}"
    # 失败轮次不能被重跑覆盖；恢复时显式选择新的RUN_ROOT。
    output.mkdir(parents=True, exist_ok=False)
    semaphores = [asyncio.Semaphore(concurrency // len(states)) for _ in states]

    async def one(index, row):
        engine = index % len(states)
        async with semaphores[engine]:
            return await _trial(row, config, output / row["label"], states[engine])

    results = await asyncio.gather(*(one(index, row) for index, row in enumerate(rows)), return_exceptions=True)
    failures = [row["label"] for row, value in zip(rows, results, strict=True) if isinstance(value, BaseException)]
    (output / "summary.json").write_text(
        json.dumps(
            {
                "ok": not failures,
                "version": VERSION,
                "expected_tasks": COUNT,
                "failed_tasks": failures,
                "harness": "slime-terminal-bash",
                "rewards": {row["label"]: value for row, value in zip(rows, results, strict=True) if not isinstance(value, BaseException)},
            },
            indent=2,
        )
        + "\n"
    )
    if failures:
        raise RuntimeError(f"TB2.1 evaluation incomplete ({len(failures)}/89 failures); see {output}")
    return results


def evaluate(args, rollout_id, data_source, evaluation=False):
    from slime.rollout.base_types import RolloutFnEvalOutput
    from slime.utils.async_utils import run

    if not evaluation:
        raise ValueError("TB2.1 is eval-only; it must never enter the training data source")
    # Slime还会在epoch边界调用eval；全量TB2.1只按用户选择的更新间隔运行。
    if (rollout_id + 1) % args.eval_interval:
        return RolloutFnEvalOutput(data={})
    datasets = args.eval_datasets
    if len(datasets) != 1 or datasets[0].name != "tb21" or datasets[0].n_samples_per_eval_prompt != 1:
        raise ValueError("TB2.1 eval requires the complete tb21 dataset with one attempt per task")
    rewards = run(_evaluate(args, rollout_id, datasets[0]))
    return RolloutFnEvalOutput(
        data={"tb21": {"rewards": rewards}},
        metrics={"eval/tb21/task_count": len(rewards)},
    )
