"""训练只提交固定checkpoint评测请求；后台worker独立消费，不接触训练推理权重。"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

logger = logging.getLogger(__name__)
DEFAULT_QUEUE = "/gpfs/shared_data/zhangguoqiang/terminal-code/RL/tb21-eval-queue"
BUDGET_KEYS = (
    "TERMINAL_RL_AGENT_TIME_BUDGET_SEC",
    "TERMINAL_RL_MAX_TURNS",
    "TERMINAL_RL_REASONING_EFFORT",
    "SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict, *, replace: bool = False) -> None:
    """只发布完整JSON；请求使用link的O_EXCL语义，拒绝覆盖不同请求。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(name, path)
        else:
            try:
                os.link(name, path)
            except FileExistsError:
                if json.loads(path.read_text()) != value:
                    raise ValueError(f"refusing conflicting evaluation request: {path}") from None
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


def build_request(args, rollout_id: int, config) -> dict:
    if rollout_id < 0 or config.name != "tb21" or config.n_samples_per_eval_prompt != 1:
        raise ValueError("background eval requires a saved training step and complete tb21/one-attempt config")
    root = Path(args.save).resolve()
    dataset = Path(config.path).resolve()
    return {
        "schema": "slime.tb21-background.v1",
        "rollout_id": rollout_id,
        "optimizer_step": rollout_id + 1,
        "checkpoint_root": str(root),
        "checkpoint_dir": str(root / f"iter_{rollout_id:07d}"),
        "hf_checkpoint": str(Path(args.hf_checkpoint).resolve()),
        "eval_config": asdict(config),
        "dataset_sha256": sha256(dataset),
        "manifest_sha256": sha256(dataset.with_name("manifest.json")),
        "context_length": args.rollout_max_context_len,
        "tool_parser": args.sglang_tool_call_parser,
        "reasoning_parser": args.sglang_reasoning_parser,
        "agent_env": {key: os.environ[key] for key in BUDGET_KEYS if key in os.environ},
        "tensorboard_dir": str(Path(os.environ["TENSORBOARD_DIR"]).resolve() / "tb21-background"),
        "pin_checkpoint_until": "completed_or_explicit_operator_resolution; no automatic checkpoint deletion",
    }


def enqueue(args, rollout_id, data_source, evaluation=False):
    from slime.rollout.base_types import RolloutFnEvalOutput

    if not evaluation:
        raise ValueError("TB2.1 queue is eval-only")
    if (rollout_id + 1) % args.eval_interval:
        return RolloutFnEvalOutput(data={})
    try:
        if len(args.eval_datasets) != 1:
            raise ValueError("TB2.1 queue requires exactly one dataset")
        request = build_request(args, rollout_id, args.eval_datasets[0])
        run_key = hashlib.sha256(request["checkpoint_root"].encode()).hexdigest()[:20]
        name = f"{run_key}-step-{rollout_id + 1:08d}.json"
        queue = Path(os.environ.get("TB21_EVAL_QUEUE", DEFAULT_QUEUE))
        atomic_json(queue / "requests" / name, request)
        logger.info("[tb21-background] queued step=%d request=%s; training continues", rollout_id + 1, name)
        # queued不是模型分数；后台worker按来源step另写TensorBoard。
        return RolloutFnEvalOutput(data={}, metrics={"eval/tb21/queued": 1, "eval/tb21/enqueue_failed": 0})
    except Exception:
        # 文件系统异常必须可见，但不能把评测失败变成训练失败。
        logger.exception("[tb21-background] ENQUEUE FAILED step=%d; inspect logs, no score emitted", rollout_id + 1)
        return RolloutFnEvalOutput(data={}, metrics={"eval/tb21/queued": 0, "eval/tb21/enqueue_failed": 1})


def checkpoint_published(request: dict) -> bool:
    root = Path(request["checkpoint_root"])
    checkpoint = Path(request["checkpoint_dir"])
    expected = root / f"iter_{request['rollout_id']:07d}"
    if checkpoint != expected or checkpoint.is_symlink():
        raise ValueError("checkpoint request must name its exact immutable iteration")
    tracker = root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        return False
    value = tracker.read_text().strip()
    if not value.isdecimal() or int(value) < request["rollout_id"]:
        return False
    return (checkpoint / ".metadata").is_file() and (checkpoint / "common.pt").is_file() and any(checkpoint.glob("*.distcp"))


def validate_request(request: dict) -> None:
    if request.get("schema") != "slime.tb21-background.v1":
        raise ValueError("unsupported background eval request")
    if request["optimizer_step"] != request["rollout_id"] + 1:
        raise ValueError("checkpoint step/rollout mismatch")
    if set(request["agent_env"]) - set(BUDGET_KEYS):
        raise ValueError("request contains unsupported environment overrides")
    config = request["eval_config"]
    if config["name"] != "tb21" or config["n_samples_per_eval_prompt"] != 1:
        raise ValueError("TB2.1 must use all tasks and one attempt")
    dataset = Path(config["path"])
    if sha256(dataset) != request["dataset_sha256"] or sha256(dataset.with_name("manifest.json")) != request["manifest_sha256"]:
        raise ValueError("pinned TB2.1 dataset changed since enqueue")


def record_result(request: dict, result: dict, output: Path) -> None:
    from torch.utils.tensorboard import SummaryWriter

    atomic_json(output / "result.json", result)
    with SummaryWriter(request["tensorboard_dir"]) as writer:
        step = request["optimizer_step"]
        writer.add_scalar("eval/tb21/completed", int(result["ok"]), step)
        if result["ok"]:
            writer.add_scalar("eval/tb21", result["success_rate"], step)
            writer.add_scalar("eval/tb21/task_count", 89, step)
        for key in ("elapsed_sec", "completed_at_unix"):
            writer.add_scalar(f"eval/tb21/{key}", result[key], step)
