"""NexTask terminal-agent rollout entrypoint for slime."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import secrets
import time
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slime.agent.adapters import AnthropicAdapter, OpenAIAdapter
from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from . import runtime
from .agent import PROMPT_PREFIX, TerminalToolAgent
from .cc_relay import E2BPullRelay
from .claude_code import RootClaudeCodeHarness
from .sandbox import NextaskE2BSandbox

logger = logging.getLogger(__name__)
logging.getLogger("e2b").setLevel(logging.WARNING)


@dataclass(frozen=True)
class TerminalConfig:
    agent_name: str
    cc_transport: str
    cc_auto_compact_window: int
    cc_count_exact_margin: int
    adapter_public_host: str | None
    adapter_bind_host: str
    adapter_port: int
    fork_merge_threshold: int | None
    agent_time_budget_sec: int
    rollout_guard_sec: int
    boot_concurrency: int
    boot_retries: int

    @classmethod
    def from_env(cls) -> TerminalConfig:
        agent_name = os.environ.get("TERMINAL_RL_AGENT", "claude_code").strip().lower()
        if agent_name not in {"claude_code", "terminal_tool"}:
            raise ValueError("TERMINAL_RL_AGENT must be claude_code or terminal_tool")
        cc_transport = os.environ.get("TERMINAL_RL_CC_TRANSPORT", "e2b_pull").strip().lower()
        if cc_transport not in {"direct", "e2b_pull"}:
            raise ValueError("TERMINAL_RL_CC_TRANSPORT must be direct or e2b_pull")
        cc_auto_compact_window = int(os.environ.get("TERMINAL_RL_CC_AUTO_COMPACT_WINDOW", "196608"))
        cc_count_exact_margin = int(os.environ.get("TERMINAL_RL_CC_COUNT_EXACT_MARGIN", "32768"))
        if cc_auto_compact_window <= 0:
            raise ValueError("TERMINAL_RL_CC_AUTO_COMPACT_WINDOW must be positive")
        if not 0 <= cc_count_exact_margin < cc_auto_compact_window:
            raise ValueError("TERMINAL_RL_CC_COUNT_EXACT_MARGIN must be in [0, auto compact window)")
        agent_budget = int(os.environ.get("TERMINAL_RL_AGENT_TIME_BUDGET_SEC", "14400"))
        guard = int(os.environ.get("TERMINAL_RL_ROLLOUT_GUARD_SEC", "0") or 0)
        if guard <= 0:
            guard = agent_budget + runtime.CONFIG.workspace_timeout_sec + runtime.CONFIG.grade_timeout_sec + 300
        fork = int(value) if (value := os.environ.get("SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS")) else None
        return cls(
            agent_name=agent_name,
            cc_transport=cc_transport,
            cc_auto_compact_window=cc_auto_compact_window,
            cc_count_exact_margin=cc_count_exact_margin,
            adapter_public_host=os.environ.get("ADAPTER_PUBLIC_HOST") or None,
            adapter_bind_host=os.environ.get(
                "ADAPTER_BIND_HOST",
                "0.0.0.0" if agent_name == "claude_code" and cc_transport == "direct" else "127.0.0.1",
            ),
            adapter_port=int(os.environ.get("ADAPTER_PORT", "0")),
            fork_merge_threshold=fork,
            agent_time_budget_sec=agent_budget,
            rollout_guard_sec=guard,
            boot_concurrency=int(os.environ.get("TERMINAL_RL_BOOT_CONCURRENCY", "8")),
            boot_retries=int(os.environ.get("TERMINAL_RL_BOOT_RETRIES", "2")),
        )


CONFIG = TerminalConfig.from_env()
_BOOT_SEM = asyncio.Semaphore(CONFIG.boot_concurrency)
_TERMINAL_AGENT = TerminalToolAgent()
_CLAUDE_CODE = RootClaudeCodeHarness()


@asynccontextmanager
async def boot_task_sandbox(template_identity: str, item_id: str) -> AsyncIterator[NextaskE2BSandbox]:
    sandbox = None
    last_error: Exception | None = None
    for attempt in range(CONFIG.boot_retries):
        candidate = NextaskE2BSandbox(template_identity, timeout=CONFIG.rollout_guard_sec + 300)
        try:
            async with _BOOT_SEM:
                await candidate.__aenter__()
                try:
                    if CONFIG.agent_name == "claude_code":
                        await _CLAUDE_CODE.install_cli(candidate)
                except BaseException:
                    await candidate.__aexit__(None, None, None)
                    raise
            sandbox = candidate
            break
        except Exception as exc:  # noqa: BLE001 - E2B transport errors are retried at this boundary
            last_error = exc
            logger.warning(
                "[terminal_agent_rl] %s boot attempt %d/%d failed: %s",
                item_id,
                attempt + 1,
                CONFIG.boot_retries,
                type(exc).__name__,
            )
            await asyncio.sleep(1 + attempt + random.random())
    if sandbox is None:
        assert last_error is not None
        raise last_error
    try:
        yield sandbox
    finally:
        await sandbox.__aexit__(None, None, None)


class _AdapterService(metaclass=SingletonMeta):
    def __init__(self, args) -> None:
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)
        self.tool_parser = getattr(args, "sglang_tool_call_parser", None) or None
        self.reasoning_parser = getattr(args, "sglang_reasoning_parser", None) or None
        sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        adapter_cls = AnthropicAdapter if CONFIG.agent_name == "claude_code" else OpenAIAdapter
        if CONFIG.agent_name == "claude_code" and CONFIG.cc_transport == "direct" and not CONFIG.adapter_public_host:
            raise RuntimeError(
                "ADAPTER_PUBLIC_HOST is required for Claude Code because the E2B sandbox must reach the Anthropic adapter"
            )
        adapter_options = {}
        if adapter_cls is AnthropicAdapter:
            adapter_options["count_tokens_exact_threshold"] = (
                CONFIG.cc_auto_compact_window - CONFIG.cc_count_exact_margin
            )
        self.adapter = adapter_cls(
            tokenizer=self.tokenizer,
            sglang_url=sglang_url,
            tool_parser=self.tool_parser,
            reasoning_parser=self.reasoning_parser,
            fork_threshold_tokens=CONFIG.fork_merge_threshold,
            **adapter_options,
        )
        self.app_handle = run_app_in_thread(
            self.adapter.app,
            host=CONFIG.adapter_bind_host,
            port=CONFIG.adapter_port,
            thread_name=f"terminal-agent-{CONFIG.agent_name}-adapter",
            runner_kwargs={
                "handler_cancellation": True,
                "access_log_class": FilteredAccessLogger,
            },
        )
        adapter_host = (
            CONFIG.adapter_public_host
            if CONFIG.agent_name == "claude_code" and CONFIG.cc_transport == "direct"
            else "127.0.0.1"
        )
        self.adapter_url = f"http://{adapter_host}:{self.app_handle.port}"
        logger.info(
            "[terminal_agent_rl] harness=%s tokenizer=%s adapter=%s context=%d tool_parser=%s reasoning_parser=%s",
            CONFIG.agent_name,
            args.hf_checkpoint,
            self.adapter_url,
            self.max_context_len,
            self.tool_parser,
            self.reasoning_parser,
        )


@asynccontextmanager
async def _agent_adapter_endpoint(sandbox: NextaskE2BSandbox, state: _AdapterService) -> AsyncIterator[str]:
    if CONFIG.agent_name == "claude_code" and CONFIG.cc_transport == "e2b_pull":
        async with E2BPullRelay(sandbox, state.adapter_url) as relay:
            yield relay.adapter_url
        return
    yield state.adapter_url


def _task_metadata(sample: Sample) -> dict[str, Any]:
    metadata = sample.metadata or {}
    required = (
        "item_id",
        "task_id",
        "task_name",
        "package_dir",
        "template_identity",
        "workspace_digest",
        "package_schema",
        "verifier_protocol",
    )
    missing = [key for key in required if not str(metadata.get(key) or "").strip()]
    if missing:
        raise ValueError(f"terminal dataset metadata is missing: {missing}")
    if metadata.get("dataset_schema") != "slime.nextask-terminal-agent-dataset.v1":
        raise ValueError(f"unsupported terminal dataset schema: {metadata.get('dataset_schema')!r}")
    if metadata["package_schema"] != "rfc-0012-shell-pairwise-dynamic-package-v3":
        raise ValueError(f"unsupported package schema: {metadata['package_schema']!r}")
    package_dir = Path(str(metadata["package_dir"])).resolve()
    if package_dir.is_symlink() or not package_dir.is_dir():
        raise ValueError(f"package_dir is unavailable: {package_dir}")
    return {**metadata, "package_dir": package_dir}


def _session_id(sample: Sample, item_id: str) -> str:
    if sample.status == Sample.Status.ABORTED:
        # 上次运行已drop_session，旧sid仍在closed中保护迟到请求。
        # 基础设施失败重试须新开会话，不能清除旧tombstone或沿用旧sid。
        return f"terminal-{item_id}-retry-{secrets.token_hex(16)}"
    if sample.session_id:
        return sample.session_id
    if sample.index is not None and sample.group_index is not None:
        return f"terminal-{item_id}-{sample.index}-{sample.group_index}"
    return f"terminal-{item_id}-{secrets.token_hex(8)}"


def _reward_from_grade(grade: dict[str, Any]) -> float | None:
    if grade.get("candidate_snapshot_cleanup_confirmed") is not True:
        return None
    verdict = str(grade.get("verdict") or "").upper()
    if verdict == "EQUAL" and grade.get("ok") is True:
        return 1.0
    if verdict != "NON_EQUAL" or grade.get("ok") is not True:
        return None
    ratio = grade.get("score_ratio")
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        return None
    value = float(ratio)
    return value if 0.0 <= value <= 1.0 else None


async def generate(args, base_sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False):
    """Run one terminal agent, grade its E2B snapshot, and emit trainable tokens."""

    state = _AdapterService(args)
    try:
        metadata = _task_metadata(base_sample)
    except Exception as exc:
        return _abort_result(base_sample, f"invalid_metadata:{type(exc).__name__}", "unknown")

    item_id = str(metadata["item_id"])
    session_id = base_sample.session_id = _session_id(base_sample, item_id)
    paths = runtime.session_paths(session_id)
    state.adapter.open_session(
        session_id,
        sampling_defaults=sampling_params,
        max_context_tokens=state.max_context_len,
    )
    t0 = time.monotonic()
    grade: dict[str, Any] | None = None
    snapshot_cleanup_confirmed = False
    claude_trajectory: str | None = None
    try:
        async with asyncio.timeout(CONFIG.rollout_guard_sec):
            async with boot_task_sandbox(str(metadata["template_identity"]), item_id) as sandbox:
                workspace = await runtime.materialize_workspace(
                    sandbox_id=sandbox.sandbox_id,
                    package_dir=metadata["package_dir"],
                )
                if workspace.get("ok") is not True:
                    raise runtime.BridgeInvocationError(f"workspace materialization failed: {workspace!r}")
                prompt = PROMPT_PREFIX + str(base_sample.prompt)
                async with _agent_adapter_endpoint(sandbox, state) as adapter_url:
                    if CONFIG.agent_name == "claude_code":
                        agent_exit_code = await _CLAUDE_CODE.run(
                            sandbox,
                            workdir="/app",
                            session_id=session_id,
                            adapter_url=adapter_url,
                            time_budget_sec=CONFIG.agent_time_budget_sec,
                            prompt=prompt,
                        )
                        agent_turns = None
                        try:
                            claude_trajectory = await sandbox.read_file(
                                "/app/.harness/trajectory.jsonl", user="root"
                            )
                        except Exception as exc:  # noqa: BLE001 - grading can continue; retain warning for audit
                            logger.warning("[terminal_agent_rl] %s failed to retain CC trajectory: %s", item_id, exc)
                    else:
                        agent_result = await _TERMINAL_AGENT.run(
                            sandbox,
                            session_id=session_id,
                            adapter_url=adapter_url,
                            time_budget_sec=CONFIG.agent_time_budget_sec,
                            prompt=prompt,
                        )
                        agent_exit_code = agent_result.exit_code
                        agent_turns = agent_result.turns
                grade = await runtime.grade_sandbox(
                    sandbox_id=sandbox.sandbox_id,
                    package_dir=metadata["package_dir"],
                    paths=paths,
                )
                snapshot_cleanup_confirmed = grade.get("candidate_snapshot_cleanup_confirmed") is True

            reward = _reward_from_grade(grade)
            if reward is None:
                reason = str(grade.get("reason_code") or grade.get("failure_origin") or "indeterminate")
                return _abort_result(base_sample, f"verifier_indeterminate:{reason}", item_id)
            if evaluation:
                return _eval_result(
                    base_sample,
                    reward=reward,
                    agent_exit_code=agent_exit_code,
                    item_id=item_id,
                    grade=grade,
                )

            samples = await state.adapter.finish_session(
                session_id,
                base_sample=base_sample,
                reward=reward,
                extra_metadata={
                    "item_id": item_id,
                    "task_id": metadata["task_id"],
                    "grading_solved": reward == 1.0,
                    "verifier_verdict": grade.get("verdict"),
                    "verifier_score": grade.get("score"),
                    "verifier_max_score": grade.get("max_score"),
                    "verifier_score_ratio": grade.get("score_ratio"),
                    "agent_harness": CONFIG.agent_name,
                },
            )
            if not samples:
                return _abort_result(base_sample, "adapter_session_empty", item_id)
            elapsed = time.monotonic() - t0
            for sample in samples:
                sample.metadata = {
                    **(sample.metadata or {}),
                    "agent_exit_code": agent_exit_code,
                    "agent_turns": agent_turns,
                    "agent_harness": CONFIG.agent_name,
                    "terminal_rollout_elapsed_s": elapsed,
                }
            logger.info(
                "[terminal_agent_rl] %s reward=%.3f verdict=%s exit=%d elapsed=%.1fs segments=%d",
                item_id,
                reward,
                grade.get("verdict"),
                agent_exit_code,
                elapsed,
                len(samples),
            )
            return samples
    except TimeoutError:
        logger.warning("[terminal_agent_rl] %s rollout timed out after %.1fs", item_id, time.monotonic() - t0)
        return _abort_result(base_sample, "wall_clock_timeout", item_id)
    except Exception as exc:  # noqa: BLE001 - external runtime boundary
        logger.warning(
            "[terminal_agent_rl] %s rollout failed: %s\n%s",
            item_id,
            exc,
            traceback.format_exc(),
        )
        return _abort_result(base_sample, f"exception:{type(exc).__name__}", item_id)
    finally:
        if claude_trajectory is not None:
            try:
                root = Path(paths["root"])
                root.mkdir(parents=True, exist_ok=True)
                temporary = root / f".claude-code-{secrets.token_hex(4)}.tmp"
                temporary.write_text(claude_trajectory, encoding="utf-8")
                temporary.replace(root / "claude-code.jsonl")
            except Exception as exc:  # noqa: BLE001 - cleanup and session drain must still run
                logger.warning("[terminal_agent_rl] %s failed to persist CC trajectory: %s", item_id, exc)
        if not snapshot_cleanup_confirmed:
            await runtime.cleanup_candidate(paths)
        await state.adapter.drop_session(session_id, wait_timeout=30)
        keep_failure = grade is None or _reward_from_grade(grade) is None
        if not keep_failure:
            runtime.cleanup_artifacts(paths)
        await asyncio.sleep(1)


def _abort_result(sample: Sample, reason: str, item_id: str) -> list[Sample]:
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = 0.0
    sample.remove_sample = True
    sample.status = Sample.Status.ABORTED
    sample.metadata = {
        **(sample.metadata or {}),
        "abort_reason": reason,
        "item_id": item_id,
    }
    logger.warning("[terminal_agent_rl] %s aborted: %s", item_id, reason)
    return [sample]


def _eval_result(
    sample: Sample,
    *,
    reward: float,
    agent_exit_code: int,
    item_id: str,
    grade: dict[str, Any],
) -> list[Sample]:
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = reward
    sample.remove_sample = True
    sample.status = Sample.Status.COMPLETED
    sample.metadata = {
        **(sample.metadata or {}),
        "item_id": item_id,
        "agent_exit_code": agent_exit_code,
        "agent_harness": CONFIG.agent_name,
        "grading_solved": reward == 1.0,
        "verifier_verdict": grade.get("verdict"),
        "verifier_score_ratio": grade.get("score_ratio"),
    }
    return [sample]
