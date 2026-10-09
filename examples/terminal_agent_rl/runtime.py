"""Async subprocess client for :mod:`examples.terminal_agent_rl.bridge`."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .e2b_config import resolve_e2b_connection
from .e2b_network import e2b_subprocess_env

logger = logging.getLogger(__name__)


class BridgeInvocationError(RuntimeError):
    """The pinned NexTask bridge could not complete an operation."""


def _positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class BridgeConfig:
    runtime_root: Path
    runtime_python: Path
    bridge_script: Path
    work_root: Path
    workspace_timeout_sec: int
    grade_timeout_sec: int
    cleanup_timeout_sec: int
    keep_artifacts: bool

    @classmethod
    def from_env(cls) -> BridgeConfig:
        runtime_root = Path(
            os.environ.get(
                "NEXTASK_RUNTIME_ROOT",
                "/gpfs/users/i-zhangguoqiang/terminal-code/NexTask-rfc0012-v3-runtime",
            )
        ).resolve()
        # Do not resolve this symlink: it is the virtualenv entrypoint. Resolving
        # it to the base interpreter drops the NexTask site-packages.
        runtime_python = Path(
            os.environ.get(
                "NEXTASK_RUNTIME_PYTHON",
                "/gpfs/users/i-zhangguoqiang/terminal-code/NexTask-rfc0012/.venv/bin/python",
            )
        ).expanduser()
        return cls(
            runtime_root=runtime_root,
            runtime_python=runtime_python,
            bridge_script=Path(__file__).with_name("bridge.py").resolve(),
            work_root=Path(os.environ.get("TERMINAL_RL_VERIFIER_WORK_ROOT", "/tmp/slime-terminal-agent")).resolve(),
            workspace_timeout_sec=_positive_int("TERMINAL_RL_WORKSPACE_TIMEOUT_SEC", 600),
            grade_timeout_sec=_positive_int("TERMINAL_RL_GRADE_TIMEOUT_SEC", 2400),
            cleanup_timeout_sec=_positive_int("TERMINAL_RL_CLEANUP_TIMEOUT_SEC", 120),
            keep_artifacts=_truthy("TERMINAL_RL_KEEP_VERIFIER_ARTIFACTS"),
        )

    def validate(self) -> None:
        if not self.runtime_python.is_file():
            raise BridgeInvocationError(f"NexTask Python is missing: {self.runtime_python}")
        if not (self.runtime_root / "src" / "pipeline").is_dir():
            raise BridgeInvocationError(f"NexTask runtime is incomplete: {self.runtime_root}")
        if not (self.runtime_root / "nexau" / "nexau").is_dir():
            raise BridgeInvocationError(f"NexAU runtime is incomplete: {self.runtime_root / 'nexau'}")
        if not self.bridge_script.is_file():
            raise BridgeInvocationError(f"bridge script is missing: {self.bridge_script}")

    def subprocess_env(self) -> dict[str, str]:
        env = dict(os.environ)
        roots = [str(self.runtime_root / "src"), str(self.runtime_root / "nexau")]
        if current := env.get("PYTHONPATH"):
            roots.append(current)
        env["PYTHONPATH"] = os.pathsep.join(roots)
        api_key, api_url, domain = resolve_e2b_connection()
        env["NEXTASK_BENCH_E2B_API_KEY"] = api_key
        if api_url:
            env["NEXTASK_BENCH_E2B_API_URL"] = api_url
        if domain:
            env["NEXTASK_BENCH_E2B_DOMAIN"] = domain
        return e2b_subprocess_env(env)


CONFIG = BridgeConfig.from_env()


def session_paths(session_id: str) -> dict[str, Path]:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in session_id)
    if not safe or safe in {".", ".."}:
        raise ValueError("session_id has no safe path representation")
    root = CONFIG.work_root / f"{safe}-{os.getpid()}-{secrets.token_hex(4)}"
    return {
        "root": root,
        "candidate": root / "candidate",
        "verifier_work": root / "verifier_work",
        "result": root / "result.json",
    }


async def _run_bridge(command: str, *arguments: str, timeout: int) -> dict[str, Any]:
    CONFIG.validate()
    proc = await asyncio.create_subprocess_exec(
        str(CONFIG.runtime_python),
        str(CONFIG.bridge_script),
        command,
        *arguments,
        cwd=str(CONFIG.runtime_root),
        env=CONFIG.subprocess_env(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError as exc:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=10)
        except TimeoutError:
            proc.kill()
            await proc.wait()
        raise BridgeInvocationError(f"bridge {command} timed out after {timeout}s") from exc

    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")
    lines = [line for line in stdout_text.splitlines() if line.strip()]
    payload: dict[str, Any] | None = None
    if lines:
        try:
            parsed = json.loads(lines[-1])
            payload = parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            payload = None
    if proc.returncode != 0 or payload is None:
        detail = (stderr_text or stdout_text)[-3000:]
        raise BridgeInvocationError(f"bridge {command} failed (exit={proc.returncode}): {detail}")
    if stderr_text.strip():
        logger.debug("NexTask bridge %s stderr: %s", command, stderr_text[-3000:])
    return payload


async def grade_sandbox(*, sandbox_id: str, package_dir: Path, paths: dict[str, Path]) -> dict[str, Any]:
    paths["root"].mkdir(parents=True, exist_ok=False)
    paths["candidate"].mkdir()
    return await _run_bridge(
        "grade",
        "--sandbox-id",
        sandbox_id,
        "--package",
        str(package_dir),
        "--candidate-dir",
        str(paths["candidate"]),
        "--work-root",
        str(paths["verifier_work"]),
        "--output",
        str(paths["result"]),
        timeout=CONFIG.grade_timeout_sec,
    )


async def materialize_workspace(*, sandbox_id: str, package_dir: Path) -> dict[str, Any]:
    return await _run_bridge(
        "materialize",
        "--sandbox-id",
        sandbox_id,
        "--package",
        str(package_dir),
        timeout=CONFIG.workspace_timeout_sec,
    )


async def cleanup_candidate(paths: dict[str, Path]) -> None:
    candidate = paths["candidate"]
    if not candidate.exists():
        return
    try:
        await _run_bridge(
            "cleanup",
            "--candidate-dir",
            str(candidate),
            timeout=CONFIG.cleanup_timeout_sec,
        )
    except Exception:
        logger.exception("failed to clean temporary candidate snapshot: %s", candidate)


def cleanup_artifacts(paths: dict[str, Path], *, force: bool = False) -> None:
    if CONFIG.keep_artifacts and not force:
        return
    root = paths["root"]
    if root.exists() and root.is_dir() and not root.is_symlink():
        shutil.rmtree(root)
