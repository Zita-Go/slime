"""TB2.1评测进程共享的超时契约。"""

from __future__ import annotations

import os

DEFAULT_AGENT_TIME_BUDGET_SEC = 6 * 60 * 60
HARBOR_AGENT_TIMEOUT_GRACE_SEC = 60
DEFAULT_TRIAL_POST_AGENT_GRACE_SEC = 3 * 60 * 60
DEFAULT_MAX_TURNS = 100000
DEFAULT_REASONING_EFFORT = "high"


def _positive_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


def agent_time_budget_sec() -> int:
    """返回单题模型做题预算；与RL训练采样预算相互独立。"""
    return _positive_env("TB21_AGENT_TIME_BUDGET_SEC", DEFAULT_AGENT_TIME_BUDGET_SEC)


def max_turns() -> int:
    """以六小时时间预算为主，轮数只保留防失控的极高硬上限。"""
    return _positive_env("TB21_MAX_TURNS", DEFAULT_MAX_TURNS)


def reasoning_effort() -> str:
    """TB2.1固定使用当前模型模板支持的最高思考档。"""
    value = os.environ.get("TB21_REASONING_EFFORT", DEFAULT_REASONING_EFFORT).strip().lower()
    if value != "high":
        raise ValueError(f"TB21_REASONING_EFFORT must be high, got {value!r}")
    return value


def harbor_agent_timeout_sec(budget: int | None = None) -> int:
    """给Agent内部正常收尾留一分钟，不把清理时间算进做题预算。"""
    if budget is None:
        budget = agent_time_budget_sec()
    return budget + HARBOR_AGENT_TIMEOUT_GRACE_SEC


def trial_timeout_sec() -> int:
    """返回包含E2B创建、Agent、verifier和清理的整题墙钟上限。"""
    budget = agent_time_budget_sec()
    default = budget + DEFAULT_TRIAL_POST_AGENT_GRACE_SEC
    timeout = _positive_env("TB21_TRIAL_TIMEOUT_SEC", default)
    minimum = harbor_agent_timeout_sec(budget)
    if timeout <= minimum:
        raise ValueError(
            "TB21_TRIAL_TIMEOUT_SEC must exceed the Harbor Agent timeout "
            f"({minimum}s), got {timeout}s"
        )
    return timeout
