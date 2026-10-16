"""Python3.13 Harbor桥：只使用TB2.1原生环境和verifier，不载入训练数据。"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import signal
from pathlib import Path

from harbor.agents.base import BaseAgent
from nexau_harbor.e2b_environment import NexAUE2BEnvironment

from .agent import PROMPT_PREFIX, TerminalToolAgent
from .tb21_timeouts import (
    AGENT_SETUP_TIMEOUT_SEC,
    VERIFIER_TIMEOUT_SEC,
    agent_time_budget_sec,
    harbor_agent_timeout_sec,
    max_turns,
    reasoning_effort,
)


class SlimeTerminalAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "slime-terminal-bash"

    def version(self) -> str:
        return "1"

    async def setup(self, environment) -> None:
        # 不上传host task目录：私有tests与solution由Harbor管理。
        pass

    async def run(self, instruction, environment, context) -> None:
        workdir = environment._workdir or "/"
        agent = TerminalToolAgent(workdir=str(workdir))
        result = await agent.run(
            _HarborSandbox(environment),
            session_id=os.environ["TB21_ADAPTER_SESSION"],
            adapter_url=os.environ["TB21_ADAPTER_URL"],
            time_budget_sec=agent_time_budget_sec(),
            prompt=PROMPT_PREFIX.replace("/app", str(workdir)) + instruction,
        )
        context.metadata = {
            "harness": self.name(),
            "agent_exit_code": result.exit_code,
            "turns": result.turns,
            "timed_out": result.timed_out,
        }


class _HarborSandbox:
    def __init__(self, environment):
        self.environment = environment

    async def exec(self, cmd, *, user="root", env=None, timeout=120, check=False, idempotent=True):
        result = await self.environment.exec(cmd, user=user, env=env, timeout_sec=timeout)
        return result.return_code, result.stdout or "", result.stderr or ""


class AuditedE2BEnvironment(NexAUE2BEnvironment):
    async def _create_sandbox(self) -> None:
        await super()._create_sandbox()
        # 即使进程后来被平台回收，也留下本trial拥有的精确sandbox身份。
        (self.trial_paths.trial_dir / "sandbox-created.json").write_text(json.dumps({"sandbox_id": self._sandbox.sandbox_id}) + "\n")

    async def stop(self, delete: bool) -> None:
        from e2b.exceptions import NotFoundException

        sandbox = self._sandbox
        await super().stop(delete)
        confirmed = sandbox is None
        if sandbox is not None and delete:
            try:
                await sandbox.get_info()
            except NotFoundException:
                confirmed = True
        receipt = {
            "live_sandbox_cleanup_confirmed": confirmed,
            "sandbox_id": sandbox.sandbox_id if sandbox else None,
        }
        (self.trial_paths.trial_dir / "sandbox-cleanup.json").write_text(json.dumps(receipt) + "\n")
        if not confirmed:
            raise RuntimeError("TB2.1 live sandbox cleanup was not confirmed")


def checked_reward(result: dict, cleanup: dict) -> float:
    if result.get("exception_info") is not None:
        raise RuntimeError("TB2.1 Harbor trial failed; inspect retained result.json")
    if cleanup.get("live_sandbox_cleanup_confirmed") is not True:
        raise RuntimeError("TB2.1 sandbox cleanup missing")
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward")
    if isinstance(reward, bool) or reward not in (0, 1):
        raise ValueError("TB2.1 native verifier must return binary reward")
    return float(reward)


async def run_trial(options) -> None:
    from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig, VerifierConfig
    from harbor.trial.trial import Trial

    from .e2b_config import resolve_e2b_connection
    from .e2b_network import configure_bridge_network

    if importlib.metadata.version("harbor") != "0.21.0":
        raise RuntimeError("TB2.1 runner requires pinned Harbor 0.21.0")
    key, api_url, domain = resolve_e2b_connection()
    os.environ["E2B_API_KEY"] = key
    if api_url:
        os.environ["E2B_API_URL"] = api_url
    if domain:
        os.environ["E2B_DOMAIN"] = domain
    configure_bridge_network()
    os.environ["TB21_ADAPTER_URL"] = options.adapter_url
    os.environ["TB21_ADAPTER_SESSION"] = options.session_id
    # TB2.1对齐step90评测语义，不继承训练侧400轮的优化吞吐保护。
    os.environ["TERMINAL_RL_MAX_TURNS"] = str(max_turns())
    os.environ["TERMINAL_RL_REASONING_EFFORT"] = reasoning_effort()
    budget = agent_time_budget_sec()
    config = TrialConfig(
        task=TaskConfig(path=options.task_dir),
        trial_name="trial",
        trials_dir=options.output,
        agent=AgentConfig(
            import_path="examples.terminal_agent_rl.tb21_harbor:SlimeTerminalAgent",
            model_name="slime-actor",
            override_timeout_sec=harbor_agent_timeout_sec(budget),
            override_setup_timeout_sec=AGENT_SETUP_TIMEOUT_SEC,
        ),
        environment=EnvironmentConfig(
            import_path="examples.terminal_agent_rl.tb21_harbor:AuditedE2BEnvironment",
            delete=True,
        ),
        verifier=VerifierConfig(disable=False, override_timeout_sec=VERIFIER_TIMEOUT_SEC),
    )
    trial = await Trial.create(config)
    # SIGTERM先取消Trial，令Harbor进入finally清理，不直接杀死仍持有sandbox的进程。
    current = asyncio.current_task()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, current.cancel)
    result = await trial.run()
    cleanup = json.loads((options.output / "trial" / "sandbox-cleanup.json").read_text())
    payload = result.model_dump(mode="json")
    reward = checked_reward(payload, cleanup)
    options.receipt.write_text(
        json.dumps(
            {
                "ok": True,
                "reward": reward,
                "cleanup": cleanup,
                "native_result": str(options.output / "trial" / "result.json"),
                "harness": "slime-terminal-bash",
                "harbor_version": "0.21.0",
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--adapter-url", required=True)
    parser.add_argument("--session-id", required=True)
    asyncio.run(run_trial(parser.parse_args()))


if __name__ == "__main__":
    main()
