"""Root Codex harness for externally isolated terminal tasks."""

from __future__ import annotations

import base64
import json
import os
import shlex

from slime.agent.sandbox import EXIT_TIME_BUDGET_EXCEEDED, Sandbox, exec_and_wait


class RootCodexHarness:
    """Run Codex as root inside the task's E2B sandbox.

    Terminal-Bench style tasks may legitimately modify system paths and manage
    services.  Codex therefore runs as root with its own sandbox disabled; E2B
    remains the outer security boundary.  ``bridge.prepare`` installs the
    pinned CLI before this harness is invoked.
    """

    name = "codex"
    config_path = "/root/.codex/config.toml"

    async def write_config(self, sb: Sandbox, *, adapter_url: str) -> None:
        effort = os.environ.get("TERMINAL_RL_REASONING_EFFORT", "medium").strip().lower()
        if effort not in {"none", "medium", "high"}:
            raise ValueError(f"unsupported TERMINAL_RL_REASONING_EFFORT: {effort!r}")
        body = (
            'model = "slime-actor"\n'
            'model_provider = "slime"\n'
            f'model_reasoning_effort = "{effort}"\n'
            'approval_policy = "never"\n'
            'sandbox_mode = "danger-full-access"\n'
            'web_search = "disabled"\n'
            '\n'
            '[model_providers.slime]\n'
            'name = "slime"\n'
            f'base_url = "{adapter_url}/v1"\n'
            'env_key = "OPENAI_API_KEY"\n'
            'wire_api = "chat"\n'
        )
        encoded = base64.b64encode(body.encode("utf-8")).decode("ascii")
        await sb.exec(
            "mkdir -p /root/.codex /tmp/codex_out && "
            f"echo {shlex.quote(encoded)} | base64 -d > {self.config_path} && "
            f"chmod 600 {self.config_path}",
            user="root",
            check=True,
            timeout=60,
        )

    async def run(
        self,
        sb: Sandbox,
        *,
        session_id: str,
        adapter_url: str,
        time_budget_sec: int,
        prompt: str,
    ) -> int:
        await self.write_config(sb, adapter_url=adapter_url)
        args = [
            "codex",
            "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--ephemeral",
            "--json",
            "--cd",
            "/app",
        ]
        if extra := os.environ.get("TERMINAL_RL_CODEX_EXTRA_ARGS", "").strip():
            args.extend(shlex.split(extra))
        args.append(prompt)
        command = 'export HOME=/root; export PATH="/usr/local/lib/nodejs/node/bin:$PATH"; ' + shlex.join(args)
        env = {
            "OPENAI_API_KEY": session_id,
            "OPENAI_BASE_URL": f"{adapter_url}/v1",
            "CODEX_HOME": "/root/.codex",
        }
        if extra_envs := os.environ.get("TERMINAL_RL_CODEX_EXTRA_ENVS", "").strip():
            parsed = json.loads(extra_envs)
            if not isinstance(parsed, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in parsed.items()):
                raise ValueError("TERMINAL_RL_CODEX_EXTRA_ENVS must be a JSON object of string values")
            env.update(parsed)
        exit_code, _ = await exec_and_wait(
            sb,
            cmd=command,
            user="root",
            env=env,
            out_file="/tmp/codex_out/trajectory.jsonl",
            time_budget_sec=time_budget_sec,
            tag="terminal-codex-run",
            want_output=False,
        )
        return exit_code if exit_code != EXIT_TIME_BUDGET_EXCEEDED else EXIT_TIME_BUDGET_EXCEEDED
