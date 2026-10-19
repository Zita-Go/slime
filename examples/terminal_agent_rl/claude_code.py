"""Terminal任务的root Claude Code Harness。"""

from __future__ import annotations

import json
import os
import shlex

from slime.agent.harness.claude_code import ClaudeCodeHarness
from slime.agent.harness.common import HarnessContext, run_agent
from slime.agent.sandbox import Sandbox


class RootClaudeCodeHarness(ClaudeCodeHarness):
    """复用标准CC协议，仅将配置与工具执行用户改为root。"""

    name = "claude_code_root"
    launch_flags = (
        "--dangerously-skip-permissions --allow-dangerously-skip-permissions "
        "--output-format stream-json --include-partial-messages --include-hook-events --verbose"
    )

    async def run(
        self,
        sb: Sandbox,
        *,
        workdir: str,
        session_id: str,
        adapter_url: str,
        time_budget_sec: int,
        prompt: str,
    ) -> int:
        context = HarnessContext(workdir=workdir, session_id=session_id, adapter_url=adapter_url)
        await self.write_config(sb, context)
        return await self.launch_and_wait(sb, context, prompt, time_budget_sec)

    async def write_config(self, sb: Sandbox, ctx: HarnessContext) -> None:
        settings = json.dumps({"hasCompletedOnboarding": True, "bypassPermissionsModeAccepted": True})
        await sb.exec(
            "mkdir -p /root/.claude && "
            f"echo {shlex.quote(settings)} "
            "| tee /root/.claude.json /root/.claude/settings.json > /dev/null",
            user="root",
            check=True,
            timeout=60,
        )

    async def launch_and_wait(self, sb: Sandbox, ctx: HarnessContext, prompt: str, time_budget_sec: int) -> int:
        extra_args = shlex.split(os.environ.get(self.extra_args_env, ""))
        # 同时覆盖新旧CC子Agent工具名，并保留用户额外禁用的工具。
        disallowed = ["Task", "Agent", "WebFetch", "WebSearch"]
        args = ["/usr/local/bin/claude", "-p", prompt, *shlex.split(self.launch_flags)]
        index = 0
        while index < len(extra_args):
            arg = extra_args[index]
            index += 1
            if arg == "--disallowedTools":
                while index < len(extra_args) and not extra_args[index].startswith("-"):
                    disallowed.extend(extra_args[index].split(","))
                    index += 1
            elif arg.startswith("--disallowedTools="):
                disallowed.extend(arg.split("=", 1)[1].split(","))
            else:
                args.append(arg)
        args.extend(["--disallowedTools", *dict.fromkeys(tool for tool in disallowed if tool)])
        command = shlex.join(args)
        env = {
            "ANTHROPIC_BASE_URL": ctx.adapter_url,
            "ANTHROPIC_AUTH_TOKEN": ctx.session_id,
            "ANTHROPIC_MODEL": ctx.model_label,
            "IS_SANDBOX": "1",
            **self.static_env,
        }
        if extra_envs := os.environ.get(self.extra_envs_env, "").strip():
            env.update(json.loads(extra_envs))
        return await run_agent(
            sb,
            workdir=ctx.workdir,
            start_cmd=command,
            env=env,
            user="root",
            time_budget_sec=time_budget_sec,
        )
