"""Host-side terminal tool loop for E2B agent rollouts."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import aiohttp

from slime.agent.sandbox import Sandbox

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an autonomous terminal agent in an isolated task environment.
Use the bash tool to inspect and modify the environment until the user's task is actually complete. Work primarily in /app, but you may modify system paths or services when the task requires it. Run relevant checks before finishing. Do not merely explain commands: execute them. Private tests and reference solutions are unavailable, so solve from the visible task and workspace only. When the task is complete, respond with a concise final summary and make no further tool calls."""

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a shell command as root inside the task sandbox and return its exit code, stdout, and stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute. Each call starts in /app. Use non-interactive commands.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Optional command timeout in seconds (1-600).",
                    "minimum": 1,
                    "maximum": 600,
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class AgentResult:
    exit_code: int
    turns: int
    timed_out: bool = False


def _bounded_text(value: str, *, max_bytes: int) -> str:
    raw = value.encode("utf-8", errors="replace")
    if len(raw) <= max_bytes:
        return value
    half = max(1, (max_bytes - 96) // 2)
    head = raw[:half].decode("utf-8", errors="replace")
    tail = raw[-half:].decode("utf-8", errors="replace")
    return f"{head}\n...[tool output truncated: {len(raw)} bytes total]...\n{tail}"


def _arguments(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    raw = function.get("arguments", {})
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"command": raw}
        return parsed if isinstance(parsed, dict) else {"command": str(parsed)}
    return {}


class TerminalToolAgent:
    def __init__(self) -> None:
        self.max_turns = int(os.environ.get("TERMINAL_RL_MAX_TURNS", "100"))
        self.default_command_timeout = int(os.environ.get("TERMINAL_RL_COMMAND_TIMEOUT_SEC", "120"))
        self.max_tool_output_bytes = int(os.environ.get("TERMINAL_RL_MAX_TOOL_OUTPUT_BYTES", "32768"))
        self.reasoning_effort = os.environ.get("TERMINAL_RL_REASONING_EFFORT", "medium").strip().lower()
        if self.max_turns <= 0:
            raise ValueError("TERMINAL_RL_MAX_TURNS must be positive")
        if not 1 <= self.default_command_timeout <= 600:
            raise ValueError("TERMINAL_RL_COMMAND_TIMEOUT_SEC must be in [1, 600]")
        if self.max_tool_output_bytes < 1024:
            raise ValueError("TERMINAL_RL_MAX_TOOL_OUTPUT_BYTES must be at least 1024")
        if self.reasoning_effort not in {"none", "medium", "high"}:
            raise ValueError("TERMINAL_RL_REASONING_EFFORT must be none, medium, or high")

    async def _run_bash(self, sandbox: Sandbox, call: dict[str, Any]) -> str:
        arguments = _arguments(call)
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return "tool_error: bash requires a non-empty string field named 'command'"
        raw_timeout = arguments.get("timeout", self.default_command_timeout)
        try:
            timeout = max(1, min(600, int(raw_timeout)))
        except (TypeError, ValueError):
            timeout = self.default_command_timeout
        try:
            exit_code, stdout, stderr = await sandbox.exec(
                f"cd /app\n{command}",
                user="root",
                timeout=timeout,
                check=False,
                idempotent=False,
            )
        except Exception as exc:  # noqa: BLE001 - tool failures are observations, not rollout crashes
            return f"tool_error: {type(exc).__name__}: {exc}"
        rendered = f"exit_code={exit_code}\nstdout:\n{stdout or ''}\nstderr:\n{stderr or ''}"
        return _bounded_text(rendered, max_bytes=self.max_tool_output_bytes)

    async def _tool_result(self, sandbox: Sandbox, call: dict[str, Any]) -> str:
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(function.get("name") or call.get("name") or "")
        if name != "bash":
            return f"tool_error: unknown tool {name!r}; available tool: bash"
        return await self._run_bash(sandbox, call)

    async def run(
        self,
        sandbox: Sandbox,
        *,
        session_id: str,
        adapter_url: str,
        time_budget_sec: int,
        prompt: str,
    ) -> AgentResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        deadline = time.monotonic() + time_budget_sec
        async with aiohttp.ClientSession(trust_env=False) as client:
            for turn in range(1, self.max_turns + 1):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return AgentResult(exit_code=-1, turns=turn - 1, timed_out=True)
                body = {
                    "model": "slime-actor",
                    "messages": messages,
                    "tools": [BASH_TOOL],
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                    "reasoning_effort": self.reasoning_effort,
                }
                try:
                    async with asyncio.timeout(remaining), client.post(
                        f"{adapter_url}/v1/chat/completions",
                        headers={"Authorization": f"Bearer {session_id}"},
                        json=body,
                    ) as response:
                        raw = await response.text()
                        if response.status >= 400:
                            raise RuntimeError(f"adapter returned HTTP {response.status}: {raw[:500]}")
                        payload = json.loads(raw)
                except TimeoutError:
                    return AgentResult(exit_code=-1, turns=turn - 1, timed_out=True)

                choices = payload.get("choices") if isinstance(payload, dict) else None
                if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                    raise RuntimeError("adapter response contains no choice")
                message = choices[0].get("message")
                if not isinstance(message, dict):
                    raise RuntimeError("adapter response contains no assistant message")
                assistant = {key: value for key, value in message.items() if key in {"role", "content", "reasoning_content", "tool_calls"}}
                assistant["role"] = "assistant"
                messages.append(assistant)

                tool_calls = message.get("tool_calls")
                if not isinstance(tool_calls, list) or not tool_calls:
                    return AgentResult(exit_code=0, turns=turn)
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    result = await self._tool_result(sandbox, call)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(call.get("id") or "tool-call"),
                            "content": result,
                        }
                    )
        return AgentResult(exit_code=-2, turns=self.max_turns)
