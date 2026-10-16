"""CPU coverage for the E2B pull relay used by Claude Code rollouts."""

from __future__ import annotations

import asyncio
import base64
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from examples.terminal_agent_rl.cc_relay import E2BPullRelay, RELAY_ROOT


class MemorySandbox:
    def __init__(self, files: dict[str, str] | None = None) -> None:
        self.files = dict(files or {})
        self.commands: list[str] = []

    async def read_file(self, path: str, *, user: str = "root") -> str:
        return self.files[path]

    async def write_file(self, path: str, content, *, user: str = "root") -> None:
        self.files[path] = str(content)

    async def exec(self, command: str, **_kwargs):
        self.commands.append(command)
        if command.startswith("rm -f "):
            self.files.pop(command.removeprefix("rm -f "), None)
        elif command.startswith("mv "):
            _, source, destination = command.split()
            self.files[destination] = self.files.pop(source)
        return 0, "", ""


def test_pull_relay_forwards_request_and_publishes_response() -> None:
    async def run_case() -> None:
        captured = {}

        async def messages(request: web.Request) -> web.Response:
            captured["headers"] = dict(request.headers)
            captured["body"] = await request.read()
            return web.Response(
                status=201,
                body=b'{"type":"message","content":[]}',
                headers={"content-type": "application/json", "cache-control": "no-store"},
            )

        app = web.Application()
        app.router.add_post("/v1/messages", messages)
        async with TestServer(app) as upstream:
            request_id = "request-1"
            request_path = f"{RELAY_ROOT}/requests/{request_id}.json"
            sandbox = MemorySandbox(
                {
                    request_path: json.dumps(
                        {
                            "id": request_id,
                            "method": "POST",
                            "path": "/v1/messages",
                            "headers": {"content-type": "application/json", "x-api-key": "session"},
                            "body_b64": base64.b64encode(b'{"model":"slime-actor"}').decode(),
                        }
                    )
                }
            )
            relay = E2BPullRelay(sandbox, str(upstream.make_url("/")).rstrip("/"))
            await relay._forward(request_id)

        response_path = f"{RELAY_ROOT}/responses/{request_id}.json"
        response = json.loads(sandbox.files[response_path])
        assert request_path not in sandbox.files
        assert captured["body"] == b'{"model":"slime-actor"}'
        assert {key.lower(): value for key, value in captured["headers"].items()}["x-api-key"] == "session"
        assert response["status"] == 201
        assert json.loads(base64.b64decode(response["body_b64"])) == {"type": "message", "content": []}
        assert response["headers"] == {"Content-Type": "application/json", "Cache-Control": "no-store"}

    asyncio.run(run_case())


def test_pull_relay_returns_sanitized_502_for_invalid_request() -> None:
    async def run_case() -> None:
        request_id = "broken"
        sandbox = MemorySandbox({f"{RELAY_ROOT}/requests/{request_id}.json": "not-json"})
        relay = E2BPullRelay(sandbox, "http://127.0.0.1:1")
        await relay._forward(request_id)
        response = json.loads(sandbox.files[f"{RELAY_ROOT}/responses/{request_id}.json"])
        assert response["status"] == 502
        assert json.loads(base64.b64decode(response["body_b64"])) == {
            "type": "error",
            "error": {"type": "JSONDecodeError"},
        }

    asyncio.run(run_case())


def test_pull_relay_rejects_non_positive_poll_interval() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        E2BPullRelay(MemorySandbox(), "http://127.0.0.1:1", poll_interval_sec=0)
