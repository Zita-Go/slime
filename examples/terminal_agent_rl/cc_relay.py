"""通过E2B文件RPC桥接sandbox内Claude Code与host侧AnthropicAdapter。"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import secrets
from pathlib import Path

import aiohttp

from slime.agent.sandbox import Sandbox

logger = logging.getLogger(__name__)

RELAY_ROOT = "/tmp/slime-cc-relay"
RELAY_PORT = 18080
SANDBOX_PROXY = r'''#!/usr/bin/env python3
import base64
import json
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = "/tmp/slime-cc-relay"
REQUESTS = ROOT + "/requests"
RESPONSES = ROOT + "/responses"
CONSUMED = ROOT + "/consumed"
TIMEOUT = int(os.environ.get("SLIME_CC_RELAY_REQUEST_TIMEOUT_SEC", "1800"))


def atomic_json(path, value):
    temporary = path + ".tmp-" + uuid.uuid4().hex
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        return

    def do_GET(self):
        if self.path == "/healthz":
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self):
        request_id = uuid.uuid4().hex
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() in {"authorization", "x-api-key", "anthropic-version", "content-type"}
        }
        atomic_json(
            REQUESTS + "/" + request_id + ".json",
            {
                "id": request_id,
                "method": "POST",
                "path": self.path,
                "headers": headers,
                "body_b64": base64.b64encode(body).decode("ascii"),
            },
        )
        response_path = RESPONSES + "/" + request_id + ".json"
        deadline = time.monotonic() + TIMEOUT
        while not os.path.exists(response_path) and time.monotonic() < deadline:
            time.sleep(0.05)
        if not os.path.exists(response_path):
            self.send_error(504, "host relay timeout")
            return
        with open(response_path, encoding="utf-8") as stream:
            response = json.load(stream)
        response_body = base64.b64decode(response["body_b64"])
        # 先留下持久的小凭据再删除响应，RPC丢确认后的发布重试仍能识别已消费。
        # 这里只确认代理已接管，不表示HTTP客户端已经接收完成。
        atomic_json(CONSUMED + "/" + request_id + ".json", {"id": request_id})
        os.unlink(response_path)
        self.send_response(int(response["status"]))
        for key, value in response.get("headers", {}).items():
            if key.lower() in {"content-type", "cache-control"}:
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)


os.makedirs(REQUESTS, exist_ok=True)
os.makedirs(RESPONSES, exist_ok=True)
os.makedirs(CONSUMED, exist_ok=True)
ThreadingHTTPServer(("127.0.0.1", 18080), Handler).serve_forever()
'''


class E2BRelayError(RuntimeError):
    """The sandbox pull relay can no longer deliver a model response."""


class E2BPullRelay:
    """Host主动轮询E2B请求文件，避免sandbox需要访问集群私网。"""

    def __init__(self, sandbox: Sandbox, upstream_url: str, *, poll_interval_sec: float = 0.1) -> None:
        if poll_interval_sec <= 0:
            raise ValueError("poll_interval_sec must be positive")
        self.sandbox = sandbox
        self.upstream_url = upstream_url.rstrip("/")
        self.poll_interval_sec = poll_interval_sec
        self.adapter_url = f"http://127.0.0.1:{RELAY_PORT}"
        self._poll_task: asyncio.Task | None = None
        self._forward_tasks: set[asyncio.Task] = set()
        self._seen: set[str] = set()
        self._failure: BaseException | None = None
        self._failure_event = asyncio.Event()

    async def __aenter__(self) -> E2BPullRelay:
        await self.sandbox.write_file(f"{RELAY_ROOT}/proxy.py", SANDBOX_PROXY)
        await self.sandbox.exec(
            f"mkdir -p {RELAY_ROOT}/requests {RELAY_ROOT}/responses {RELAY_ROOT}/consumed; "
            f"setsid python3 {RELAY_ROOT}/proxy.py > {RELAY_ROOT}/proxy.log 2>&1 & echo $! > {RELAY_ROOT}/proxy.pid",
            user="root",
            timeout=30,
            check=True,
            idempotent=False,
        )
        for _ in range(50):
            code, output, _ = await self.sandbox.exec(
                f"python3 -c \"import urllib.request; print(urllib.request.urlopen('{self.adapter_url}/healthz', timeout=2).read().decode())\"",
                timeout=5,
                check=False,
            )
            if code == 0 and output.strip() == "ok":
                break
            await asyncio.sleep(0.1)
        else:
            raise RuntimeError("sandbox Claude Code relay did not become ready")
        self._poll_task = asyncio.create_task(self._poll_loop(), name="e2b-cc-pull-relay")
        self._poll_task.add_done_callback(self._poll_done)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._poll_task is not None:
            self._poll_task.cancel()
            await asyncio.gather(self._poll_task, return_exceptions=True)
        for task in list(self._forward_tasks):
            task.cancel()
        if self._forward_tasks:
            await asyncio.gather(*self._forward_tasks, return_exceptions=True)
        await self.sandbox.exec(
            f"test -f {RELAY_ROOT}/proxy.pid && kill $(cat {RELAY_ROOT}/proxy.pid) 2>/dev/null || true",
            timeout=15,
            check=False,
        )

    async def wait_failure(self) -> None:
        """Raise promptly when a background poll/forward task fails permanently."""

        await self._failure_event.wait()
        assert self._failure is not None
        raise E2BRelayError("Claude Code E2B pull relay failed") from self._failure

    def _record_failure(self, error: BaseException) -> None:
        if self._failure is None:
            self._failure = error
            self._failure_event.set()

    def _poll_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        try:
            task.result()
        except Exception as exc:  # noqa: BLE001 - wake the rollout waiting on relay health
            self._record_failure(exc)
            logger.exception("Claude Code relay polling failed")
        else:
            error = RuntimeError("Claude Code relay polling stopped unexpectedly")
            self._record_failure(error)
            logger.error("%s", error)

    async def _poll_loop(self) -> None:
        # Wait inside the sandbox for at most 30 seconds per RPC.  Polling from
        # the host every 100ms would create hundreds of idle E2B control-plane
        # calls per minute for every concurrent rollout.
        attempts = max(1, round(30 / self.poll_interval_sec))
        interval = f"{self.poll_interval_sec:.3f}"
        while True:
            code, output, _ = await self.sandbox.exec(
                f"for _ in $(seq 1 {attempts}); do "
                f"files=$(find {RELAY_ROOT}/requests -maxdepth 1 -type f -name '*.json' -printf '%f\\n'); "
                'if [ -n "$files" ]; then printf \'%s\\n\' "$files"; exit 0; fi; '
                f"sleep {interval}; done",
                timeout=40,
                check=False,
            )
            if code == 0:
                for filename in output.splitlines():
                    request_id = Path(filename).stem
                    if request_id in self._seen:
                        continue
                    self._seen.add(request_id)
                    task = asyncio.create_task(self._forward(request_id), name=f"cc-relay-{request_id}")
                    self._forward_tasks.add(task)
                    task.add_done_callback(self._forward_done)
            if code != 0:
                await asyncio.sleep(self.poll_interval_sec)

    def _forward_done(self, task: asyncio.Task) -> None:
        self._forward_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception as exc:  # noqa: BLE001 - wake the rollout waiting on relay health
            self._record_failure(exc)
            logger.exception("Claude Code relay response publication failed")

    async def _forward(self, request_id: str) -> None:
        request_path = f"{RELAY_ROOT}/requests/{request_id}.json"
        try:
            request = json.loads(await self.sandbox.read_file(request_path))
            # Stop the next long poll from rediscovering a request while its
            # upstream generation is still running.  The in-memory payload is
            # sufficient from this point onward.
            await self.sandbox.exec(f"rm -f {request_path}", timeout=15, check=True)
            body = base64.b64decode(request["body_b64"])
            timeout = aiohttp.ClientTimeout(total=None, sock_read=1200)
            async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as client:
                async with client.request(
                    request["method"],
                    self.upstream_url + request["path"],
                    headers=request.get("headers") or {},
                    data=body,
                ) as response:
                    response_body = await response.read()
                    result = {
                        "status": response.status,
                        "headers": {
                            key: value
                            for key, value in response.headers.items()
                            if key.lower() in {"content-type", "cache-control"}
                        },
                        "body_b64": base64.b64encode(response_body).decode("ascii"),
                    }
        except Exception as exc:  # noqa: BLE001 - failure is returned to the sandbox client
            result = {
                "status": 502,
                "headers": {"content-type": "application/json"},
                "body_b64": base64.b64encode(
                    json.dumps({"type": "error", "error": {"type": type(exc).__name__}}).encode()
                ).decode("ascii"),
            }
        response_path = f"{RELAY_ROOT}/responses/{request_id}.json"
        consumed_path = f"{RELAY_ROOT}/consumed/{request_id}.json"
        temporary = f"{RELAY_ROOT}/responses/.{request_id}-{secrets.token_hex(4)}.json"
        await self.sandbox.write_file(temporary, json.dumps(result, separators=(",", ":")))
        # 正式响应会被消费者删除，不能单独充当发布完成凭据。
        # consumed记录保留至sandbox销毁，覆盖“已消费后才重试”的丢确认窗口。
        # 两个凭据都没有时仍须成功mv，不能吞掉真实的数据丢失。
        await self.sandbox.exec(
            f"if test -f {response_path} || test -f {consumed_path}; then rm -f {temporary}; else mv {temporary} {response_path}; fi",
            timeout=15,
            check=True,
            idempotent=True,
        )
