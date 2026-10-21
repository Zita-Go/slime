"""CPU coverage for the E2B pull relay used by Claude Code rollouts."""

from __future__ import annotations

import asyncio
import ast
import base64
from collections import Counter
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import urllib.request

import httpx
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from examples.terminal_agent_rl.cc_relay import E2BPullRelay, E2BRelayError, RELAY_ROOT, SANDBOX_PROXY
from slime.agent.sandbox import E2BSandbox

NUM_GPUS = 0


class LocalRPCSandbox(E2BSandbox):
    """真实shell与生产RPC重试，仅注入确认丢失；不需要E2B服务或GPU。"""

    rpc_backoff_base_sec = 0
    rpc_backoff_cap_sec = 0

    def __init__(self, root: Path, *, lose_publish_ack: bool = False, lose_payload: bool = False):
        super().__init__("local-diagnostic", rpc_retries=2)
        self.root = root
        self.lose_publish_ack = lose_publish_ack
        self.lose_payload = lose_payload
        self.publish_attempts = Counter()

    def local_path(self, path: str) -> Path:
        return self.root / Path(path).relative_to(RELAY_ROOT)

    async def read_file(self, path, **_kwargs):
        return self.local_path(path).read_text()

    async def write_file(self, path, content, **_kwargs):
        local = self.local_path(path)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(content)
        if self.lose_payload and "/responses/." in path:
            local.unlink()

    async def _reset_conn_pool(self):
        # 本地替身没有连接池；生产重试状态机保持不变。
        pass

    async def exec(self, command, **kwargs):
        async def execute():
            process = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                command.replace(RELAY_ROOT, str(self.root)),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode and kwargs.get("check"):
                raise RuntimeError(stderr.decode())
            if command.startswith("if test -f "):
                response = self.local_path(command.split()[3].rstrip(";"))
                self.publish_attempts[response.stem] += 1
                if self.lose_publish_ack and self.publish_attempts[response.stem] == 1:
                    deadline = asyncio.get_running_loop().time() + 3
                    while response.exists() and asyncio.get_running_loop().time() < deadline:
                        await asyncio.sleep(0.005)
                    assert not response.exists(), "真实代理尚未消费响应"
                    raise httpx.ReadError("injected: publish completed but RPC ACK lost")
            return process.returncode, stdout.decode(), stderr.decode()

        return await self._rpc_retry("exec", execute, idempotent=kwargs.get("idempotent", True))


@pytest.fixture
def local_proxy(tmp_path):
    # 执行生产代理的类/函数，跳过顶层serve_forever，以临时目录和空闲端口运行。
    tree = ast.parse(SANDBOX_PROXY)
    tree.body = [node for node in tree.body if not isinstance(node, ast.Expr)]
    namespace = {}
    exec(compile(tree, "<sandbox-proxy>", "exec"), namespace)
    namespace["ROOT"] = str(tmp_path)
    for name in ("REQUESTS", "RESPONSES", "CONSUMED"):
        namespace[name] = str(tmp_path / name.lower())
        Path(namespace[name]).mkdir()
    real_os = namespace["os"]

    def checked_unlink(path):
        # 不允许在写入消费凭据前先删除响应，否则仍有无凭据的竞态窗口。
        marker = tmp_path / "consumed" / Path(path).name
        assert json.loads(marker.read_text()) == {"id": Path(path).stem}
        return real_os.unlink(path)

    namespace["os"] = SimpleNamespace(**vars(real_os))
    namespace["os"].unlink = checked_unlink
    server = namespace["ThreadingHTTPServer"](("127.0.0.1", 0), namespace["Handler"], bind_and_activate=False)
    server.request_queue_size = 32
    server.server_bind()
    server.server_activate()
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive()


@pytest.mark.parametrize("concurrency", [1, 8])
def test_publish_retry_after_real_proxy_consumption(local_proxy, tmp_path, concurrency):
    async def run_case():
        upstream_calls = []

        async def messages(request):
            body = await request.json()
            upstream_calls.append(body["index"])
            return web.json_response({"index": body["index"], "text": "DELIVERED"})

        def request(index):
            message = urllib.request.Request(
                local_proxy + "/v1/messages",
                data=json.dumps({"index": index}).encode(),
                headers={"content-type": "application/json"},
            )
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(message, timeout=5) as response:
                assert response.status == 200
                return json.load(response)

        app = web.Application()
        app.router.add_post("/v1/messages", messages)
        sandbox = LocalRPCSandbox(tmp_path, lose_publish_ack=True)
        async with TestServer(app, shutdown_timeout=0.1) as upstream:
            relay = E2BPullRelay(sandbox, str(upstream.make_url("/")).rstrip("/"))
            clients = [asyncio.create_task(asyncio.to_thread(request, i)) for i in range(concurrency)]
            try:
                deadline = asyncio.get_running_loop().time() + 3
                while len(list((tmp_path / "requests").glob("*.json"))) < concurrency:
                    assert asyncio.get_running_loop().time() < deadline
                    await asyncio.sleep(0.005)
                request_ids = [p.stem for p in (tmp_path / "requests").glob("*.json")]
                await asyncio.gather(*(relay._forward(sid) for sid in request_ids))
                received = await asyncio.gather(*clients)
            finally:
                await asyncio.gather(*clients, return_exceptions=True)
        assert sorted(upstream_calls) == list(range(concurrency)), "重试不得重新请求模型"
        assert sorted(received, key=lambda r: r["index"]) == [{"index": i, "text": "DELIVERED"} for i in range(concurrency)]
        assert sandbox.publish_attempts == Counter({sid: 2 for sid in request_ids})
        assert not list((tmp_path / "responses").iterdir())
        for sid in request_ids:
            assert json.loads((tmp_path / "consumed" / f"{sid}.json").read_text()) == {"id": sid}

    asyncio.run(asyncio.wait_for(run_case(), timeout=15))


def test_publish_missing_payload_is_not_silently_accepted(tmp_path):
    async def run_case():
        sandbox = LocalRPCSandbox(tmp_path, lose_payload=True)
        await sandbox.write_file(f"{RELAY_ROOT}/requests/missing.json", "not-json")
        (tmp_path / "consumed").mkdir()
        (tmp_path / "consumed/other-request.json").write_text("{}")
        relay = E2BPullRelay(sandbox, "http://127.0.0.1:1")
        with pytest.raises(RuntimeError, match="cannot stat"):
            await relay._forward("missing")

    asyncio.run(run_case())


class MemorySandbox:
    def __init__(self, files: dict[str, str] | None = None) -> None:
        self.files = dict(files or {})
        self.commands: list[str] = []
        self.exec_options: list[dict] = []

    async def read_file(self, path: str, *, user: str = "root") -> str:
        return self.files[path]

    async def write_file(self, path: str, content, *, user: str = "root") -> None:
        self.files[path] = str(content)

    async def exec(self, command: str, **_kwargs):
        self.commands.append(command)
        self.exec_options.append(dict(_kwargs))
        if command.startswith("rm -f "):
            self.files.pop(command.removeprefix("rm -f "), None)
        elif command.startswith("if test -f "):
            parts = command.split()
            destination = parts[3].rstrip(";")
            consumed = parts[7].rstrip(";")
            source = parts[11].rstrip(";")
            assert parts[14] == source and parts[15].rstrip(";") == destination
            if destination in self.files or consumed in self.files:
                self.files.pop(source, None)
            else:
                self.files[destination] = self.files.pop(source)
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
        publish_command = sandbox.commands[-1]
        assert sandbox.exec_options[-1]["idempotent"] is True
        assert "if test -f" in publish_command
        await sandbox.exec(publish_command, idempotent=True)
        assert response_path in sandbox.files

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


def test_pull_relay_surfaces_background_publication_failure() -> None:
    class FailingPublishSandbox(MemorySandbox):
        async def write_file(self, path: str, content, *, user: str = "root") -> None:
            if "/responses/." in path:
                raise RuntimeError("response store unavailable")
            await super().write_file(path, content, user=user)

    async def run_case() -> None:
        request_id = "broken-publication"
        sandbox = FailingPublishSandbox({f"{RELAY_ROOT}/requests/{request_id}.json": "not-json"})
        relay = E2BPullRelay(sandbox, "http://127.0.0.1:1")
        task = asyncio.create_task(relay._forward(request_id))
        relay._forward_tasks.add(task)
        task.add_done_callback(relay._forward_done)
        with pytest.raises(E2BRelayError, match="pull relay failed"):
            await asyncio.wait_for(relay.wait_failure(), timeout=1)
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run_case())


def test_sandbox_proxy_forwards_count_tokens_instead_of_fabricating_zero() -> None:
    assert 'if self.path.startswith("/v1/messages/count_tokens")' not in SANDBOX_PROXY
    assert 'body = b\'{"input_tokens":0}\'' not in SANDBOX_PROXY


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
