"""代理诊断必须显式路由且在失败时清理已经创建的沙箱。"""

import asyncio
import sys
from types import SimpleNamespace

import pytest

from examples.terminal_agent_rl import smoke_e2b_proxy as smoke

NUM_GPUS = 0


@pytest.mark.parametrize("failure", [None, "files", "command", "cleanup"])
def test_explicit_proxy_and_cleanup(monkeypatch, failure):
    proxy = "http://10.51.6.27:5023"
    calls = []
    contents = {}

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["proxy"] == proxy and kwargs["trust_env"] is False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url):
            return SimpleNamespace(status_code=404)

    class Files:
        async def write(self, path, content):
            contents[path] = content

        async def read(self, path):
            return "wrong" if failure == "files" else contents[path]

    class Commands:
        async def run(self, cmd, **kwargs):
            if failure == "command":
                raise RuntimeError("command failed")
            return SimpleNamespace(exit_code=0, stdout=next(iter(contents.values())))

    class Sandbox:
        sandbox_id = "test-sandbox"
        connection_config = SimpleNamespace(proxy=proxy)
        files = Files()
        commands = Commands()

        @classmethod
        async def create(cls, **kwargs):
            assert kwargs["proxy"] == proxy
            assert kwargs["api_key"] == "secret"
            calls.append("create")
            return cls()

        async def kill(self):
            calls.append("kill")
            if failure == "cleanup":
                raise RuntimeError("cleanup failed")
            return False  # 已经不存在也是确认清理。

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(AsyncClient=Client))
    monkeypatch.setitem(sys.modules, "e2b", SimpleNamespace(AsyncSandbox=Sandbox))
    monkeypatch.setattr(smoke.importlib.metadata, "version", lambda name: "2.21.0")
    monkeypatch.setattr(smoke, "resolve_e2b_connection", lambda: ("secret", "https://e2b.example", "e2b.example"))
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setenv("NO_PROXY", "localhost")

    result = asyncio.run(smoke.run(proxy, "template"))

    assert calls == ["create", "kill"]
    assert result["ok"] is (failure is None)
    assert result["live_sandbox_cleanup_confirmed"] is (failure != "cleanup")
    assert "secret" not in str(result)
    assert "HTTPS_PROXY" not in smoke.os.environ
    assert smoke.os.environ["NO_PROXY"] == "*"
