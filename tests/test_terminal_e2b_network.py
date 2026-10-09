"""E2B 路由覆盖 Slime 异步 SDK 与独立 NexTask bridge 的全部客户端。"""

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from examples.terminal_agent_rl import bridge, e2b_network as network, runtime
from examples.terminal_agent_rl.sandbox import NextaskE2BSandbox

NUM_GPUS = 0
PROXY = "http://10.51.6.27:5023"


@pytest.mark.parametrize("value", [None, "", PROXY])
def test_bridge_env_never_inherits_current_machine_proxy(monkeypatch, value):
    monkeypatch.setenv(network.PROXY_VARIABLE, value or "")
    env = {"HTTPS_PROXY": "http://127.0.0.1:7890", "all_proxy": "socks5://localhost:7890", "NO_PROXY": "e2b.example", "PATH": "/bin"}
    cleaned = network.e2b_subprocess_env(env)
    assert cleaned["PATH"] == "/bin"
    assert "all_proxy" not in cleaned
    assert cleaned[network.PROXY_VARIABLE] == (value or "")
    assert cleaned["NO_PROXY"] == ("127.0.0.1,localhost" if value else "*")
    assert cleaned.get("HTTPS_PROXY") == (value or None)
    assert env["HTTPS_PROXY"] == "http://127.0.0.1:7890"


@pytest.mark.parametrize("value", ["https://proxy:5023", "http://user:password@proxy:5023", "http://proxy", "http://proxy:noport", "http://proxy:5023/path"])
def test_invalid_e2b_proxy_is_rejected_without_printing_value(monkeypatch, value):
    monkeypatch.setenv(network.PROXY_VARIABLE, value)
    with pytest.raises(ValueError, match="必须是无凭据") as exc:
        network.resolve_e2b_proxy()
    assert value not in str(exc.value)


@pytest.mark.parametrize("proxy", [None, PROXY])
def test_bridge_sdk_defaults_cover_all_new_connections(monkeypatch, proxy):
    # 只修改假的SDK类；不会污染其他测试或真实SDK的构造器。
    class Config:
        def __init__(self, api_key=None, proxy=None):
            self.api_key = api_key
            self.proxy = proxy

    monkeypatch.setitem(sys.modules, "e2b.connection_config", SimpleNamespace(ConnectionConfig=Config))
    monkeypatch.setattr(network.os, "environ", {network.PROXY_VARIABLE: proxy or "", "HTTPS_PROXY": "http://localhost:7890"})
    network.configure_bridge_network()
    first = Config.__init__
    network.configure_bridge_network()
    assert Config.__init__ is first
    assert Config(api_key="key").proxy == proxy
    assert Config("key", None).proxy == proxy
    assert Config("key", proxy).api_key == "key"
    with pytest.raises(ValueError, match="不一致"):
        Config(proxy="http://other:9000")
    assert network.os.environ.get("HTTPS_PROXY") == proxy


def test_async_sandbox_receives_explicit_e2b_proxy(tmp_path, monkeypatch):
    config = tmp_path / "e2b.json"
    config.write_text(json.dumps({"api_key": "key", "api_url": "https://e2b.example", "domain": "e2b.example"}))
    config.chmod(0o600)
    monkeypatch.setattr(network.os, "environ", {network.PROXY_VARIABLE: PROXY, "TERMINAL_RL_E2B_CONFIG_FILE": str(config)})
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(sandbox_id="sandbox")

    monkeypatch.setitem(sys.modules, "e2b", SimpleNamespace(AsyncSandbox=SimpleNamespace(create=create)))
    sandbox = NextaskE2BSandbox("template", timeout=300)
    asyncio.run(sandbox.__aenter__())
    assert calls[0]["proxy"] == PROXY
    assert calls[0]["template"] == "template"
    assert calls[0]["api_key"] == "key"
    assert sandbox.sandbox_id == "sandbox"
    child = runtime.CONFIG.subprocess_env()
    assert child[network.PROXY_VARIABLE] == PROXY
    assert child["HTTPS_PROXY"] == PROXY
    assert child["NEXTASK_BENCH_E2B_API_KEY"] == "key"


def test_bridge_installs_route_before_cleanup_but_not_export(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(network, "configure_bridge_network", lambda: calls.append("network"))
    monkeypatch.setattr(bridge, "cleanup_candidate", lambda path: calls.append("cleanup") or {"ok": True})
    monkeypatch.setattr(bridge, "export_collection", lambda *args: calls.append("export") or {"ok": True})
    assert bridge.main(["cleanup", "--candidate-dir", str(tmp_path)]) == 0
    assert calls == ["network", "cleanup"]
    calls.clear()
    assert bridge.main(["export", "--collection", str(tmp_path), "--output", str(tmp_path / "data.jsonl")]) == 0
    assert calls == ["export"]
