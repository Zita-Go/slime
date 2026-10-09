"""E2B 专用代理配置；不依赖当前机器的通用代理。"""

from __future__ import annotations

import functools
import inspect
import os
from urllib.parse import urlsplit

PROXY_VARIABLE = "TERMINAL_RL_E2B_PROXY"
_GENERIC_PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def resolve_e2b_proxy() -> str | None:
    value = os.environ.get(PROXY_VARIABLE, "").strip()
    if not value:
        return None
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme == "http"
            and parsed.hostname
            and parsed.port
            and not parsed.username
            and not parsed.password
            and parsed.path in {"", "/"}
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(f"{PROXY_VARIABLE} 必须是无凭据的 http://host:port，或空值表示直连")
    return value


def e2b_subprocess_env(env: dict[str, str]) -> dict[str, str]:
    """只改 bridge 子进程环境，不改变 Ray/SGLang 或当前 shell 的网络。"""
    result = {key: value for key, value in env.items() if key not in _GENERIC_PROXY_VARIABLES}
    proxy = resolve_e2b_proxy()
    result[PROXY_VARIABLE] = proxy or ""
    if proxy:
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            result[name] = proxy
        bypass = "127.0.0.1,localhost"
    else:
        bypass = "*"
    result["NO_PROXY"] = result["no_proxy"] = bypass
    return result


def configure_bridge_network() -> None:
    """为固定 NexTask 的所有 SDK 操作提供统一路由，仅在 bridge 内调用。

    NexTask 内部的创建、恢复和删除快照不接收 proxy 参数；在进程入口适配
    SDK 配置构造器，保持 SDK 公共方法、冻结 runtime 与 verifier 身份不变。
    """
    from e2b.connection_config import ConnectionConfig

    clean = e2b_subprocess_env(dict(os.environ))
    for name in _GENERIC_PROXY_VARIABLES:
        os.environ.pop(name, None)
    for name in (*_GENERIC_PROXY_VARIABLES, "NO_PROXY", "no_proxy", PROXY_VARIABLE):
        if name in clean:
            os.environ[name] = clean[name]
    original = ConnectionConfig.__init__
    if getattr(original, "_terminal_rl_network", False):
        return
    signature = inspect.signature(original)
    if "proxy" not in signature.parameters:
        raise RuntimeError("E2B SDK ConnectionConfig 缺少 proxy 参数")

    @functools.wraps(original)
    def configured(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        proxy = resolve_e2b_proxy()
        explicit = bound.arguments.get("proxy")
        if explicit is not None and explicit != proxy:
            raise ValueError("E2B SDK 显式代理与 TERMINAL_RL_E2B_PROXY 不一致")
        bound.arguments["proxy"] = proxy
        return original(*bound.args, **bound.kwargs)

    configured._terminal_rl_network = True
    ConnectionConfig.__init__ = configured
