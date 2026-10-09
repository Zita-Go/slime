"""在 GPU Pod 内验证显式 HTTP 代理及 E2B 创建、文件、命令、清理链路。"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import secrets
import socket
from pathlib import Path
from urllib.parse import urlsplit

from .e2b_config import resolve_e2b_connection


async def run(proxy: str, template: str, *, require_gpu: bool = False) -> dict:
    # 本进程只使用传入的代理，不继承当前机器的代理设置。
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(name, None)
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "*"

    import httpx
    from e2b import AsyncSandbox

    parsed = urlsplit(proxy)
    if parsed.scheme != "http" or not parsed.hostname or not parsed.port or parsed.username or parsed.password:
        raise ValueError("proxy 必须是无凭据的 http://host:port")
    version = importlib.metadata.version("e2b")
    if version != "2.21.0":
        raise RuntimeError(f"需要固定 SDK e2b 2.21.0，实际为 {version}")
    api_key, api_url, domain = resolve_e2b_connection()
    gpu_count = None
    if require_gpu:
        import torch

        gpu_count = torch.cuda.device_count()
        if gpu_count < 1:
            raise RuntimeError("本次验证必须在至少一张 GPU 的 Pod 内执行")

    result = {
        "ok": False,
        "host": socket.gethostname(),
        "gpu_count": gpu_count,
        "proxy": proxy,
        "api_host": urlsplit(api_url or "https://api.e2b.dev").hostname,
        "sdk_version": version,
        "template_identity": template,
        "api_reachable": False,
        "sandbox_created": False,
        "file_roundtrip": False,
        "command_executed": False,
        "live_sandbox_cleanup_confirmed": False,
        "stage": "api_https",
    }
    sandbox = None
    try:
        async with asyncio.timeout(180):
            async with httpx.AsyncClient(proxy=proxy, trust_env=False, timeout=30, http2=True) as client:
                response = await client.get((api_url or "https://api.e2b.dev").rstrip("/") + "/")
                result["api_http_status"] = response.status_code
                # 根路径通常为404；收到HTTP响应即证明CONNECT与TLS链路成立。
                result["api_reachable"] = True
            result["stage"] = "create"
            options = {"api_key": api_key, "proxy": proxy, "request_timeout": 30}
            if api_url:
                options["api_url"] = api_url
            if domain:
                options["domain"] = domain
            sandbox = await AsyncSandbox.create(template=template, timeout=300, **options)
            result["sandbox_id"] = sandbox.sandbox_id
            result["sandbox_created"] = True
            if sandbox.connection_config.proxy != proxy:
                raise RuntimeError("返回的 sandbox 未保留指定代理")
            marker = "E2B_PROXY_OK_" + secrets.token_hex(8)
            path = "/tmp/terminal-rl-proxy-probe.txt"
            result["stage"] = "files"
            await sandbox.files.write(path, marker)
            if await sandbox.files.read(path) != marker:
                raise RuntimeError("E2B文件读写不一致")
            result["file_roundtrip"] = True
            result["stage"] = "command"
            command = await sandbox.commands.run(f"cat {path}", timeout=30)
            if command.exit_code != 0 or command.stdout.strip() != marker:
                raise RuntimeError("E2B命令执行结果不一致")
            result["command_executed"] = True
    except Exception as exc:  # noqa: BLE001 - 诊断回执不包含可能带凭据的完整异常文本
        result["error_type"] = type(exc).__name__
    finally:
        if sandbox is not None:
            try:
                async with asyncio.timeout(45):
                    await sandbox.kill()
                result["live_sandbox_cleanup_confirmed"] = True
            except Exception as exc:  # noqa: BLE001
                result["cleanup_error_type"] = type(exc).__name__
    result["ok"] = all(
        result[key]
        for key in ("api_reachable", "sandbox_created", "file_roundtrip", "command_executed", "live_sandbox_cleanup_confirmed")
    ) and "error_type" not in result
    if result["ok"]:
        result["stage"] = "completed"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", required=True)
    parser.add_argument("--prompt-data", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    with args.prompt_data.open(encoding="utf-8") as source:
        row = json.loads(next(line for line in source if line.strip()))
    result = asyncio.run(run(args.proxy, row["metadata"]["template_identity"], require_gpu=args.require_gpu))
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.receipt.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(args.receipt)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
