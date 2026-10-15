"""真实Harbor/E2B/verifier冒烟；固定空回答，不使用模型/GPU、不代表TB2.1模型分数。"""

from __future__ import annotations

import argparse
import asyncio
import json
from types import SimpleNamespace

from aiohttp import web

from .tb21_dataset import validate_dataset


async def run(options) -> None:
    from .tb21_harbor import run_trial

    rows = validate_dataset(options.prompt_data)
    assert "openssl-selfsigned-cert" in {row["label"] for row in rows}
    calls = 0

    async def answer(request):
        nonlocal calls
        calls += 1
        return web.json_response({"choices": [{"message": {"role": "assistant", "content": "No-model harness smoke."}}]})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", answer)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    options.output.mkdir(parents=True, exist_ok=False)
    try:
        await run_trial(
            SimpleNamespace(
                task_dir=options.prompt_data.parent / "tasks" / "openssl-selfsigned-cert",
                output=options.output,
                receipt=options.output / "receipt.json",
                adapter_url=f"http://127.0.0.1:{port}",
                session_id="no-model-smoke",
            )
        )
        receipt = json.loads((options.output / "receipt.json").read_text())
        receipt.update(no_model_smoke=True, model_gpu_used=False, adapter_requests=calls)
        (options.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        assert calls > 0, "Harbor agent never called the adapter"
        print(json.dumps(receipt))
    finally:
        await runner.cleanup()


def main() -> None:
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
