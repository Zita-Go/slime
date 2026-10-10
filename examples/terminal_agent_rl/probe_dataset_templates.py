"""从静态审计回执逐个验证E2B模板；不运行模型，不继承通用代理。"""

import argparse
import asyncio
import json
from pathlib import Path

from .smoke_e2b_proxy import run


async def probe(audit: dict, receipt: Path, concurrency: int) -> dict:
    if audit.get("ok") is not True or not audit.get("records"):
        raise ValueError("需要成功的静态题包审计回执")
    templates = sorted({r["template_identity"] for r in audit["records"]})
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def one(template):
        async with sem:
            result = await run(None, template)
            results.append(result)
            print(json.dumps({"checked": len(results), "total": len(templates), "template_identity": template, "ok": result["ok"], "stage": result["stage"]}), flush=True)

    await asyncio.gather(*(one(template) for template in templates))
    payload = {"ok": all(r["ok"] and r["live_sandbox_cleanup_confirmed"] for r in results), "routing": "direct-no-inherited-proxy", "template_count": len(templates), "records": sorted(results, key=lambda r: r["template_identity"])}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    tmp = receipt.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(receipt)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 8:
        parser.error("concurrency必须在1到8之间")
    result = asyncio.run(probe(json.loads(args.audit.read_text()), args.receipt, args.concurrency))
    print(json.dumps({"ok": result["ok"], "template_count": result["template_count"], "receipt": str(args.receipt)}))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
