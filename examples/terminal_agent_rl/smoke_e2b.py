"""Real E2B smoke for public workspace materialization and v3 grading."""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
from pathlib import Path

from . import runtime
from .e2b_network import resolve_e2b_proxy
from .sandbox import NextaskE2BSandbox


async def run(package: Path, template_identity: str) -> dict:
    paths = runtime.session_paths(f"e2b-smoke-{secrets.token_hex(6)}")
    cleanup_confirmed = False
    result: dict | None = None
    try:
        async with NextaskE2BSandbox(template_identity, timeout=runtime.CONFIG.grade_timeout_sec + 900) as sandbox:
            materialized = await runtime.materialize_workspace(
                sandbox_id=sandbox.sandbox_id,
                package_dir=package,
            )
            code, stdout, stderr = await sandbox.exec(
                "cd /app && test -f task.txt && test ! -e verify.py && test ! -e test.sh && test ! -e reference_solution",
                user="root",
                timeout=30,
                check=False,
            )
            if code != 0:
                raise RuntimeError(f"public workspace boundary check failed: stdout={stdout!r} stderr={stderr!r}")
            grade = await runtime.grade_sandbox(
                sandbox_id=sandbox.sandbox_id,
                package_dir=package,
                paths=paths,
            )
            cleanup_confirmed = grade.get("candidate_snapshot_cleanup_confirmed") is True
            if not cleanup_confirmed:
                raise RuntimeError(f"candidate snapshot cleanup was not confirmed: {grade!r}")
            if grade.get("ok") is not True or grade.get("verdict") not in {"EQUAL", "NON_EQUAL"}:
                raise RuntimeError(f"verifier was not determinate: {grade!r}")
            result = {
                "ok": True,
                "e2b_proxy": resolve_e2b_proxy(),
                "template_identity": template_identity,
                "public_asset_count": materialized.get("asset_count"),
                "public_workspace_digest": materialized.get("public_workspace_digest"),
                "verdict": grade.get("verdict"),
                "score": grade.get("score"),
                "max_score": grade.get("max_score"),
                "score_ratio": grade.get("score_ratio"),
                "candidate_snapshot_cleanup_confirmed": True,
            }
        if sandbox.cleanup_confirmed is not True:
            raise RuntimeError("live sandbox cleanup was not confirmed")
        assert result is not None
        result["live_sandbox_cleanup_confirmed"] = True
        return result
    finally:
        if not cleanup_confirmed:
            await runtime.cleanup_candidate(paths)
        runtime.cleanup_artifacts(paths, force=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--template-identity", required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = asyncio.run(run(args.package.resolve(), args.template_identity))
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.receipt.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(args.receipt)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
