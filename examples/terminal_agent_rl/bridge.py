"""Host-side bridge to the pinned NexTask RFC-0012 runtime.

This file is intentionally free of slime imports.  It is executed by the
NexTask Python environment, while the rollout worker remains in slime's Python
environment.  Keeping the two dependency graphs in separate processes avoids
silently mixing their pydantic, E2B, and agent-runtime versions.

The bridge owns every operation that can see private package material:

* strict package validation and JSONL export;
* Codex runtime preparation in the solver sandbox;
* candidate snapshot publication and v3 pairwise verification;
* deletion of the temporary candidate snapshot.

The model sandbox is created from the package's frozen E2B template and never
receives ``reference_solution``, ``verify.py``, ``test.sh``, or package
manifests from the host.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

V3_PACKAGE_SCHEMA = "rfc-0012-shell-pairwise-dynamic-package-v3"
V3_PROTOCOL = "rfc-0012-shell-pairwise-verifier-v3"
EXPORT_SCHEMA = "slime.nextask-terminal-agent-dataset.v1"


class BridgeError(RuntimeError):
    """A deterministic bridge precondition or integrity failure."""


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BridgeError(f"JSON must be an object: {path}")
    return payload


def _require_real_dir(path: Path, *, label: str) -> Path:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_dir():
        raise BridgeError(f"{label} must be a real directory: {path}")
    return path


def _require_regular_file(path: Path, *, label: str) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise BridgeError(f"{label} must be a regular file: {path}")
    return path


def _package_dir(collection: Path, item_id: str, task_name: str) -> Path:
    item_dir = collection / "items" / item_id
    package = item_dir / "final" / task_name
    return _require_real_dir(package, label=f"package for {item_id}")


def _validate_replay(replay: Any, package: Path) -> None:
    descriptor = replay.descriptor
    if descriptor.schema_version != V3_PACKAGE_SCHEMA:
        raise BridgeError(f"unsupported dynamic package schema for {package}: {descriptor.schema_version!r}")
    if descriptor.verifier.protocol_version != V3_PROTOCOL:
        raise BridgeError(f"unsupported verifier protocol for {package}: {descriptor.verifier.protocol_version!r}")
    if replay.runtime_config.backend != "e2b":
        raise BridgeError(f"package does not require E2B: {package}")
    if replay.runtime_config.verifier_entrypoint != "shell_pairwise":
        raise BridgeError(f"package does not use shell_pairwise: {package}")
    if replay.runtime_config.execution_profile != "official_compat":
        raise BridgeError(f"package does not use official_compat: {package}")


def _build_export_row(
    *,
    collection: Path,
    entry: dict[str, Any],
    replay: Any,
    candidate_state_loader: Callable[..., Any],
) -> dict[str, Any]:
    item_id = str(entry.get("item_id") or "").strip()
    task_name = str(entry.get("task_name") or "").strip()
    task_id = str(entry.get("task_id") or "").strip()
    if not item_id or not task_name or not task_id:
        raise BridgeError(f"collection entry has no item/task identity: {entry!r}")

    package = _package_dir(collection, item_id, task_name)
    _validate_replay(replay, package)
    task_file = _require_regular_file(package / "task.txt", label=f"public task for {item_id}")
    prompt = task_file.read_text(encoding="utf-8").strip()
    if not prompt:
        raise BridgeError(f"public task is empty: {task_file}")

    reference_dir = _require_real_dir(package / "reference_solution", label=f"reference solution for {item_id}")
    reference_state = candidate_state_loader(reference_dir, required=True)
    if reference_state.base_template_identity != replay.descriptor.template_identity:
        raise BridgeError(f"reference template identity mismatch: {package}")
    if reference_state.baseline_workspace_digest != replay.workspace_manifest.workspace_digest:
        raise BridgeError(f"reference workspace identity mismatch: {package}")

    # package_dir remains host-only metadata consumed by this bridge.
    # generate.py never writes the path or package contents into the solver
    # sandbox; the reference path is deliberately not serialized at all.
    return {
        "prompt": prompt,
        "label": task_name,
        "metadata": {
            "dataset_schema": EXPORT_SCHEMA,
            "item_id": item_id,
            "task_id": task_id,
            "task_name": task_name,
            "package_dir": str(package),
            "template_identity": replay.descriptor.template_identity,
            "workspace_digest": replay.workspace_manifest.workspace_digest,
            "package_schema": replay.descriptor.schema_version,
            "verifier_protocol": replay.descriptor.verifier.protocol_version,
            "runtime_fingerprint": replay.descriptor.runtime_fingerprint,
            "replay_identity": replay.replay_identity,
            "package_manifest_digest": replay.package_manifest_digest,
        },
    }


def export_collection(
    collection: Path,
    output: Path,
    *,
    package_loader: Callable[[Path], Any] | None = None,
    candidate_state_loader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Validate a collection with the pinned runtime and atomically emit JSONL."""

    if package_loader is None or candidate_state_loader is None:
        from pipeline.dynamic_verification.candidate_state import load_candidate_state
        from pipeline.dynamic_verification.package import load_dynamic_verification_package

        package_loader = package_loader or load_dynamic_verification_package
        candidate_state_loader = candidate_state_loader or load_candidate_state

    collection = _require_real_dir(Path(collection), label="collection")
    manifest = _read_json_object(collection / "collection.json")
    raw_entries = manifest.get("items")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise BridgeError("collection.json must contain a non-empty items list")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in sorted(raw_entries, key=lambda value: (str(value.get("item_id")), str(value.get("task_name")))):
        if not isinstance(raw, dict):
            raise BridgeError("collection item must be an object")
        item_id = str(raw.get("item_id") or "").strip()
        task_name = str(raw.get("task_name") or "").strip()
        identity = (item_id, task_name)
        if identity in seen:
            raise BridgeError(f"duplicate collection task: {identity!r}")
        seen.add(identity)
        package = _package_dir(collection, item_id, task_name)
        replay = package_loader(package)
        rows.append(
            _build_export_row(
                collection=collection,
                entry=raw,
                replay=replay,
                candidate_state_loader=candidate_state_loader,
            )
        )

    declared_count = manifest.get("item_count")
    if declared_count != len(rows):
        raise BridgeError(f"collection item_count mismatch: declared={declared_count!r} discovered={len(rows)}")

    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, output)
    finally:
        Path(tmp_name).unlink(missing_ok=True)

    return {
        "ok": True,
        "command": "export",
        "schema": EXPORT_SCHEMA,
        "collection": str(collection),
        "output": str(output),
        "count": len(rows),
    }


def _e2b_connection() -> tuple[str | None, str | None, str | None]:
    from pipeline.bench_worker import _resolve_bench_e2b_connection

    api_key, api_url, domain = _resolve_bench_e2b_connection()
    if not api_key:
        raise BridgeError("E2B API key is not configured")
    return api_key, api_url, domain


def _connect_sandbox(sandbox_id: str, *, timeout: int) -> Any:
    from e2b import Sandbox

    api_key, api_url, domain = _e2b_connection()
    options: dict[str, Any] = {"api_key": api_key}
    if api_url:
        options["api_url"] = api_url
    if domain:
        options["domain"] = domain
    return Sandbox.connect(sandbox_id, timeout=timeout, **options)


def _remove_terminal_framework_paths(sandbox: Any) -> list[str]:
    paths = (
        "/root/.codex",
        "/app/.harness",
        "/tmp/codex_out",
        "/tmp/.terminal-codex-run.done",
        "/tmp/.terminal-codex-run.sh",
        "/tmp/.terminal-codex-run.spawned",
    )
    command = "rm -rf -- " + " ".join(shlex.quote(path) for path in paths)
    sandbox.commands.run(command, timeout=60, user="root")
    return list(paths)


def materialize_workspace(*, sandbox_id: str, package: Path) -> dict[str, Any]:
    """Upload the exact Final package public closure used by NexTask Bench."""

    from pipeline.bench import _load_public_assets
    from pipeline.dynamic_verification.package import load_dynamic_verification_package
    from pipeline.e2b_utils import e2b_upload_tar, tar_gz_directory

    package = _require_real_dir(Path(package), label="package")
    replay = load_dynamic_verification_package(package)
    _validate_replay(replay, package)
    public_assets, public_workspace_digest = _load_public_assets(package, keep_verify=False)
    sandbox = _connect_sandbox(sandbox_id, timeout=replay.runtime_config.sandbox_timeout_s)

    with tempfile.TemporaryDirectory(prefix="terminal-rl-public-workspace-") as tmp_name:
        export_root = Path(tmp_name)
        for asset in public_assets:
            relative = Path(asset["path"])
            if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
                raise BridgeError(f"unsafe workspace target: {asset['path']!r}")
            source = package / relative
            destination = export_root / relative
            if asset["node_type"] == "empty_dir":
                if source.is_symlink() or not source.is_dir() or any(source.iterdir()):
                    raise BridgeError(f"workspace empty-dir asset changed: {relative}")
                destination.mkdir(parents=True, exist_ok=False)
            elif asset["node_type"] == "file":
                if source.is_symlink() or not source.is_file():
                    raise BridgeError(f"workspace file asset changed: {relative}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            else:
                raise BridgeError(f"unsupported solver-visible node type {asset['node_type']!r}: {relative}")
        archive = tar_gz_directory(export_root)

    sandbox.commands.run("rm -rf -- /app && mkdir -p /app", timeout=60, user="root")
    e2b_upload_tar(sandbox, archive, "/app")
    check = sandbox.commands.run("test -f /app/task.txt && echo READY", timeout=30, user="root")
    if str(check.stdout or "").strip() != "READY":
        raise BridgeError("materialized workspace has no /app/task.txt")
    return {
        "ok": True,
        "command": "materialize",
        "sandbox_id": sandbox_id,
        "template_identity": replay.descriptor.template_identity,
        "workspace_digest": replay.workspace_manifest.workspace_digest,
        "public_workspace_digest": public_workspace_digest,
        "asset_count": len(public_assets),
        "archive_bytes": len(archive),
    }


def _indeterminate(error: str, *, retryable: bool = True, failure_origin: str = "sandbox_infra") -> dict[str, Any]:
    return {
        "ok": False,
        "equal": False,
        "verdict": "INDETERMINATE",
        "reason_code": "terminal_rl_bridge_error",
        "error": error,
        "retryable": retryable,
        "failure_origin": failure_origin,
        "verifier_mode": "dynamic_pairwise",
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def grade_sandbox(
    *,
    sandbox_id: str,
    package: Path,
    candidate_dir: Path,
    work_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Snapshot one live candidate, compare it, and always delete the snapshot."""

    from pipeline.bench_worker import _dynamic_bench_verify
    from pipeline.dynamic_verification.candidate_state import (
        create_candidate_state,
        delete_candidate_state_snapshot,
        write_candidate_state,
    )
    from pipeline.dynamic_verification.package import DynamicPackageIntegrityError, load_dynamic_verification_package
    from pipeline.e2b_codex import sanitize_codex_sandbox_for_snapshot

    package = _require_real_dir(Path(package), label="package")
    replay = load_dynamic_verification_package(package)
    _validate_replay(replay, package)
    reference = _require_real_dir(package / "reference_solution", label="reference solution")
    candidate_dir = Path(candidate_dir).resolve()
    work_root = Path(work_root).resolve()
    output = Path(output).resolve()
    if candidate_dir.exists():
        if candidate_dir.is_symlink() or not candidate_dir.is_dir() or any(candidate_dir.iterdir()):
            raise BridgeError(f"candidate_dir must be a new empty directory: {candidate_dir}")
    else:
        candidate_dir.mkdir(parents=True)
    work_root.mkdir(parents=True, exist_ok=True)

    api_key, api_url, domain = _e2b_connection()
    sandbox = _connect_sandbox(sandbox_id, timeout=replay.runtime_config.sandbox_timeout_s)
    state = None
    payload: dict[str, Any]
    cleanup_confirmed = False
    try:
        removed = _remove_terminal_framework_paths(sandbox)
        sanitization = sanitize_codex_sandbox_for_snapshot(sandbox)
        sanitization = {
            **sanitization,
            "terminal_rl_removed_framework_paths": removed,
        }
        state = create_candidate_state(
            sandbox,
            base_template_identity=replay.descriptor.template_identity,
            baseline_workspace_digest=replay.workspace_manifest.workspace_digest,
            api_key=api_key,
            api_url=api_url,
            domain=domain,
            sanitization_report=sanitization,
        )
        write_candidate_state(candidate_dir, state)
        payload = _dynamic_bench_verify(
            candidate_src=candidate_dir,
            reference_dir=reference,
            original_task_dir=package,
            out_path=output,
            replay=replay,
        )
    except (BridgeError, DynamicPackageIntegrityError) as exc:
        payload = _indeterminate(
            f"{type(exc).__name__}: {exc}",
            retryable=False,
            failure_origin="framework_integrity",
        )
    except Exception as exc:  # noqa: BLE001 - bridge boundary must persist a typed result
        payload = _indeterminate(f"{type(exc).__name__}: {exc}")
    finally:
        if state is None:
            cleanup_confirmed = True
        else:
            try:
                # False means the snapshot was already absent and is therefore
                # also a confirmed cleanup state.
                delete_candidate_state_snapshot(
                    state,
                    api_key=api_key,
                    api_url=api_url,
                    domain=domain,
                )
                cleanup_confirmed = True
            except Exception as exc:  # noqa: BLE001
                payload = _indeterminate(f"candidate snapshot cleanup failed: {type(exc).__name__}: {exc}")

    payload = {
        **payload,
        "candidate_snapshot_cleanup_confirmed": cleanup_confirmed,
        "bridge_schema": "slime.nextask-terminal-agent-grade.v1",
    }
    _write_json_atomic(output, payload)
    return payload


def cleanup_candidate(candidate_dir: Path) -> dict[str, Any]:
    """Idempotently delete a published temporary candidate snapshot."""

    from pipeline.dynamic_verification.candidate_state import delete_candidate_state_snapshot, load_candidate_state

    candidate_dir = Path(candidate_dir).resolve()
    if not candidate_dir.exists():
        return {"ok": True, "command": "cleanup", "state": "candidate_dir_missing"}
    state = load_candidate_state(candidate_dir, required=False)
    if state is None:
        return {"ok": True, "command": "cleanup", "state": "candidate_state_missing"}
    api_key, api_url, domain = _e2b_connection()
    delete_candidate_state_snapshot(state, api_key=api_key, api_url=api_url, domain=domain)
    return {"ok": True, "command": "cleanup", "state": "snapshot_absent"}


def _print_result(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="validate a NexTask collection and emit slime JSONL")
    export.add_argument("--collection", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)

    grade = sub.add_parser("grade", help="snapshot and grade one live E2B candidate")
    grade.add_argument("--sandbox-id", required=True)
    grade.add_argument("--package", type=Path, required=True)
    grade.add_argument("--candidate-dir", type=Path, required=True)
    grade.add_argument("--work-root", type=Path, required=True)
    grade.add_argument("--output", type=Path, required=True)

    cleanup = sub.add_parser("cleanup", help="delete a temporary candidate snapshot if present")
    cleanup.add_argument("--candidate-dir", type=Path, required=True)

    materialize = sub.add_parser("materialize", help="upload only solver-visible package assets into /app")
    materialize.add_argument("--sandbox-id", required=True)
    materialize.add_argument("--package", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command != "export":
            if __package__:
                from .e2b_network import configure_bridge_network
            else:
                from e2b_network import configure_bridge_network
            configure_bridge_network()
        if args.command == "export":
            payload = export_collection(args.collection, args.output)
        elif args.command == "materialize":
            payload = materialize_workspace(sandbox_id=args.sandbox_id, package=args.package)
        elif args.command == "grade":
            payload = grade_sandbox(
                sandbox_id=args.sandbox_id,
                package=args.package,
                candidate_dir=args.candidate_dir,
                work_root=args.work_root,
                output=args.output,
            )
        elif args.command == "cleanup":
            payload = cleanup_candidate(args.candidate_dir)
        else:  # pragma: no cover - argparse owns command validation
            raise AssertionError(args.command)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        _print_result({"ok": False, "command": args.command, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    _print_result(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
