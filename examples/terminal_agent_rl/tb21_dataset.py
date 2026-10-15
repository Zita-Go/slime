"""固定TB2.1全量测试集；与NexTask训练数据完全分离。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

DATASET = "terminal-bench/terminal-bench-2-1"
VERSION = "sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a"
COUNT = 89
SCHEMA = "slime.tb21-eval.v1"


def file_hashes(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"TB2.1 task must not contain symlinks: {path}")
        if path.is_file():
            with path.open("rb") as stream:
                result[path.relative_to(root).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def validate_dataset(path: Path) -> list[dict]:
    manifest = json.loads(path.with_name("manifest.json").read_text())
    if (manifest.get("dataset"), manifest.get("version"), manifest.get("schema")) != (DATASET, VERSION, SCHEMA):
        raise ValueError("TB2.1 dataset identity mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    tasks = manifest["tasks"]
    if len(rows) != COUNT or len(tasks) != COUNT or len({row["label"] for row in rows}) != COUNT:
        raise ValueError("TB2.1 evaluation requires all 89 unique tasks")
    if set(tasks) != {row["label"] for row in rows}:
        raise ValueError("TB2.1 JSONL and manifest task sets differ")
    registry = manifest["registry_metadata"]
    identities = {item["name"]: item["ref"] for item in registry["task_ids"]}
    if registry["name"] != DATASET or registry["version"] != VERSION or len(identities) != COUNT:
        raise ValueError("TB2.1 registry snapshot mismatch")
    if identities != {name: task["digest"] for name, task in tasks.items()}:
        raise ValueError("TB2.1 task digests do not match the fixed registry snapshot")
    for row in rows:
        name = row["label"]
        task = tasks[name]
        root = path.parent / "tasks" / name
        if root.resolve().parent != (path.parent / "tasks").resolve():
            raise ValueError("Invalid TB2.1 task path")
        if row.get("metadata") != {"dataset_schema": SCHEMA, "task_name": name, "task_digest": task["digest"]}:
            raise ValueError(f"TB2.1 task metadata mismatch: {name}")
        if not task["digest"].startswith("sha256:") or len(task["digest"]) != 71:
            raise ValueError(f"Missing registry task digest: {name}")
        if file_hashes(root) != task["files"]:
            raise ValueError(f"TB2.1 task files changed: {name}")
        for required in ("instruction.md", "task.toml", "tests/test.sh", "environment/Dockerfile"):
            if required not in task["files"]:
                raise ValueError(f"Incomplete TB2.1 task {name}: {required}")
        if row["prompt"] != (root / "instruction.md").read_text().strip():
            raise ValueError(f"TB2.1 public instruction changed: {name}")
    return rows


async def prepare(output: Path) -> None:
    from harbor.registry.client.package import PackageDatasetClient
    from harbor.tasks.client import TaskClient

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite TB2.1 snapshot: {output}")
    client = PackageDatasetClient()
    metadata = await asyncio.wait_for(client.get_dataset_metadata(f"{DATASET}@{VERSION}"), timeout=60)
    if metadata.version != VERSION or len(metadata.task_ids) != COUNT:
        raise ValueError("Official TB2.1 registry identity/count mismatch")
    # Download only if absent in Harbor's content-addressed cache; no dataset mutations.
    downloaded = await TaskClient().download_tasks(task_ids=metadata.task_ids, overwrite=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".tb21-", dir=output.parent))
    rows, tasks = [], {}
    for task_id, downloaded_task in zip(metadata.task_ids, downloaded.results, strict=True):
        name = task_id.name
        if name in tasks or Path(name).name != name:
            raise ValueError(f"Duplicate/invalid TB2.1 task: {name}")
        target = staging / "tasks" / name
        shutil.copytree(downloaded_task.path, target, symlinks=True)
        tasks[name] = {"digest": task_id.ref, "files": file_hashes(target)}
        rows.append(
            {
                "prompt": (target / "instruction.md").read_text().strip(),
                "label": name,
                "metadata": {"dataset_schema": SCHEMA, "task_name": name, "task_digest": task_id.ref},
            }
        )
    (staging / "manifest.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "dataset": DATASET,
                "version": VERSION,
                "tasks": tasks,
                "registry_metadata": metadata.model_dump(mode="json"),
            },
            indent=2,
        )
        + "\n"
    )
    (staging / "tb21.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    validate_dataset(staging / "tb21.jsonl")
    staging.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--check", type=Path)
    options = parser.parse_args()
    if options.output:
        asyncio.run(prepare(options.output))
        path = options.output / "tb21.jsonl"
    else:
        path = options.check
    rows = validate_dataset(path)
    print(json.dumps({"ok": True, "count": len(rows), "version": VERSION, "path": str(path)}))


if __name__ == "__main__":
    main()
