"""训练数据可扩容，但必须保持导出身份和collection完整集合。"""

import asyncio
import copy
import json

import pytest

from examples.terminal_agent_rl import probe_dataset_templates as probes
from examples.terminal_agent_rl.bridge import EXPORT_SCHEMA, V3_PACKAGE_SCHEMA, V3_PROTOCOL
from examples.terminal_agent_rl.validate_dataset import validate_dataset

NUM_GPUS = 0


def _row(index):
    return {"prompt": "Public task text.", "label": f"task-{index}", "metadata": {
        "dataset_schema": EXPORT_SCHEMA, "package_schema": V3_PACKAGE_SCHEMA, "verifier_protocol": V3_PROTOCOL,
        "item_id": f"item-{index}", "task_id": f"id-{index}", "task_name": f"task-{index}",
        "package_dir": f"/host/package-{index}", "template_identity": "nextask-codex-template-alias",
        "workspace_digest": "1" * 64, "package_manifest_digest": "2" * 64, "runtime_fingerprint": "3" * 64, "replay_identity": "4" * 64,
    }}


def _write(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _collection(rows):
    return {"item_count": len(rows), "items": [
        {"item_id": r["metadata"]["item_id"], "task_id": r["metadata"]["task_id"], "task_name": r["metadata"]["task_name"], "package_manifest_sha256": r["metadata"]["package_manifest_digest"]}
        for r in rows
    ]}


@pytest.mark.parametrize("count", [1, 101, 138])
def test_nonempty_dataset_size_is_not_fixed(tmp_path, count):
    path = tmp_path / "slime_train.jsonl"
    _write(path, [_row(i) for i in range(count)])
    assert validate_dataset(path)["count"] == count


@pytest.mark.parametrize("content", ["", "\n \n", "not-json\n", "[]\n"])
def test_empty_or_invalid_data_is_rejected(tmp_path, content):
    path = tmp_path / "train.jsonl"
    path.write_text(content)
    with pytest.raises(ValueError):
        validate_dataset(path)


@pytest.mark.parametrize("key,value", [("prompt", ""), ("metadata", None), ("label", "wrong-label")])
def test_required_row_fields(tmp_path, key, value):
    row = _row(0)
    row[key] = value
    path = tmp_path / "train.jsonl"
    _write(path, [row])
    with pytest.raises(ValueError):
        validate_dataset(path)


@pytest.mark.parametrize("key,value", [("dataset_schema", "old-schema"), ("verifier_protocol", "old-protocol"), ("template_identity", ""), ("workspace_digest", "wrong-digest")])
def test_invalid_runtime_metadata_is_rejected(tmp_path, key, value):
    row = _row(0)
    row["metadata"][key] = value
    path = tmp_path / "train.jsonl"
    _write(path, [row])
    with pytest.raises(ValueError):
        validate_dataset(path)


def test_duplicate_item_is_rejected(tmp_path):
    path = tmp_path / "train.jsonl"
    _write(path, [_row(0), _row(0)])
    with pytest.raises(ValueError, match="重复"):
        validate_dataset(path)


def test_current_138_items_must_match_collection_and_digests(tmp_path):
    rows = [_row(i) for i in range(138)]
    path = tmp_path / "slime_train.jsonl"
    collection = _collection(rows)
    (tmp_path / "collection.json").write_text(json.dumps(collection))
    _write(path, rows)
    assert validate_dataset(path)["collection_checked"] is True
    _write(path, rows[:101])
    with pytest.raises(ValueError, match="重新导出"):
        validate_dataset(path)
    changed = copy.deepcopy(rows)
    changed[0]["metadata"]["package_manifest_digest"] = "a" * 64
    _write(path, changed)
    with pytest.raises(ValueError, match="digest"):
        validate_dataset(path)


@pytest.mark.parametrize("fail", [False, True])
def test_live_probe_deduplicates_templates_and_requires_cleanup(tmp_path, monkeypatch, fail):
    calls = []

    async def run(proxy, template):
        assert proxy is None  # CPU直连，不继承通用代理。
        calls.append(template)
        return {"ok": True, "live_sandbox_cleanup_confirmed": not (fail and template == "b"), "template_identity": template, "stage": "completed"}

    monkeypatch.setattr(probes, "run", run)
    audit = {"ok": True, "records": [{"template_identity": t} for t in ["b", "a", "b"]]}
    receipt = tmp_path / "receipt.json"
    result = asyncio.run(probes.probe(audit, receipt, 2))
    assert sorted(calls) == ["a", "b"]
    assert result["template_count"] == 2
    assert result["ok"] is (not fail)
    assert json.loads(receipt.read_text()) == result


def test_failed_static_audit_cannot_start_live_probes(tmp_path):
    with pytest.raises(ValueError):
        asyncio.run(probes.probe({"ok": False}, tmp_path / "receipt.json", 1))
