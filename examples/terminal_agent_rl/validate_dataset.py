"""训练前校验实际 JSONL 与邻近 collection.json，不固定题目数量。"""

import argparse
import json
import re
from pathlib import Path

from .bridge import EXPORT_SCHEMA, V3_PACKAGE_SCHEMA, V3_PROTOCOL


def validate_dataset(path: Path) -> dict:
    rows = {}
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"训练 JSONL 第{number}行不是合法JSON") from exc
            if not isinstance(row, dict) or not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                raise ValueError(f"训练 JSONL 第{number}行缺少非空原始题面")
            metadata = row.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(f"训练 JSONL 第{number}行缺少metadata")
            for key, value in (("dataset_schema", EXPORT_SCHEMA), ("package_schema", V3_PACKAGE_SCHEMA), ("verifier_protocol", V3_PROTOCOL)):
                if metadata.get(key) != value:
                    raise ValueError(f"训练 JSONL 第{number}行的{key}不受支持")
            for key in ("item_id", "task_id", "task_name", "package_dir", "template_identity"):
                if not isinstance(metadata.get(key), str) or not metadata[key].strip():
                    raise ValueError(f"训练 JSONL 第{number}行缺少{key}")
            for key in ("workspace_digest", "package_manifest_digest", "runtime_fingerprint", "replay_identity"):
                if not isinstance(metadata.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", metadata[key]):
                    raise ValueError(f"训练 JSONL 第{number}行的{key}不是SHA256")
            if row.get("label") != metadata["task_name"]:
                raise ValueError(f"训练 JSONL 第{number}行的label与task_name不一致")
            if metadata["item_id"] in rows:
                raise ValueError(f"训练 JSONL 包含重复item_id：{metadata['item_id']}")
            rows[metadata["item_id"]] = row
    if not rows:
        raise ValueError("训练 JSONL 不能为空")

    collection_path = path.parent / "collection.json"
    if collection_path.exists():
        collection = json.loads(collection_path.read_text(encoding="utf-8"))
        entries = collection.get("items") if isinstance(collection, dict) else None
        if not isinstance(entries, list) or not entries or collection.get("item_count") != len(entries):
            raise ValueError("collection.json的item_count/items不一致")
        expected = {}
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("item_id"), str) or not entry["item_id"]:
                raise ValueError("collection.json题目身份无效")
            if entry["item_id"] in expected:
                raise ValueError("collection.json包含重复item_id")
            expected[entry["item_id"]] = entry
        if set(rows) != set(expected):
            raise ValueError(f"训练 JSONL 与collection不一致：训练{len(rows)}题/collection{len(expected)}题，请重新导出")
        for item_id, row in rows.items():
            entry = expected[item_id]
            metadata = row["metadata"]
            if metadata["task_id"] != entry.get("task_id") or metadata["task_name"] != entry.get("task_name"):
                raise ValueError(f"训练题目身份与collection不一致：{item_id}")
            if metadata["package_manifest_digest"] != entry.get("package_manifest_sha256"):
                raise ValueError(f"训练题包digest与collection不一致：{item_id}")
    return {"ok": True, "count": len(rows), "unique_templates": len({r["metadata"]["template_identity"] for r in rows.values()}), "collection_checked": collection_path.exists()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-data", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate_dataset(args.prompt_data)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"[ERROR] {exc}\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
