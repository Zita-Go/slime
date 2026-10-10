"""在本进程内适配 NorthJob 清单，强制同一 Slime Job 的 Pod 分散到物理主机。"""

import copy
import importlib
import json
import sys
from pathlib import Path

HOST_GROUP_LABEL = "slime.rl/physical-host-group"


def require_distinct_hosts(job):
    """保留 SDK 清单其他字段；选择器仅匹配本 Job、同命名空间的 Pod。"""
    result = copy.deepcopy(job)
    metadata = result["metadata"]
    alias = metadata["annotations"]["JobAlias"]
    namespace = metadata["namespace"]
    term = {
        "labelSelector": {"matchLabels": {HOST_GROUP_LABEL: alias}},
        "namespaces": [namespace],
        "topologyKey": "kubernetes.io/hostname",
    }
    for task in result["spec"]["tasks"]:
        template = task["template"]
        labels = template.setdefault("metadata", {}).setdefault("labels", {})
        if HOST_GROUP_LABEL in labels and labels[HOST_GROUP_LABEL] != alias:
            raise ValueError("物理主机分组标签已被其他配置占用")
        labels[HOST_GROUP_LABEL] = alias
        anti_affinity = template["spec"].setdefault("affinity", {}).setdefault("podAntiAffinity", {})
        required = anti_affinity.setdefault("requiredDuringSchedulingIgnoredDuringExecution", [])
        if term not in required:
            required.append(copy.deepcopy(term))
    return result


def validate_distinct_hosts(job):
    expected = require_distinct_hosts(job)
    if expected != job:
        raise ValueError("上传清单缺少必选物理主机反亲和约束")


def run_create(args, sdk, *, dry_run=False):
    """复用 SDK 的入口/资源/认证与上传 API；不改共享 SDK 文件。"""
    original_build = sdk.build_job_yaml
    original_upload = sdk.infrawave_register_job_info_by_upload_yaml

    def build(*parts, **options):
        return require_distinct_hosts(original_build(*parts, **options))

    def upload(**options):
        import yaml

        job = yaml.safe_load(Path(options["yamlPath"]).read_text())
        validate_distinct_hosts(job)
        print(json.dumps({"physical_host_policy": "required-distinct", "job_alias": job["metadata"]["annotations"]["JobAlias"], "dry_run": dry_run}, ensure_ascii=False))
        if dry_run:
            return {"code": 0, "dry_run": True}
        return original_upload(**options)

    sdk.build_job_yaml = build
    sdk.infrawave_register_job_info_by_upload_yaml = upload
    try:
        result = sdk.create_job(args)
    finally:
        sdk.build_job_yaml = original_build
        sdk.infrawave_register_job_info_by_upload_yaml = original_upload
    if not isinstance(result, dict) or result.get("code") not in {0, 200}:
        raise RuntimeError("NorthJob 未确认接受任务；请先核验实际 Job，不自动重试")
    return result


def main():
    # 包装器选项以 -- 与原 NorthJob 参数隔开，不重新实现 SDK 参数解析。
    argv = sys.argv[1:]
    dry_run = bool(argv and argv[0] == "--dry-run")
    if dry_run:
        argv = argv[1:]
    if not argv or argv[0] != "--":
        raise SystemExit("用法: submit_slime_northjob.py [--dry-run] -- create <NorthJob参数>")
    cli = importlib.import_module("northmirror.launcher.northjob")
    sdk = importlib.import_module("northmirror.launcher.create_job")
    original_argv = sys.argv
    try:
        sys.argv = ["northjob", *argv[1:]]
        args = cli.parse_args()
    finally:
        sys.argv = original_argv
    if args.command != "create" or args.job_type != "train" or args.train_launcher != "raw" or args.nnodes < 2:
        raise SystemExit("仅支持多节点 Slime train/raw 任务")
    run_create(args, sdk, dry_run=dry_run)


if __name__ == "__main__":
    main()
