"""只读检查 Job/Pod 与 Ray ready 标记，产出明确的启动结果供 watch 续接。"""

import argparse
import json
import subprocess
from pathlib import Path


def classify(job, pods, marker, expected_nodes, expected_gpus):
    phase = job.get("status", {}).get("state", {}).get("phase", "Unknown")
    result = {"status": "waiting", "result_kind": "ray-cluster", "actual_job": job["metadata"]["name"], "job_phase": phase}
    if phase in {"Aborted", "Aborting", "Failed", "Terminated", "Completed", "Succeeded"}:
        return {**result, "status": "failed", "reason": f"Job进入终止状态{phase}"}
    rows = []
    for pod in pods:
        status = pod.get("status", {})
        main = next((c for c in status.get("containerStatuses", []) if c["name"] == "master"), None)
        terminated = (main or {}).get("state", {}).get("terminated")
        if terminated:
            return {**result, "status": "failed", "reason": "主容器已退出", "pod": pod["metadata"]["name"], "termination": terminated}
        containers = pod["spec"].get("containers", [])
        main_spec = next((c for c in containers if c["name"] == "master"), {})
        limits = main_spec.get("resources", {}).get("limits", {})
        gpu_count = int(limits.get("nvidia.com/gpu", limits.get("nvidia.com/b300", 0)))
        ready = (
            status.get("phase") == "Running"
            and not pod["metadata"].get("deletionTimestamp")
            and any(c.get("type") == "Ready" and c.get("status") == "True" for c in status.get("conditions", []))
        )
        rows.append({"pod": pod["metadata"]["name"], "physical_host": pod["spec"].get("nodeName"), "phase": status.get("phase"), "ready": ready, "gpus": gpu_count})
    result["pods"] = rows
    hosts = sorted({p["physical_host"] for p in rows if p["physical_host"]})
    result["physical_hosts"] = hosts
    result["physical_host_count"] = len(hosts)
    if marker and len(rows) == expected_nodes and all(p["ready"] and p["gpus"] == expected_gpus for p in rows):
        if expected_gpus > 0 and len(hosts) != expected_nodes:
            return {**result, "status": "failed", "reason": f"GPU Pod未分散到{expected_nodes}台物理主机，实际{len(hosts)}台"}
        result["status"] = "ready"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--namespace", default="bg-agentic-coding")
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--result-file", help="仅用于无GPU入口诊断的回执文件名，不代表Ray就绪")
    args = parser.parse_args()

    def get(*parts):
        return json.loads(subprocess.check_output(["kubectl", "get", *parts, "-n", args.namespace, "-o", "json"], text=True))

    receipt = None
    if args.result_file:
        if Path(args.result_file).name != args.result_file:
            parser.error("result-file必须是文件名")
        path = args.run_dir / args.result_file
        if path.exists():
            receipt = json.loads(path.read_text())
    jobs = [j for j in get("jobs.batch.volcano.sh")["items"] if j.get("metadata", {}).get("annotations", {}).get("JobAlias") == args.job_name]
    if receipt is not None:
        result = {"status": "ready" if receipt.get("ok") is True else "failed", "result_kind": "entrypoint-only", "receipt": receipt, "job_alias": args.job_name}
    elif not jobs:
        previous = args.run_dir / "startup-status.json"
        seen = json.loads(previous.read_text()).get("actual_job") if previous.exists() else None
        result = {"status": "failed" if seen else "waiting", "reason": "未找到Job", "job_alias": args.job_name}
    elif len(jobs) != 1:
        result = {"status": "failed", "reason": "JobAlias非唯一", "job_alias": args.job_name}
    else:
        job = jobs[0]
        pods = get("pods", "-l", "volcano.sh/job-name=" + job["metadata"]["name"])["items"]
        result = classify(job, pods, (args.run_dir / "cluster.ready").exists(), args.nodes, args.gpus_per_node)
    output = args.run_dir / "startup-status.json"
    old = json.loads(output.read_text()) if output.exists() else None
    if old != result:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        tmp = output.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(output)
        print(json.dumps(result, ensure_ascii=False))
    # watch的目标是取得启动诊断；消费者必须检查status，failed不能启动训练。
    return 0 if result["status"] in {"ready", "failed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
