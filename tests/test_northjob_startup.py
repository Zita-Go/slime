"""启动器准备必须保留训练Python并暴露初始化失败。"""

import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

NUM_GPUS = 0
ROOT = Path(__file__).resolve().parents[1]
PREPARE = ROOT / "scripts/northjob/prepare_northlaunch.sh"
spec = importlib.util.spec_from_file_location("startup_check", ROOT / "scripts/northjob/check_slime_startup.py")
startup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(startup)


def _prepare(tmp_path, exit_code=0, collision=False, missing=False):
    target = tmp_path / "sdk/northlaunch"
    target.parent.mkdir()
    if not missing:
        target.write_text(f"#!/bin/sh\nexit {exit_code}\n")
        target.chmod(0o755)
    link_dir = tmp_path / "links"
    link_dir.mkdir()
    if collision:
        (link_dir / "northlaunch").write_text("user-owned")
    run_dir = tmp_path / "run"
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "NORTHLAUNCH_LINK_DIR": str(link_dir), "NORTHLAUNCH_RUN_DIR": str(run_dir), "RANK": "0", "JOB_NAME": "actual-cloud-job"}
    check = 'import sys,os; print(sys.executable); assert os.environ["JOB_NAME"]=="actual-cloud-job"'
    result = subprocess.run(
        ["bash", "-c", 'source "$1" test-job "$2" && python -c ' + shlex.quote(check), "test", str(PREPARE), str(target)],
        env=env, capture_output=True, text=True,
    )
    return result, link_dir, run_dir


def test_prepare_only_links_launcher_and_preserves_python(tmp_path):
    result, link_dir, run_dir = _prepare(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert sys.executable in result.stdout
    assert sorted(p.name for p in link_dir.iterdir()) == ["northlaunch"]
    assert (run_dir / "launcher-0.exit_code").read_text().strip() == "0"


@pytest.mark.parametrize("options", [{"exit_code": 7}, {"collision": True}, {"missing": True}])
def test_prepare_failure_is_not_hidden_by_and_list(tmp_path, options):
    result, link_dir, run_dir = _prepare(tmp_path, **options)
    assert result.returncode != 0
    assert "[READY]" not in result.stdout
    assert (run_dir / "launcher-0.log").exists()
    assert int((run_dir / "launcher-0.exit_code").read_text()) != 0
    if options.get("collision"):
        assert (link_dir / "northlaunch").read_text() == "user-owned"


def _job(phase="Running"):
    return {"metadata": {"name": "actual-job"}, "status": {"state": {"phase": phase}}}


def _pod(terminated=None, ready=True):
    state = {"terminated": terminated} if terminated else {"running": {}}
    return {
        "metadata": {"name": "pod"},
        "spec": {"containers": [{"name": "master", "resources": {"limits": {"nvidia.com/gpu": "8"}}}]},
        "status": {"phase": "Running", "conditions": [{"type": "Ready", "status": "True" if ready else "False"}], "containerStatuses": [{"name": "master", "state": state}]},
    }


def test_aborted_job_is_failed_even_with_stale_ready_marker():
    result = startup.classify(_job("Aborted"), [], True, 8, 8)
    assert result["status"] == "failed"


def test_running_sidecar_does_not_hide_main_container_exit():
    result = startup.classify(_job(), [_pod({"exitCode": 127, "reason": "Error"})], False, 1, 8)
    assert result["status"] == "failed"
    assert result["termination"]["exitCode"] == 127


def test_ready_requires_marker_and_complete_resources():
    pods = [_pod() for _ in range(8)]
    assert startup.classify(_job(), pods, False, 8, 8)["status"] == "waiting"
    assert startup.classify(_job(), pods[:7], True, 8, 8)["status"] == "waiting"
    assert startup.classify(_job(), pods, True, 8, 4)["status"] == "waiting"
    assert startup.classify(_job(), pods, True, 8, 8)["status"] == "ready"


def test_bootstrap_records_missing_role_before_preflight(tmp_path):
    # 不接触GPU/Ray：ROLE缺失在所有模型/资源操作之前退出。
    env = {key: value for key, value in os.environ.items() if key != "ROLE"}
    env["RANK"] = "0"
    name = "test-missing-role-" + tmp_path.name.replace("_", "-")
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/northjob/bootstrap_slime_ray_node.sh"), name, "2", "8", "1", "1", "100000000", "interactive:-"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "NorthJob 未设置 ROLE" in result.stdout
    assert "bootstrap entered" in result.stdout


def test_missing_job_waits_until_a_real_job_was_seen(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["check", "--job-name", "test-job", "--run-dir", str(tmp_path)])
    monkeypatch.setattr(startup.subprocess, "check_output", lambda *args, **kwargs: '{"items": []}')
    assert startup.main() == 1
    assert startup.main() == 1
    (tmp_path / "startup-status.json").write_text('{"actual_job": "known-job"}')
    assert startup.main() == 0
    assert json.loads((tmp_path / "startup-status.json").read_text())["status"] == "failed"


def test_entrypoint_receipt_is_not_mislabelled_as_ray_ready(tmp_path, monkeypatch):
    (tmp_path / "probe.json").write_text('{"ok": true}')
    monkeypatch.setattr(sys, "argv", ["check", "--job-name", "test-job", "--run-dir", str(tmp_path), "--result-file", "probe.json"])
    monkeypatch.setattr(startup.subprocess, "check_output", lambda *args, **kwargs: '{"items": []}')
    assert startup.main() == 0
    assert json.loads((tmp_path / "startup-status.json").read_text())["result_kind"] == "entrypoint-only"


@pytest.mark.parametrize("cpu_mode, mode", [("false", "interactive"), ("true", "train")])
def test_zero_gpu_launcher_cannot_accidentally_run_training(cpu_mode, mode):
    env = {**os.environ, "CPU_STARTUP_SMOKE": cpu_mode, "MODE": mode, "GPU_PER_NODE": "0", "NNODES": "2", "ACTOR_NODES": "1", "ROLLOUT_NODES": "1", "DRY_RUN": "true"}
    result = subprocess.run(["bash", str(ROOT / "scripts/northjob/launch_slime_multinode.sh")], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "[DRY-RUN]" not in result.stdout
