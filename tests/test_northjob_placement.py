"""物理主机分散必须是必选约束；dry-run不能触达提交API。"""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

NUM_GPUS = 0
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("slime_submit", ROOT / "scripts/northjob/submit_slime_northjob.py")
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)


def _job(alias="test-job"):
    template = {"metadata": {"labels": {"user-label": "keep"}}, "spec": {
        "affinity": {"nodeAffinity": {"user-rule": "keep"}},
        "containers": [{"name": "master", "resources": {"limits": {"nvidia.com/gpu": "2"}}}],
    }}
    return {
        "metadata": {"namespace": "test-ns", "annotations": {"JobAlias": alias}},
        "spec": {"tasks": [{"name": "master", "replicas": 1, "template": copy.deepcopy(template)}, {"name": "worker", "replicas": 1, "template": copy.deepcopy(template)}]},
    }


def test_placement_is_hard_scoped_and_preserves_original_fields():
    job = _job()
    original = copy.deepcopy(job)
    placed = submit.require_distinct_hosts(job)
    assert job == original
    for task in placed["spec"]["tasks"]:
        template = task["template"]
        assert template["metadata"]["labels"] == {"user-label": "keep", submit.HOST_GROUP_LABEL: "test-job"}
        assert template["spec"]["affinity"]["nodeAffinity"] == {"user-rule": "keep"}
        assert template["spec"]["containers"] == job["spec"]["tasks"][0]["template"]["spec"]["containers"]
        required = template["spec"]["affinity"]["podAntiAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]
        assert required == [{"labelSelector": {"matchLabels": {submit.HOST_GROUP_LABEL: "test-job"}}, "namespaces": ["test-ns"], "topologyKey": "kubernetes.io/hostname"}]
    assert submit.require_distinct_hosts(placed) == placed
    assert submit.require_distinct_hosts(_job("other-job")) != placed


def test_existing_anti_affinity_is_retained_and_label_collision_fails():
    job = _job()
    template = job["spec"]["tasks"][0]["template"]
    original = {"labelSelector": {"matchLabels": {"other": "rule"}}, "topologyKey": "zone"}
    template["spec"]["affinity"]["podAntiAffinity"] = {"requiredDuringSchedulingIgnoredDuringExecution": [original]}
    placed = submit.require_distinct_hosts(job)
    assert placed["spec"]["tasks"][0]["template"]["spec"]["affinity"]["podAntiAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][0] == original
    template["metadata"]["labels"][submit.HOST_GROUP_LABEL] = "user-owned"
    with pytest.raises(ValueError, match="占用"):
        submit.require_distinct_hosts(job)


def test_explicit_network_probe_forces_host_network_without_mutating_source():
    job = _job()
    original = copy.deepcopy(job)
    networked = submit.require_host_network(job)
    assert job == original
    for task in networked["spec"]["tasks"]:
        assert task["template"]["spec"]["hostNetwork"] is True
        assert task["template"]["spec"]["dnsPolicy"] == "ClusterFirstWithHostNet"
    submit.validate_host_network(networked)
    with pytest.raises(ValueError, match="hostNetwork"):
        submit.validate_host_network(job)


def _sdk(tmp_path, *, response=None, tamper=False):
    uploads = []
    sdk = SimpleNamespace(build_job_yaml=_job)

    def upload(**options):
        uploads.append(options["yamlPath"])
        return response if response is not None else {"code": 200}

    sdk.infrawave_register_job_info_by_upload_yaml = upload

    def create(args):
        job = sdk.build_job_yaml()
        if tamper:
            job["spec"]["tasks"][1]["template"]["spec"].pop("affinity")
        path = tmp_path / "job.yaml"
        path.write_text(yaml.safe_dump(job))
        return sdk.infrawave_register_job_info_by_upload_yaml(username="test", password="not-a-real-secret", yamlPath=str(path))

    sdk.create_job = create
    return sdk, uploads


@pytest.mark.parametrize("dry_run", [True, False])
def test_dry_run_never_uploads_and_real_submission_uses_validated_yaml(tmp_path, dry_run, capsys):
    sdk, uploads = _sdk(tmp_path)
    build, upload = sdk.build_job_yaml, sdk.infrawave_register_job_info_by_upload_yaml
    result = submit.run_create(None, sdk, dry_run=dry_run)
    assert result["code"] in {0, 200}
    assert len(uploads) == (0 if dry_run else 1)
    submit.validate_distinct_hosts(yaml.safe_load((tmp_path / "job.yaml").read_text()))
    assert sdk.build_job_yaml is build
    assert sdk.infrawave_register_job_info_by_upload_yaml is upload
    assert "not-a-real-secret" not in capsys.readouterr().out


def test_forced_host_network_is_validated_before_upload(tmp_path):
    sdk, uploads = _sdk(tmp_path)
    submit.run_create(None, sdk, force_host_network=True)
    saved = yaml.safe_load((tmp_path / "job.yaml").read_text())
    submit.validate_distinct_hosts(saved)
    submit.validate_host_network(saved)
    assert len(uploads) == 1


def test_tampered_saved_manifest_is_rejected_before_upload(tmp_path):
    sdk, uploads = _sdk(tmp_path, tamper=True)
    build, upload = sdk.build_job_yaml, sdk.infrawave_register_job_info_by_upload_yaml
    with pytest.raises(ValueError, match="反亲和"):
        submit.run_create(None, sdk)
    assert not uploads
    assert sdk.build_job_yaml is build
    assert sdk.infrawave_register_job_info_by_upload_yaml is upload


@pytest.mark.parametrize("response", [{"code": 500}, {"message": "missing code"}])
def test_api_rejection_fails_without_retry(tmp_path, response):
    sdk, uploads = _sdk(tmp_path, response=response)
    with pytest.raises(RuntimeError, match="未确认接受"):
        submit.run_create(None, sdk)
    assert len(uploads) == 1
