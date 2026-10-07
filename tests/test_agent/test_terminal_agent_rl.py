"""CPU-only coverage for the NexTask terminal-agent rollout adapter."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")
    for name in ("AutoProcessor", "AutoTokenizer", "PreTrainedTokenizerBase", "ProcessorMixin"):
        setattr(transformers_stub, name, type(name, (), {}))
    sys.modules["transformers"] = transformers_stub

if not hasattr(asyncio, "timeout"):

    @contextlib.asynccontextmanager
    async def _timeout_shim(_delay):
        yield

    asyncio.timeout = _timeout_shim

import examples.terminal_agent_rl.bridge as bridge  # noqa: E402
import examples.terminal_agent_rl.generate as generate  # noqa: E402
from examples.terminal_agent_rl import runtime  # noqa: E402
from slime.agent.adapters import common as adapters_common  # noqa: E402
from slime.agent import sandbox as sandbox_module  # noqa: E402
from slime.utils.misc import SingletonMeta  # noqa: E402
from slime.utils.types import Sample  # noqa: E402
from tests.test_agent._fakes import FakeSandbox, FakeTokenizer, fake_call_sglang_generate  # noqa: E402

_REAL_SLEEP = asyncio.sleep


async def _fast_sleep(_seconds):
    await _REAL_SLEEP(0)


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        hf_checkpoint="unused",
        rollout_max_context_len=0,
        sglang_tool_call_parser=None,
        sglang_reasoning_parser=None,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=1,
    )


def _sample(package_dir: Path) -> Sample:
    return Sample(
        index=3,
        group_index=2,
        prompt="Repair the service and leave it running.",
        metadata={
            "dataset_schema": bridge.EXPORT_SCHEMA,
            "item_id": "item-1",
            "task_id": "task-id-1",
            "task_name": "task_task-id-1",
            "package_dir": str(package_dir),
            "reference_dir": str(package_dir / "reference_solution"),
            "template_identity": "template-1",
            "workspace_digest": "1" * 64,
            "package_schema": bridge.V3_PACKAGE_SCHEMA,
            "verifier_protocol": bridge.V3_PROTOCOL,
        },
    )


async def _codex_agent(env: dict[str, str]) -> int:
    base_url = env["OPENAI_BASE_URL"]
    token = env["OPENAI_API_KEY"]
    history = [{"role": "user", "content": "solve the terminal task"}]
    async with aiohttp.ClientSession(trust_env=False) as session:
        for _ in range(2):
            async with session.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {token}"},
                json={"model": "m", "max_tokens": 64, "messages": history},
            ) as response:
                payload = await response.json()
            message = payload["choices"][0]["message"]
            history.append({"role": "assistant", "content": message.get("content") or ""})
            history.append({"role": "user", "content": "continue"})
    return 0


def _patch_rollout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, grade: dict) -> FakeTokenizer:
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(
        generate,
        "CONFIG",
        dataclasses.replace(
            generate.CONFIG,
            adapter_public_host="127.0.0.1",
            adapter_bind_host="127.0.0.1",
            adapter_port=0,
            rollout_guard_sec=60,
            agent_time_budget_sec=30,
            boot_retries=1,
        ),
    )
    monkeypatch.setattr(generate, "load_tokenizer", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(generate, "NextaskE2BSandbox", FakeSandbox.factory(on_launch=_codex_agent))
    monkeypatch.setattr(sandbox_module.asyncio, "sleep", _fast_sleep)
    monkeypatch.setattr(
        adapters_common,
        "call_sglang_generate",
        fake_call_sglang_generate(
            [
                ("inspect the machine", "stop", None),
                ("task completed", "stop", None),
            ],
            tokenizer,
        ),
    )

    async def _prepare(**_kwargs):
        return {"ok": True, "codex_version": "codex-cli test"}

    async def _grade(**_kwargs):
        return dict(grade)

    async def _cleanup(_paths):
        return None

    monkeypatch.setattr(runtime, "prepare_sandbox", _prepare)
    monkeypatch.setattr(runtime, "grade_sandbox", _grade)
    monkeypatch.setattr(runtime, "cleanup_candidate", _cleanup)
    monkeypatch.setattr(runtime, "cleanup_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runtime,
        "session_paths",
        lambda _sid: {
            "root": tmp_path / "work",
            "candidate": tmp_path / "work" / "candidate",
            "verifier_work": tmp_path / "work" / "verifier",
            "result": tmp_path / "work" / "result.json",
        },
    )
    SingletonMeta.clear_instances(generate._AdapterService)
    return tokenizer


def test_terminal_generate_closes_agent_loop_and_uses_partial_reward(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "reference_solution").mkdir()
    grade = {
        "ok": True,
        "verdict": "NON_EQUAL",
        "score": 60,
        "max_score": 100,
        "score_ratio": 0.6,
        "candidate_snapshot_cleanup_confirmed": True,
    }

    async def run_case(monkeypatch: pytest.MonkeyPatch):
        _patch_rollout(monkeypatch, tmp_path, grade)
        samples = await generate.generate(_args(), _sample(package), {"max_new_tokens": 64})
        assert samples
        assert sum(float(sample.reward) for sample in samples) == pytest.approx(0.6)
        assert all(sample.status == Sample.Status.COMPLETED for sample in samples)
        assert all(sum(sample.loss_mask or []) > 0 for sample in samples)
        assert all(sample.metadata["verifier_verdict"] == "NON_EQUAL" for sample in samples)
        assert all(sample.metadata["agent_exit_code"] == 0 for sample in samples)

    with pytest.MonkeyPatch.context() as monkeypatch:
        asyncio.run(run_case(monkeypatch))


def test_terminal_generate_drops_indeterminate_verifier_result(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "reference_solution").mkdir()
    grade = {
        "ok": False,
        "verdict": "INDETERMINATE",
        "reason_code": "sandbox_infra",
        "candidate_snapshot_cleanup_confirmed": True,
    }

    async def run_case(monkeypatch: pytest.MonkeyPatch):
        _patch_rollout(monkeypatch, tmp_path, grade)
        samples = await generate.generate(_args(), _sample(package), {})
        assert len(samples) == 1
        assert samples[0].status == Sample.Status.ABORTED
        assert samples[0].remove_sample is True
        assert samples[0].metadata["abort_reason"] == "verifier_indeterminate:sandbox_infra"

    with pytest.MonkeyPatch.context() as monkeypatch:
        asyncio.run(run_case(monkeypatch))


def test_reward_requires_determinate_verdict_and_snapshot_cleanup() -> None:
    assert generate._reward_from_grade(
        {"ok": True, "verdict": "EQUAL", "candidate_snapshot_cleanup_confirmed": True}
    ) == 1.0
    assert generate._reward_from_grade(
        {
            "ok": True,
            "verdict": "NON_EQUAL",
            "score_ratio": 0.25,
            "candidate_snapshot_cleanup_confirmed": True,
        }
    ) == 0.25
    assert (
        generate._reward_from_grade(
            {"ok": False, "verdict": "INDETERMINATE", "candidate_snapshot_cleanup_confirmed": True}
        )
        is None
    )
    assert generate._reward_from_grade({"ok": True, "verdict": "EQUAL"}) is None


def test_export_collection_emits_only_public_prompt_and_host_metadata(tmp_path: Path) -> None:
    collection = tmp_path / "collection"
    package = collection / "items" / "item-1" / "final" / "task_task-id-1"
    reference = package / "reference_solution"
    reference.mkdir(parents=True)
    (package / "task.txt").write_text("Public task text.\n", encoding="utf-8")
    (reference / "private.txt").write_text("SECRET REFERENCE", encoding="utf-8")
    (collection / "collection.json").write_text(
        json.dumps(
            {
                "item_count": 1,
                "items": [
                    {
                        "item_id": "item-1",
                        "task_id": "task-id-1",
                        "task_name": "task_task-id-1",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    descriptor = SimpleNamespace(
        schema_version=bridge.V3_PACKAGE_SCHEMA,
        verifier=SimpleNamespace(protocol_version=bridge.V3_PROTOCOL),
        template_identity="template-1",
        runtime_fingerprint="2" * 64,
    )
    replay = SimpleNamespace(
        descriptor=descriptor,
        runtime_config=SimpleNamespace(
            backend="e2b",
            verifier_entrypoint="shell_pairwise",
            execution_profile="official_compat",
        ),
        workspace_manifest=SimpleNamespace(workspace_digest="1" * 64),
        replay_identity="3" * 64,
        package_manifest_digest="4" * 64,
    )
    reference_state = SimpleNamespace(
        base_template_identity="template-1",
        baseline_workspace_digest="1" * 64,
    )
    output = tmp_path / "train.jsonl"

    result = bridge.export_collection(
        collection,
        output,
        package_loader=lambda _path: replay,
        candidate_state_loader=lambda _path, required: reference_state,
    )

    assert result["count"] == 1
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["prompt"] == "Public task text."
    assert row["metadata"]["template_identity"] == "template-1"
    assert "SECRET REFERENCE" not in json.dumps(row)
