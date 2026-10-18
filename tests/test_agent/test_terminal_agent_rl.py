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
from examples.terminal_agent_rl.agent import TerminalToolAgent  # noqa: E402
from examples.terminal_agent_rl.claude_code import RootClaudeCodeHarness  # noqa: E402
from examples.terminal_agent_rl.e2b_config import resolve_e2b_connection  # noqa: E402
from examples.terminal_agent_rl.sandbox import NextaskE2BSandbox  # noqa: E402
from examples.terminal_agent_rl import runtime  # noqa: E402
from slime.agent.adapters import common as adapters_common  # noqa: E402
from slime.agent import sandbox as sandbox_module  # noqa: E402
from slime.utils.misc import SingletonMeta  # noqa: E402
from slime.utils.types import Sample  # noqa: E402
from tests.test_agent._fakes import FakeSandbox, FakeTokenizer, fake_call_sglang_generate  # noqa: E402

_REAL_SLEEP = asyncio.sleep
NUM_GPUS = 0


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


def test_training_defaults_select_claude_code_with_four_hour_budget(monkeypatch):
    monkeypatch.delenv("TERMINAL_RL_AGENT", raising=False)
    monkeypatch.delenv("ADAPTER_PUBLIC_HOST", raising=False)
    monkeypatch.delenv("TERMINAL_RL_AGENT_TIME_BUDGET_SEC", raising=False)
    monkeypatch.delenv("TERMINAL_RL_ROLLOUT_GUARD_SEC", raising=False)
    monkeypatch.delenv("TERMINAL_RL_MAX_TURNS", raising=False)
    monkeypatch.delenv("TERMINAL_RL_REASONING_EFFORT", raising=False)
    config = generate.TerminalConfig.from_env()
    agent = TerminalToolAgent()
    assert config.agent_name == "claude_code"
    assert config.cc_transport == "e2b_pull"
    assert config.adapter_public_host is None
    assert config.adapter_bind_host == "127.0.0.1"
    assert config.agent_time_budget_sec == 14400
    assert config.rollout_guard_sec == (
        14400 + runtime.CONFIG.workspace_timeout_sec + runtime.CONFIG.grade_timeout_sec + 300
    )
    assert agent.max_turns == 400
    assert agent.reasoning_effort == "high"


def _patch_rollout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, grade: dict) -> FakeTokenizer:
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(
        generate,
        "CONFIG",
        dataclasses.replace(
            generate.CONFIG,
            agent_name="terminal_tool",
            adapter_public_host=None,
            adapter_bind_host="127.0.0.1",
            adapter_port=0,
            rollout_guard_sec=60,
            agent_time_budget_sec=30,
            boot_retries=1,
        ),
    )
    monkeypatch.setattr(generate, "load_tokenizer", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(generate, "NextaskE2BSandbox", FakeSandbox.factory())
    monkeypatch.setattr(sandbox_module.asyncio, "sleep", _fast_sleep)
    monkeypatch.setattr(
        adapters_common,
        "call_sglang_generate",
        fake_call_sglang_generate(
            [("task completed", "stop", None)],
            tokenizer,
        ),
    )

    async def _grade(**_kwargs):
        return dict(grade)

    async def _materialize(**_kwargs):
        return {"ok": True, "asset_count": 1}

    async def _cleanup(_paths):
        return None

    monkeypatch.setattr(runtime, "grade_sandbox", _grade)
    monkeypatch.setattr(runtime, "materialize_workspace", _materialize)
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
        tokenizer = _patch_rollout(monkeypatch, tmp_path, grade)
        samples = await generate.generate(_args(), _sample(package), {"max_new_tokens": 64})
        assert samples
        assert sum(float(sample.reward) for sample in samples) == pytest.approx(0.6)
        assert all(sample.status == Sample.Status.COMPLETED for sample in samples)
        assert all(sum(sample.loss_mask or []) > 0 for sample in samples)
        assert all(sample.metadata["verifier_verdict"] == "NON_EQUAL" for sample in samples)
        assert all(sample.metadata["agent_exit_code"] == 0 for sample in samples)
        assert tokenizer.template_kwargs == [{"reasoning_effort": "high"}]

    with pytest.MonkeyPatch.context() as monkeypatch:
        asyncio.run(run_case(monkeypatch))


def test_terminal_generate_uses_claude_code_and_anthropic_adapter(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    grade = {
        "ok": True,
        "verdict": "NON_EQUAL",
        "score_ratio": 0.5,
        "candidate_snapshot_cleanup_confirmed": True,
    }

    async def install_cli(_self, sandbox):
        sandbox.claude_code_installed = True

    async def run_claude(_self, sandbox, *, session_id, adapter_url, **_kwargs):
        assert sandbox.claude_code_installed is True
        await sandbox.write_file("/app/.harness/trajectory.jsonl", '{"type":"result"}\n', user="root")
        async with aiohttp.ClientSession() as client:
            async with client.post(
                f"{adapter_url}/v1/messages",
                headers={"x-api-key": session_id},
                json={
                    "model": "slime-actor",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "solve the task"}],
                },
            ) as response:
                payload = await response.json()
        assert response.status == 200 and payload["type"] == "message"
        return 0

    async def run_case(monkeypatch: pytest.MonkeyPatch):
        _patch_rollout(monkeypatch, tmp_path, grade)

        async def grade_with_exclusive_root(*, paths, **_kwargs):
            Path(paths["root"]).mkdir(parents=True, exist_ok=False)
            return dict(grade)

        monkeypatch.setattr(runtime, "grade_sandbox", grade_with_exclusive_root)
        monkeypatch.setattr(
            generate,
            "CONFIG",
            dataclasses.replace(
                generate.CONFIG,
                agent_name="claude_code",
                cc_transport="direct",
                adapter_public_host="127.0.0.1",
                adapter_bind_host="127.0.0.1",
            ),
        )
        monkeypatch.setattr(type(generate._CLAUDE_CODE), "install_cli", install_cli)
        monkeypatch.setattr(type(generate._CLAUDE_CODE), "run", run_claude)
        SingletonMeta.clear_instances(generate._AdapterService)
        samples = await generate.generate(_args(), _sample(package), {"max_tokens": 64})
        assert samples and sum(float(sample.reward) for sample in samples) == pytest.approx(0.5)
        assert (tmp_path / "work" / "claude-code.jsonl").read_text() == '{"type":"result"}\n'
        assert all(sample.metadata["agent_harness"] == "claude_code" for sample in samples)
        assert all(sample.metadata["agent_exit_code"] == 0 for sample in samples)

    with pytest.MonkeyPatch.context() as monkeypatch:
        asyncio.run(run_case(monkeypatch))


def test_terminal_claude_code_runs_as_root(monkeypatch):
    captured = {}

    async def launched(env):
        captured.update(env)
        return 0

    async def run_case():
        sandbox = FakeSandbox(on_launch=launched)
        harness = RootClaudeCodeHarness()
        await harness.write_config(
            sandbox,
            SimpleNamespace(workdir="/app", session_id="session-root", adapter_url="http://adapter:18001", model_label="slime-actor"),
        )
        result = await harness.launch_and_wait(
            sandbox,
            SimpleNamespace(workdir="/app", session_id="session-root", adapter_url="http://adapter:18001", model_label="slime-actor"),
            "solve",
            30,
        )
        assert result == 0
        assert any("/root/.claude/settings.json" in command and user == "root" for command, user in sandbox.exec_log)
        assert any("chown root:root /app/.harness" in command for command, _user in sandbox.exec_log)
        launcher = next(value for path, value in sandbox.files.items() if path.endswith(".run.sh"))
        assert "export HOME=/root" in launcher
        assert "--dangerously-skip-permissions" in launcher
        assert "--allow-dangerously-skip-permissions" in launcher
        assert "--permission-mode bypassPermissions" not in launcher
        assert captured["ANTHROPIC_BASE_URL"] == "http://adapter:18001"
        assert captured["ANTHROPIC_AUTH_TOKEN"] == "session-root"
        assert captured["IS_SANDBOX"] == "1"

    monkeypatch.setattr(sandbox_module.asyncio, "sleep", _fast_sleep)
    asyncio.run(run_case())


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


def test_terminal_aborted_retry_uses_fresh_session_not_closed_tombstone(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    grade = {
        "ok": False,
        "verdict": "INDETERMINATE",
        "reason_code": "sandbox_infra",
        "candidate_snapshot_cleanup_confirmed": True,
    }

    async def exercise(monkeypatch):
        tokenizer = _patch_rollout(monkeypatch, tmp_path, grade)
        monkeypatch.setattr(
            adapters_common,
            "call_sglang_generate",
            fake_call_sglang_generate([("first try", "stop", None), ("second try", "stop", None)], tokenizer),
        )
        base = _sample(package)
        failed = await generate.generate(_args(), base, {"max_new_tokens": 64})
        first_sid = base.session_id
        state = generate._AdapterService(_args())
        assert first_sid in state.adapter.closed
        assert failed[0].status == Sample.Status.ABORTED
        grade.update(ok=True, verdict="NON_EQUAL", score_ratio=0.5)
        retried = await generate.generate(_args(), failed[0], {"max_new_tokens": 64})
        assert base.session_id != first_sid
        assert first_sid in state.adapter.closed  # 不能为了重试撤销旧会话的迟到请求保护。
        assert all(s.status == Sample.Status.COMPLETED and not s.remove_sample for s in retried)
        assert sum(s.reward for s in retried) == pytest.approx(0.5)
        assert all(sum(s.loss_mask) > 0 for s in retried)
        assert base.index == 3 and base.group_index == 2
        assert base.prompt == "Repair the service and leave it running."

    with pytest.MonkeyPatch.context() as monkeypatch:
        asyncio.run(exercise(monkeypatch))


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


def test_terminal_bash_tool_returns_bounded_command_observation() -> None:
    async def run_case():
        sandbox = FakeSandbox(responses=[("printf demo", (7, "hello", "warning"))])
        agent = TerminalToolAgent()
        result = await agent._run_bash(
            sandbox,
            {
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": "printf demo", "timeout": 10}),
                }
            },
        )
        assert "exit_code=7" in result
        assert "hello" in result
        assert "warning" in result
        assert sandbox.exec_log == [("cd /app\nprintf demo", "root")]

    asyncio.run(run_case())


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
    assert "reference_dir" not in row["metadata"]
    assert "SECRET REFERENCE" not in json.dumps(row)


def test_e2b_credentials_load_from_protected_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "NEXTASK_BENCH_E2B_API_KEY",
        "NEXTASK_SOLVE_E2B_API_KEY",
        "NEXTASK_GEN_E2B_API_KEY",
        "E2B_API_KEY",
        "E2B_API_KEY_CODEX",
        "E2B_API_URL",
        "E2B_DOMAIN",
    ):
        monkeypatch.delenv(name, raising=False)
    config = tmp_path / "e2b.json"
    config.write_text(
        json.dumps({"api_key": "secret", "api_url": "https://e2b.example", "domain": "e2b.example"}),
        encoding="utf-8",
    )
    config.chmod(0o600)
    monkeypatch.setenv("TERMINAL_RL_E2B_CONFIG_FILE", str(config))
    assert resolve_e2b_connection() == ("secret", "https://e2b.example", "e2b.example")

    config.chmod(0o644)
    with pytest.raises(RuntimeError, match="permissions"):
        resolve_e2b_connection()


def test_nextask_sandbox_requires_confirmed_live_cleanup() -> None:
    class FakeRemote:
        def __init__(self, error: Exception | None = None) -> None:
            self.error = error

        async def kill(self):
            if self.error is not None:
                raise self.error
            return False  # already absent is a confirmed cleanup state

    async def run_case():
        sandbox = NextaskE2BSandbox("template")
        sandbox._sb = FakeRemote()
        sandbox.sandbox_id = "sandbox-1"
        await sandbox.__aexit__(None, None, None)
        assert sandbox.cleanup_confirmed is True

        failed = NextaskE2BSandbox("template")
        failed._sb = FakeRemote(RuntimeError("control plane unavailable"))
        failed.sandbox_id = "sandbox-2"
        failed.rpc_retries = 1
        with pytest.raises(RuntimeError, match="cleanup was not confirmed"):
            await failed.__aexit__(None, None, None)
        assert failed.cleanup_confirmed is False

    asyncio.run(run_case())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
