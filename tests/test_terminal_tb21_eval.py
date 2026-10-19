"""TB2.1只读测试集、评测输出与启动参数的CPU回归。"""

import ast
import asyncio
import copy
import hashlib
import json
import logging
import math
import os
import signal
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.terminal_agent_rl import tb21_dataset, tb21_eval, tb21_queue, tb21_timeouts, tb21_worker  # noqa: E402
from examples.terminal_agent_rl.agent import BASH_TOOL, SYSTEM_PROMPT, TerminalToolAgent  # noqa: E402
from slime.utils.eval_config import build_eval_dataset_configs  # noqa: E402
from slime.utils.misc import should_run_periodic_action  # noqa: E402

NUM_GPUS = 0


@pytest.fixture
def dataset(tmp_path):
    tasks, rows, identities = {}, [], []
    for index in range(89):
        name = f"task-{index}"
        root = tmp_path / "tasks" / name
        for filename, text in {
            "instruction.md": f"Public instruction {index}\n",
            "task.toml": 'version = "1.0"\n',
            "environment/Dockerfile": "FROM ubuntu:24.04\n",
            "tests/test.sh": "private-test-only",
            "solution/solve.sh": "private-answer-only",
        }.items():
            path = root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        digest = "sha256:" + f"{index:064x}"
        tasks[name] = {"digest": digest, "files": tb21_dataset.file_hashes(root)}
        identities.append({"org": "terminal-bench", "name": name, "ref": digest})
        rows.append(
            {
                "prompt": f"Public instruction {index}",
                "label": name,
                "metadata": {
                    "dataset_schema": tb21_dataset.SCHEMA,
                    "task_name": name,
                    "task_digest": digest,
                },
            }
        )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": tb21_dataset.SCHEMA,
                "dataset": tb21_dataset.DATASET,
                "version": tb21_dataset.VERSION,
                "tasks": tasks,
                "registry_metadata": {
                    "name": tb21_dataset.DATASET,
                    "version": tb21_dataset.VERSION,
                    "task_ids": identities,
                },
            }
        )
    )
    path = tmp_path / "tb21.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_exact_full_dataset_and_public_prompt(dataset):
    rows = tb21_dataset.validate_dataset(dataset)
    assert len(rows) == 89
    assert "private-test-only" not in dataset.read_text()
    assert "private-answer-only" not in dataset.read_text()


@pytest.mark.parametrize("change", ["subset", "duplicate", "prompt", "metadata", "private-file", "symlink", "registry"])
def test_dataset_drift_is_rejected(dataset, change):
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    if change == "subset":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0]
    elif change == "prompt":
        rows[0]["prompt"] += " private answer"
    elif change == "metadata":
        rows[0]["metadata"]["task_digest"] = "sha256:" + "f" * 64
    elif change == "private-file":
        (dataset.parent / "tasks/task-0/tests/test.sh").write_text("changed")
    elif change == "symlink":
        (dataset.parent / "tasks/task-0/link").symlink_to("instruction.md")
    elif change == "registry":
        path = dataset.with_name("manifest.json")
        manifest = json.loads(path.read_text())
        manifest["registry_metadata"]["version"] = "latest"
        path.write_text(json.dumps(manifest))
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        tb21_dataset.validate_dataset(dataset)


@pytest.mark.parametrize("value", [0, 1, 0.0, 1.0])
def test_only_binary_reward_with_cleanup(value):
    assert tb21_eval.validate_receipt({"ok": True, "reward": value, "cleanup": {"live_sandbox_cleanup_confirmed": True}}) == value


@pytest.mark.parametrize(
    "patch",
    [
        {"ok": False},
        {"reward": None},
        {"reward": True},
        {"reward": 0.5},
        {"reward": float("nan")},
        {"cleanup": {}},
    ],
)
def test_invalid_trial_is_not_zero_or_omitted(patch):
    receipt = {"ok": True, "reward": 1, "cleanup": {"live_sandbox_cleanup_confirmed": True}}
    receipt.update(patch)
    with pytest.raises((RuntimeError, ValueError)):
        tb21_eval.validate_receipt(receipt)


def test_eval_config_has_one_attempt_no_train_hook():
    raw = yaml.safe_load((ROOT / "examples/terminal_agent_rl/tb21_eval.yaml").read_text())["eval"]
    configs = build_eval_dataset_configs(SimpleNamespace(), raw["datasets"], defaults=raw["defaults"])
    assert len(configs) == 1
    config = configs[0]
    assert config.name == "tb21" and config.n_samples_per_eval_prompt == 1
    assert (config.temperature, config.top_p, config.top_k) == (1.0, 0.95, 20)
    assert config.custom_generate_function_path is None
    assert config.max_response_len == 65536


@pytest.mark.parametrize("top_p,expected", [(0.95, {"return_top_p_token_ids": True}), (1.0, None)])
def test_tb21_trial_requests_real_top_p_replay_metadata(tmp_path, monkeypatch, top_p, expected):
    opened = []

    class Adapter:
        def open_session(self, session_id, **kwargs):
            opened.append(kwargs["sampling_defaults"])

        async def drop_session(self, session_id, wait_timeout):
            pass

    class Process:
        returncode = 0

        async def wait(self):
            return 0

    async def launch(*args, **kwargs):
        receipt = Path(args[args.index("--receipt") + 1])
        receipt.write_text(json.dumps({"ok": True, "reward": 0, "cleanup": {"live_sandbox_cleanup_confirmed": True}}))
        return Process()

    monkeypatch.setattr(tb21_eval.asyncio, "create_subprocess_exec", launch)
    monkeypatch.setenv("TB21_RUNTIME_PYTHON", sys.executable)
    config = SimpleNamespace(
        path=str(tmp_path / "tb21.jsonl"),
        temperature=0.7,
        top_p=top_p,
        top_k=40,
        max_response_len=8192,
        stop_token_ids=[248046, 248044],
    )
    state = SimpleNamespace(adapter=Adapter(), adapter_url="http://127.0.0.1:1", max_context_len=262144)
    reward = asyncio.run(tb21_eval._trial({"label": "task"}, config, tmp_path / "trial", state))
    assert reward == 0
    assert opened[0].get("custom_params") == expected


def test_eval_hook_cannot_be_used_for_training():
    with pytest.raises(ValueError, match="eval-only"):
        tb21_eval.evaluate(SimpleNamespace(), 0, None, evaluation=False)


def test_tb21_timeout_defaults_are_six_hour_agent_and_nine_hour_trial(monkeypatch):
    monkeypatch.delenv("TB21_AGENT_TIME_BUDGET_SEC", raising=False)
    monkeypatch.delenv("TB21_TRIAL_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("TB21_MAX_TURNS", raising=False)
    monkeypatch.delenv("TB21_REASONING_EFFORT", raising=False)
    assert tb21_timeouts.agent_time_budget_sec() == 21600
    assert tb21_timeouts.harbor_agent_timeout_sec() == 21660
    assert tb21_timeouts.trial_timeout_sec() == 32400
    assert tb21_timeouts.max_turns() == 100000
    assert tb21_timeouts.reasoning_effort() == "high"
    assert tb21_timeouts.AGENT_SETUP_TIMEOUT_SEC == 3600
    assert tb21_timeouts.VERIFIER_TIMEOUT_SEC == 1800


def test_tb21_timeout_override_must_leave_room_for_harbor_cleanup(monkeypatch):
    monkeypatch.setenv("TB21_AGENT_TIME_BUDGET_SEC", "99")
    monkeypatch.setenv("TB21_TRIAL_TIMEOUT_SEC", "159")
    with pytest.raises(ValueError, match="must exceed"):
        tb21_timeouts.trial_timeout_sec()
    monkeypatch.setenv("TB21_TRIAL_TIMEOUT_SEC", "160")
    assert tb21_timeouts.trial_timeout_sec() == 160


def test_tb21_agent_settings_reject_low_or_unsupported_overrides(monkeypatch):
    monkeypatch.setenv("TB21_MAX_TURNS", "0")
    with pytest.raises(ValueError, match="positive integer"):
        tb21_timeouts.max_turns()
    monkeypatch.setenv("TB21_REASONING_EFFORT", "medium")
    with pytest.raises(ValueError, match="must be high"):
        tb21_timeouts.reasoning_effort()


def test_epoch_boundary_does_not_trigger_extra_tb21_run():
    output = tb21_eval.evaluate(SimpleNamespace(eval_interval=5), 16, None, evaluation=True)
    assert output.data == {}


def test_phase_intervals_and_final_save():
    assert [i + 1 for i in range(3) if should_run_periodic_action(i, 1, None, 3)] == [1, 2, 3]
    assert [i + 1 for i in range(12) if should_run_periodic_action(i, 5, None)] == [5, 10]


def test_training_agent_defaults_unchanged_and_tb_task_cwd():
    assert TerminalToolAgent().system_prompt == SYSTEM_PROMPT
    assert TerminalToolAgent().bash_tool == BASH_TOOL
    original = copy.deepcopy(BASH_TOOL)
    agent = TerminalToolAgent(workdir="/workspace/test dir")
    commands = []

    class Environment:
        async def exec(self, command, **kwargs):
            commands.append(command)
            return 0, "ok", ""

    asyncio.run(agent._run_bash(Environment(), {"function": {"arguments": '{"command":"pwd"}'}}))
    assert commands == ["cd '/workspace/test dir'\npwd"]
    assert "/workspace/test dir" in agent.system_prompt
    assert BASH_TOOL == original


@pytest.mark.parametrize("failed", [False, True])
def test_evaluation_keeps_full_denominator_and_surfaces_infra_failure(dataset, tmp_path, monkeypatch, failed):
    fake = types.ModuleType("examples.terminal_agent_rl.generate")
    fake._AdapterService = lambda args: object()
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setenv("TB21_EVAL_OUTPUT", str(tmp_path / "eval"))
    called = []

    async def trial(row, config, output, state):
        called.append(row["label"])
        if failed and row["label"] == "task-0":
            raise RuntimeError("infrastructure failed")
        return 1.0 if row["label"] == "task-1" else 0.0

    monkeypatch.setattr(tb21_eval, "_trial", trial)
    coroutine = tb21_eval._evaluate(None, 4, SimpleNamespace(path=str(dataset)))
    if failed:
        with pytest.raises(RuntimeError, match="incomplete"):
            asyncio.run(coroutine)
    else:
        rewards = asyncio.run(coroutine)
        assert len(rewards) == 89 and sum(rewards) == 1
    assert len(called) == 89
    receipt = json.loads((tmp_path / "eval/rollout-000004/summary.json").read_text())
    assert receipt["ok"] is not failed
    assert receipt["failed_tasks"] == (["task-0"] if failed else [])


def test_tensorboard_native_log_roundtrip(tmp_path, monkeypatch):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from slime.observability import logging_utils
    from slime.observability.tensorboard_utils import _TensorboardAdapter
    from slime.utils.misc import SingletonMeta

    monkeypatch.setenv("TENSORBOARD_DIR", str(tmp_path))
    monkeypatch.delitem(SingletonMeta._instances, _TensorboardAdapter, raising=False)
    args = SimpleNamespace(use_tensorboard=True, use_wandb=False, tb_project_name="test", tb_experiment_name="test")
    logging_utils.log(args, {"train/step": 4, "train/loss": 0.125, "train/grad_norm": 0.5}, "train/step")
    logging_utils.log(args, {"eval/step": 5, "eval/tb21": 1 / 89, "eval/tb21/task_count": 89}, "eval/step")
    _TensorboardAdapter(args).finish()
    events = EventAccumulator(str(tmp_path)).Reload()
    assert events.Scalars("train/loss")[0].value == 0.125
    assert events.Scalars("eval/tb21/task_count")[0].value == 89
    assert events.Scalars("eval/tb21")[0].step == 5
    assert "train/step" not in events.Tags()["scalars"]
    SingletonMeta._instances.pop(_TensorboardAdapter)


@pytest.mark.parametrize("phase,save,num", [("smoke", "1", "3"), ("long", "5", "100")])
@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("variant", ["baseline", "rollout8", "pack16k", "deepep"])
def test_actual_shell_arguments_and_ray_tensorboard_env(tmp_path, phase, save, num, resume, variant):
    executable = sys.executable
    mockbin = tmp_path / "bin"
    mockbin.mkdir()
    for name, body in {
        "python3": "import sys\nif sys.argv[1:]==['-']:\n code=sys.stdin.read()\n if '\"env_vars\"' in code: exec(code)\n",
        "git": "print('expected')\n",
        "ray": "import json,sys\nif sys.argv[1]=='job': print(json.dumps(sys.argv[1:]))\n",
    }.items():
        target = mockbin / name
        target.write_text(f"#!{executable}\n" + body)
        target.chmod(0o755)
    hf = tmp_path / "hf"
    hf.mkdir()
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "latest_checkpointed_iteration.txt").write_text("release")
    for name in ("model.safetensors.index.json", "chat_template.jinja"):
        (hf / name).touch()
    vendor = tmp_path / "vendor/e2b"
    vendor.mkdir(parents=True)
    (vendor / "__init__.py").touch()
    config = tmp_path / "secret.json"
    config.write_text("{}")
    config.chmod(0o600)
    data = tmp_path / "train.jsonl"
    data.touch()
    node_tarball = tmp_path / "node.tar.gz"
    cc_tarball = tmp_path / "claude-code.tgz"
    node_tarball.write_bytes(b"fixed-node")
    cc_tarball.write_bytes(b"fixed-claude-code")
    env = dict(
        os.environ,
        PATH=f"{mockbin}:{os.environ['PATH']}",
        TRAINING_PHASE=phase,
        HF_CHECKPOINT=str(hf),
        REF_MODEL_PATH=str(ref),
        PROMPT_DATA=str(data),
        NEXTASK_RUNTIME_PYTHON=str(mockbin / "python3"),
        TB21_RUNTIME_PYTHON=str(mockbin / "python3"),
        EXPECTED_NEXTASK_COMMIT="expected",
        EXPECTED_NEXAU_COMMIT="expected",
        E2B_VENDOR_DIR=str(vendor.parent),
        TERMINAL_RL_E2B_CONFIG_FILE=str(config),
        MASTER_ADDR="127.0.0.1",
        RUN_ROOT=str(tmp_path / "run"),
        LOAD_PATH=str(ref) if resume else "",
        TERMINAL_RL_AGENT="claude_code",
        SLIME_AGENT_NODE_TARBALL=str(node_tarball),
        SLIME_AGENT_CC_TARBALL=str(cc_tarball),
        EXPECTED_NODE_TARBALL_SHA256=hashlib.sha256(node_tarball.read_bytes()).hexdigest(),
        EXPECTED_CC_TARBALL_SHA256=hashlib.sha256(cc_tarball.read_bytes()).hexdigest(),
    )
    for variable in (
        "SAVE_INTERVAL",
        "NUM_ROLLOUT",
        "EVAL_INTERVAL",
        "USE_TENSORBOARD",
        "ENABLE_TB21_EVAL",
        "TENSORBOARD_DIR",
        "TB21_EVAL_MODE",
        "MOE_DISPATCHER",
        "MAX_GEN_LEN",
        "TERMINAL_RL_AGENT_TIME_BUDGET_SEC",
        "TERMINAL_RL_CC_TRANSPORT",
        "TERMINAL_RL_CC_AUTO_COMPACT_WINDOW",
        "TERMINAL_RL_CC_COUNT_EXACT_MARGIN",
        "TERMINAL_RL_MAX_TURNS",
        "TERMINAL_RL_REASONING_EFFORT",
        "TB21_MAX_TURNS",
        "TB21_REASONING_EFFORT",
    ):
        env.pop(variable, None)
    env.update(SGLANG_SERVER_CONCURRENCY="8" if variant == "rollout8" else "4", MAX_TOKENS_PER_GPU="16384" if variant == "pack16k" else "8192", MOE_DISPATCHER="deepep" if variant == "deepep" else "alltoall")
    result = subprocess.run(["bash", str(ROOT / "examples/terminal_agent_rl/run_nex_n25_mini.sh")], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout.splitlines()[-1])
    assert argv[argv.index("--save-interval") + 1] == save
    assert argv[argv.index("--num-rollout") + 1] == num
    assert argv[argv.index("--eval-interval") + 1] == "5"
    assert "--use-tensorboard" in argv and "--skip-eval-before-train" in argv
    assert "--accumulate-allreduce-grads-in-fp32" in argv and "--optimizer-cpu-offload" in argv
    assert argv[argv.index("--n-samples-per-prompt") + 1] == "16"
    assert argv[argv.index("--global-batch-size") + 1] == "128"
    assert ("--load" in argv) is resume
    assert ("--use-checkpoint-opt-param-scheduler" in argv) is resume
    assert not {"--finetune", "--no-load-optim", "--no-load-rng", "--override-opt-param-scheduler"} & set(argv)
    assert argv[argv.index("--eval-function-path") + 1] == "examples.terminal_agent_rl.tb21_queue.enqueue"
    runtime = json.loads(next(a.split("=", 1)[1] for a in argv if a.startswith("--runtime-env-json=")))
    assert runtime["env_vars"]["TENSORBOARD_DIR"] == str(tmp_path / "run/tensorboard")
    assert runtime["env_vars"]["TB21_EVAL_OUTPUT"] == str(tmp_path / "run/tb21-eval")
    assert "TB21_EVAL_QUEUE" in runtime["env_vars"]
    assert runtime["env_vars"]["TERMINAL_RL_AGENT_TIME_BUDGET_SEC"] == "14400"
    assert runtime["env_vars"]["TERMINAL_RL_MAX_TURNS"] == "400"
    assert runtime["env_vars"]["TERMINAL_RL_REASONING_EFFORT"] == "high"
    assert runtime["env_vars"]["TERMINAL_RL_AGENT"] == "claude_code"
    assert runtime["env_vars"]["TERMINAL_RL_CC_TRANSPORT"] == "e2b_pull"
    assert runtime["env_vars"]["TERMINAL_RL_CC_AUTO_COMPACT_WINDOW"] == "196608"
    assert runtime["env_vars"]["TERMINAL_RL_CC_COUNT_EXACT_MARGIN"] == "32768"
    assert runtime["env_vars"]["TERMINAL_RL_KEEP_VERIFIER_ARTIFACTS"] == "0"
    assert runtime["env_vars"]["ADAPTER_BIND_HOST"] == "127.0.0.1"
    assert "ADAPTER_PUBLIC_HOST" not in runtime["env_vars"]
    assert runtime["env_vars"]["SLIME_AGENT_NODE_TARBALL"] == str(node_tarball)
    assert runtime["env_vars"]["SLIME_AGENT_CC_TARBALL"] == str(cc_tarball)
    assert "autoCompactWindow" in runtime["env_vars"]["SLIME_AGENT_CC_EXTRA_ARGS"]
    assert "--agents" not in runtime["env_vars"]["SLIME_AGENT_CC_EXTRA_ARGS"]
    assert "worker" not in runtime["env_vars"]["SLIME_AGENT_CC_EXTRA_ARGS"]
    assert runtime["env_vars"]["TB21_AGENT_TIME_BUDGET_SEC"] == "21600"
    assert runtime["env_vars"]["TB21_TRIAL_TIMEOUT_SEC"] == "32400"
    assert runtime["env_vars"]["TB21_MAX_TURNS"] == "100000"
    assert runtime["env_vars"]["TB21_REASONING_EFFORT"] == "high"
    assert argv[argv.index("--rollout-max-response-len") + 1] == "32768"
    assert argv[argv.index("--rollout-temperature") + 1] == "1.0"
    assert argv[argv.index("--rollout-top-p") + 1] == "0.95"
    assert argv[argv.index("--rollout-top-k") + 1] == "40"
    assert argv[argv.index("--sglang-server-concurrency") + 1] == env["SGLANG_SERVER_CONCURRENCY"]
    assert argv[argv.index("--max-tokens-per-gpu") + 1] == env["MAX_TOKENS_PER_GPU"]
    dispatchers = [argv[i + 1] for i, value in enumerate(argv) if value == "--moe-token-dispatcher-type"]
    assert dispatchers[-1] == ("flex" if variant == "deepep" else "alltoall")
    assert ("--moe-enable-deepep" in argv) is (variant == "deepep")
    if variant == "deepep":
        assert argv[argv.index("--moe-flex-dispatcher-backend") + 1] == "deepep"


@pytest.mark.parametrize("name,value,reason", [("MOE_DISPATCHER", "typo", "MOE_DISPATCHER"), ("TB21_EVAL_MODE", "typo", "TB21_EVAL_MODE"), ("SAVE_INTERVAL", "3", "整除")])
def test_background_and_performance_shell_reject_invalid_options(name, value, reason):
    env = dict(os.environ, USE_TENSORBOARD="0", ENABLE_TB21_EVAL="1", TB21_EVAL_MODE="async", EVAL_INTERVAL="5", SAVE_INTERVAL="5", MOE_DISPATCHER="alltoall")
    env[name] = value
    result = subprocess.run(["bash", str(ROOT / "examples/terminal_agent_rl/run_nex_n25_mini.sh")], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and reason in result.stderr


@pytest.fixture
def queue_args(dataset, tmp_path, monkeypatch):
    from slime.utils.eval_config import EvalDatasetConfig

    monkeypatch.setenv("TB21_EVAL_QUEUE", str(tmp_path / "queue"))
    monkeypatch.setenv("TENSORBOARD_DIR", str(tmp_path / "tensorboard"))
    return SimpleNamespace(
        save=str(tmp_path / "checkpoints"),
        hf_checkpoint=str(tmp_path / "base"),
        rollout_max_context_len=262144,
        sglang_tool_call_parser="qwen3_coder",
        sglang_reasoning_parser="qwen3",
        eval_interval=5,
        eval_datasets=[EvalDatasetConfig(name="tb21", path=str(dataset), n_samples_per_eval_prompt=1)],
    )


def test_background_hook_only_enqueues_fixed_checkpoint_and_deduplicates(queue_args, tmp_path):
    args = queue_args
    assert tb21_queue.enqueue(args, 2, None, evaluation=True).data == {}
    for _ in range(2):
        result = tb21_queue.enqueue(args, 4, None, evaluation=True)
        assert result.data == {} and result.metrics["eval/tb21/queued"] == 1
    requests = list((tmp_path / "queue/requests").glob("*.json"))
    assert len(requests) == 1
    payload = json.loads(requests[0].read_text())
    assert payload["checkpoint_dir"] == str(tmp_path / "checkpoints/iter_0000004")
    assert payload["optimizer_step"] == 5 and "sglang_router_ip" not in payload
    assert not (tmp_path / "checkpoints").exists()
    tb21_queue.validate_request(payload)


def test_background_enqueue_io_failure_does_not_fail_training_or_emit_score(queue_args, monkeypatch):
    def fail(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(tb21_queue, "atomic_json", fail)
    result = tb21_queue.enqueue(queue_args, 4, None, evaluation=True)
    assert result.metrics == {"eval/tb21/queued": 0, "eval/tb21/enqueue_failed": 1}
    assert result.data == {}
    with pytest.raises(ValueError, match="eval-only"):
        tb21_queue.enqueue(queue_args, 4, None, evaluation=False)


def test_background_request_immutability_and_dataset_drift(queue_args, tmp_path, monkeypatch):
    tb21_queue.enqueue(queue_args, 4, None, evaluation=True)
    monkeypatch.setenv("TERMINAL_RL_AGENT_TIME_BUDGET_SEC", "99")
    result = tb21_queue.enqueue(queue_args, 4, None, evaluation=True)
    assert result.metrics["eval/tb21/enqueue_failed"] == 1
    path = next((tmp_path / "queue/requests").glob("*.json"))
    request = json.loads(path.read_text())
    Path(queue_args.eval_datasets[0].path).write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        tb21_queue.validate_request(request)


def test_background_checkpoint_waits_for_published_exact_iteration(queue_args, tmp_path):
    request = tb21_queue.build_request(queue_args, 4, queue_args.eval_datasets[0])
    directory = Path(request["checkpoint_dir"])
    directory.mkdir(parents=True)
    for name in (".metadata", "common.pt", "__0_0.distcp"):
        (directory / name).touch()
    assert not tb21_queue.checkpoint_published(request)
    tracker = directory.parent / "latest_checkpointed_iteration.txt"
    tracker.write_text("3")
    assert not tb21_queue.checkpoint_published(request)
    tracker.write_text("9")
    assert tb21_queue.checkpoint_published(request)  # 只取固定iter4，不跟随latest变成iter9。
    request["checkpoint_dir"] = str(directory.parent / "iter_0000009")
    with pytest.raises(ValueError, match="exact"):
        tb21_queue.checkpoint_published(request)


@pytest.mark.parametrize("failed", [False, True])
def test_background_worker_is_once_per_request_and_failure_is_not_score(queue_args, tmp_path, monkeypatch, failed):
    tb21_queue.enqueue(queue_args, 4, None, evaluation=True)
    monkeypatch.setattr(tb21_worker, "checkpoint_published", lambda request: True)
    called, recorded = [], []

    def evaluate(request, output):
        called.append(request["optimizer_step"])
        if failed:
            raise RuntimeError("infra failed")
        return {"ok": True, "success_rate": 1 / 89, "optimizer_step": 5}

    monkeypatch.setattr(tb21_worker, "run_evaluation", evaluate)
    monkeypatch.setattr(tb21_worker, "record_result", lambda req, result, output: recorded.append(result))
    assert tb21_worker.consume_one(tmp_path / "queue")
    assert not tb21_worker.consume_one(tmp_path / "queue")
    assert called == [5]
    status = json.loads(next((tmp_path / "queue/jobs").glob("*/status.json")).read_text())
    assert status["status"] == ("failed" if failed else "completed")
    assert recorded[0]["ok"] is not failed
    if failed:
        assert "success_rate" not in recorded[0]


def test_background_four_engines_enforce_total20_and_per_engine5(dataset, tmp_path, monkeypatch):
    monkeypatch.setenv("TB21_EVAL_OUTPUT", str(tmp_path / "pool-eval"))
    monkeypatch.setenv("TB21_CONCURRENCY", "20")
    states = [SimpleNamespace(index=i) for i in range(4)]
    running, peaks, calls = [0] * 4, [0] * 4, []

    async def trial(row, config, output, state):
        i = state.index
        running[i] += 1
        peaks[i] = max(peaks[i], running[i])
        calls.append(row["label"])
        await asyncio.sleep(0.001)
        running[i] -= 1
        return 0.0

    monkeypatch.setattr(tb21_eval, "_trial", trial)
    result = asyncio.run(tb21_eval._evaluate(None, 4, SimpleNamespace(path=str(dataset)), states=states))
    assert peaks == [5] * 4 and len(set(calls)) == len(result) == 89


def test_background_server_uses_fixed_model_and_five_requests_per_tp2():
    argv = tb21_worker.server_command(Path("/snapshot/step5"), 31001, 262144)
    assert argv[argv.index("--model-path") + 1] == "/snapshot/step5"
    assert argv[argv.index("--tp-size") + 1] == "2"
    assert argv[argv.index("--max-running-requests") + 1] == "5"


def test_background_cleans_owned_child_groups_even_if_leader_exited(monkeypatch):
    signals = []
    monkeypatch.setattr(tb21_worker.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    process = SimpleNamespace(pid=12345, wait=lambda **kwargs: 1)
    tb21_worker.stop_servers([process])
    assert signals == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]


@pytest.mark.parametrize("ok", [True, False])
def test_background_tensorboard_uses_checkpoint_step_not_current_train_step(queue_args, tmp_path, ok):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    request = tb21_queue.build_request(queue_args, 4, queue_args.eval_datasets[0])
    result = {"ok": ok, "elapsed_sec": 100, "completed_at_unix": 200}
    if ok:
        result["success_rate"] = 1 / 89
    tb21_queue.record_result(request, result, tmp_path / "result")
    events = EventAccumulator(request["tensorboard_dir"]).Reload()
    assert events.Scalars("eval/tb21/completed")[0].step == 5
    assert ("eval/tb21" in events.Tags()["scalars"]) is ok


def test_background_snapshot_preserves_checkpoint_text_and_frozen_vision(tmp_path, monkeypatch):
    import safetensors.torch
    import torch
    import torch.distributed.checkpoint as dist_cp

    from examples.terminal_agent_rl.tb21_snapshot import export_snapshot

    # 真实GPU PYTHONPATH中Megatron/tools会遮蔽Slime的namespace tools。
    monkeypatch.setitem(sys.modules, "tools", types.ModuleType("tools"))

    origin = tmp_path / "base"
    origin.mkdir()
    weights = {
        "model.language_model.embed_tokens.weight": torch.zeros(4, 2, dtype=torch.bfloat16),
        "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.zeros(2, 4, 2, dtype=torch.bfloat16),
        "model.language_model.layers.0.mlp.experts.down_proj": torch.zeros(2, 2, 2, dtype=torch.bfloat16),
        "model.visual.frozen.weight": torch.ones(2, dtype=torch.bfloat16),
    }
    filename = "model-00001-of-00001.safetensors"
    safetensors.torch.save_file(weights, origin / filename)
    (origin / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: filename for k in weights}}))
    (origin / "config.json").write_text("{}")
    checkpoint = tmp_path / "iter_0000004"
    state = {
        "embedding.word_embeddings.weight": torch.full((4, 2), 0.25, dtype=torch.bfloat16),
        "decoder.layers.0.mlp.experts.experts.linear_fc1.weight": torch.full((2, 4, 2), 0.5, dtype=torch.bfloat16),
        "decoder.layers.0.mlp.experts.experts.linear_fc2.weight": torch.full((2, 2, 2), 0.75, dtype=torch.bfloat16),
    }
    dist_cp.save(state, checkpoint_id=checkpoint)
    args = SimpleNamespace(vocab_size=4, q_lora_rank=None, kv_channels=2, hidden_size=2, num_attention_heads=1, num_query_groups=1)
    torch.save({"iteration": 4, "args": args}, checkpoint / "common.pt")
    output = tmp_path / "snapshot"
    export_snapshot(checkpoint, origin, output, 5)
    result = safetensors.torch.load_file(output / filename)
    assert torch.equal(result["model.language_model.embed_tokens.weight"], state["embedding.word_embeddings.weight"])
    assert torch.equal(result["model.language_model.layers.0.mlp.experts.gate_up_proj"], state["decoder.layers.0.mlp.experts.experts.linear_fc1.weight"])
    assert torch.equal(result["model.visual.frozen.weight"], weights["model.visual.frozen.weight"])
    assert json.loads((output / "snapshot-receipt.json").read_text())["ok"]
    with pytest.raises(FileExistsError):
        export_snapshot(checkpoint, origin, output, 5)
    state.pop("embedding.word_embeddings.weight")
    missing = tmp_path / "missing"
    dist_cp.save(state, checkpoint_id=missing)
    torch.save({"iteration": 4, "args": args}, missing / "common.pt")
    with pytest.raises(ValueError, match="incomplete"):
        export_snapshot(missing, origin, tmp_path / "rejected", 5)


@pytest.fixture
def native_scheduler_class():
    native = Path(os.environ.get("MEGATRON_ROOT", "/root/Megatron-LM"))
    if not native.exists():
        pytest.skip("Set MEGATRON_ROOT to the pinned Megatron checkout for native scheduler tests")
    relative = "megatron/core/optimizer_param_scheduler.py"
    source = subprocess.check_output(["git", "-C", str(native), "show", f"1dcf0dafa884ad52ffb243625717a3471643e087:{relative}"], text=True)
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "OptimizerParamScheduler")
    namespace = {"Optional": Optional, "math": math, "logging": logging, "logger": logging.getLogger(__name__), "log_single_rank": lambda *a, **k: None}
    exec(compile(ast.Module(body=[node], type_ignores=[]), relative, "exec"), namespace)
    return namespace["OptimizerParamScheduler"]


@pytest.mark.parametrize("new_total_samples", [512, 12800])
def test_native_scheduler_resume_preserves_progress_lr_and_optimizer(native_scheduler_class, new_total_samples):
    import torch

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-6, weight_decay=0.1)
    parameter.grad = torch.tensor([0.5])
    optimizer.step()
    optimizer_state = copy.deepcopy(optimizer.state_dict()["state"])
    model_value = parameter.detach().clone()

    def scheduler(total, restore):
        return native_scheduler_class(
            optimizer,
            init_lr=0.0,
            max_lr=1e-6,
            min_lr=0.0,
            lr_warmup_steps=0,
            lr_decay_steps=total,
            lr_decay_style="constant",
            start_wd=0.1,
            end_wd=0.1,
            wd_incr_steps=total,
            wd_incr_style="constant",
            use_checkpoint_opt_param_scheduler=restore,
            override_opt_param_scheduler=False,
        )

    original = scheduler(128, False)
    original.step(128)
    saved = original.state_dict()
    with pytest.raises(AssertionError, match="total number of iterations"):
        scheduler(new_total_samples, False).load_state_dict(saved)
    restored = scheduler(new_total_samples, True)
    restored.load_state_dict(saved)
    assert restored.num_steps == 128
    assert restored.lr_decay_steps == restored.wd_incr_steps == 128
    for expected in (256, 384, 512):
        restored.step(128)
        assert restored.num_steps == expected
        assert optimizer.param_groups[0]["lr"] == 1e-6
        assert optimizer.param_groups[0]["weight_decay"] == 0.1
    torch.testing.assert_close(parameter.detach(), model_value, rtol=0, atol=0)
    actual = optimizer.state_dict()["state"]
    for key, state in optimizer_state.items():
        for name, value in state.items():
            torch.testing.assert_close(actual[key][name], value, rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
