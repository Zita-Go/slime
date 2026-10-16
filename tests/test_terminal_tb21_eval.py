"""TB2.1只读测试集、评测输出与启动参数的CPU回归。"""

import ast
import asyncio
import copy
import json
import logging
import math
import os
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

from examples.terminal_agent_rl import tb21_dataset, tb21_eval  # noqa: E402
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
    assert (config.temperature, config.top_p, config.top_k) == (0.7, 0.95, 40)
    assert config.custom_generate_function_path is None
    assert config.max_response_len == 8192


def test_eval_hook_cannot_be_used_for_training():
    with pytest.raises(ValueError, match="eval-only"):
        tb21_eval.evaluate(SimpleNamespace(), 0, None, evaluation=False)


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
def test_actual_shell_arguments_and_ray_tensorboard_env(tmp_path, phase, save, num, resume):
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
    )
    for variable in ("SAVE_INTERVAL", "NUM_ROLLOUT", "EVAL_INTERVAL", "USE_TENSORBOARD", "ENABLE_TB21_EVAL", "TENSORBOARD_DIR"):
        env.pop(variable, None)
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
    assert argv[argv.index("--eval-function-path") + 1] == "examples.terminal_agent_rl.tb21_eval.evaluate"
    runtime = json.loads(next(a.split("=", 1)[1] for a in argv if a.startswith("--runtime-env-json=")))
    assert runtime["env_vars"]["TENSORBOARD_DIR"] == str(tmp_path / "run/tensorboard")
    assert runtime["env_vars"]["TB21_EVAL_OUTPUT"] == str(tmp_path / "run/tb21-eval")


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
