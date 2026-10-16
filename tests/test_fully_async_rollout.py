"""CPU unit tests for the fully-async rollout worker's queue contract.

The module docstring of ``slime.rollout.fully_async_rollout`` promises that the
worker's output queue "stays warm" across ``generate_rollout`` calls: each call
takes ``rollout_batch_size`` completed groups and leaves the rest queued.

Three behaviours are pinned here:

  1. ``_generate_rollout_async`` consumes exactly ``rollout_batch_size`` groups
     and leaves the surplus in the queue. (It used to drain the whole queue and
     slice — throwing away fully generated, reward-scored groups whose prompts
     had already been consumed from the data buffer.)
  2. The task done-callback never blocks. It runs on the event-loop thread, so
     a bounded queue that filled up would freeze every in-flight generation.
  3. Backpressure exists anyway: ``_loop`` stops pulling new prompts while a
     full pool of completed groups is already waiting to be consumed.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import types
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

# ``fully_async_rollout`` imports ``sglang_rollout``, which needs sglang_router
# and (transitively) transformers — both deliberately absent from the CPU CI
# env. The tests below never dial a server or touch a tokenizer, so stub the
# imports, same as tests/test_agent/test_agent_rollout_cpu.py.
try:
    import sglang_router  # noqa: F401
except ImportError:
    _router_stub = types.ModuleType("sglang_router")
    _router_stub.__version__ = "0.2.3"
    sys.modules["sglang_router"] = _router_stub
try:
    import transformers  # noqa: F401
except ImportError:
    _tf_stub = types.ModuleType("transformers")
    for _name in ("AutoProcessor", "AutoTokenizer", "PreTrainedTokenizerBase", "ProcessorMixin"):
        setattr(_tf_stub, _name, type(_name, (), {}))
    sys.modules["transformers"] = _tf_stub

import pytest

import slime.rollout.fully_async_rollout as fa
from slime.rollout import sglang_rollout
from slime.rollout.base_types import iter_samples
from slime.rollout.data_source import RolloutDataSourceWithBuffer, pop_first
from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.rollout.filter_hub.dynamic_sampling_filters import check_reward_nonzero_std
from slime.utils.staleness import compute_staleness_metrics, sample_staleness
from slime.utils.types import Sample
from slime.utils.async_utils import AsyncPacer

NUM_GPUS = 0


def test_custom_data_source_keeps_original_entrypoint(monkeypatch):
    args = SimpleNamespace()
    # A custom data source may have its own unrelated scheduler attribute.
    source = SimpleNamespace(scheduler=object())
    expected = object()

    async def generate(received_args, rollout_id, received_source):
        assert received_args is args and received_source is source
        assert rollout_id == 3
        return expected

    monkeypatch.setattr(fa, "_generate_rollout_async", generate)
    monkeypatch.setattr(fa, "run", asyncio.run)
    assert fa.generate_rollout_fully_async(args, 3, source) is expected


class _FakeGenerateState:
    def __init__(self, args):
        self.sampling_params = {}


class _FakeDataBuffer:
    """Finite fuel: one group per ``get_samples`` call until exhausted."""

    def __init__(self, groups):
        self._groups = deque(groups)
        self.requeued = []

    def get_samples(self, n):
        assert n == 1
        if not self._groups:
            return []
        return [self._groups.popleft()]

    def add_samples(self, groups):
        self.requeued.extend(groups)


def _make_group(index: int) -> list[Sample]:
    sample = Sample(index=index, prompt=f"p{index}")
    sample.status = Sample.Status.COMPLETED
    return [sample]


def _make_worker(monkeypatch, data_buffer=None, concurrency=4) -> fa.AsyncRolloutWorker:
    monkeypatch.setattr(fa, "GenerateState", _FakeGenerateState)
    args = SimpleNamespace(rollout_batch_size=4, n_samples_per_prompt=1)
    return fa.AsyncRolloutWorker(args, data_buffer or _FakeDataBuffer([]), concurrency=concurrency)


@pytest.mark.unit
def test_rollout_takes_target_groups_and_leaves_surplus_queued(monkeypatch):
    worker = _make_worker(monkeypatch)
    for gid in range(10):
        worker.output_queue.put((gid, _make_group(gid)))
    monkeypatch.setattr(fa, "_get_global_worker", lambda args, data_buffer: worker)

    args = SimpleNamespace(
        rollout_batch_size=4, rollout_data_transport="object-store", rollout_sample_filter_path=None
    )
    out = asyncio.run(fa._generate_rollout_async(args, rollout_id=0, data_buffer=None))

    assert len(out) == 4
    # FIFO: the oldest four groups ship first.
    assert [group[0].index for group in out] == [0, 1, 2, 3]
    # The other six are still queued for the next rollout, not thrown away.
    assert worker.queue_size() == 6
    assert [gid for gid, _ in worker.get_completed_groups()] == [4, 5, 6, 7, 8, 9]


@pytest.mark.unit
def test_get_completed_groups_limit(monkeypatch):
    worker = _make_worker(monkeypatch)
    for gid in range(5):
        worker.output_queue.put((gid, _make_group(gid)))

    assert [gid for gid, _ in worker.get_completed_groups(limit=2)] == [0, 1]
    assert [gid for gid, _ in worker.get_completed_groups()] == [2, 3, 4]
    assert worker.get_completed_groups(limit=3) == []


@pytest.mark.unit
def test_concurrency_is_scaled_by_samples_per_prompt(monkeypatch):
    monkeypatch.setattr(fa, "GenerateState", _FakeGenerateState)
    args = SimpleNamespace(n_samples_per_prompt=4)
    assert fa.AsyncRolloutWorker(args, _FakeDataBuffer([]), concurrency=10).concurrency == 2
    assert fa.AsyncRolloutWorker(args, _FakeDataBuffer([]), concurrency=2).concurrency == 1


@pytest.mark.unit
def test_dynamic_filter_drops_groups_and_refills(monkeypatch):
    worker = _make_worker(monkeypatch)
    for gid in range(8):
        group = _make_group(gid)
        group[0].reward = float(gid % 2)
        worker.output_queue.put((gid, group))
    monkeypatch.setattr(fa, "_get_global_worker", lambda args, data_buffer: worker)

    def keep_odd(args, group):
        return DynamicFilterOutput(keep=bool(group[0].reward), reason="even")

    monkeypatch.setattr(fa, "load_function", lambda path: keep_odd)
    args = SimpleNamespace(
        rollout_batch_size=3,
        rollout_data_transport="object-store",
        rollout_sample_filter_path=None,
        dynamic_sampling_filter_path="test.keep_odd",
    )
    result = asyncio.run(fa._generate_rollout_async(args, rollout_id=7, data_buffer=None))

    assert [group[0].index for group in result.samples] == [1, 3, 5]
    assert "_dropped_samples" not in result.metrics
    assert result.metrics["rollout/dynamic_filter/dropped_groups"] == 3
    assert result.metrics["rollout/dynamic_filter/dropped_ratio"] == 0.5
    assert result.metrics["rollout/dynamic_filter/drop_even"] == 3
    assert [gid for gid, _ in worker.get_completed_groups()] == [6, 7]


@pytest.mark.unit
def test_reward_std_filter_accepts_nested_custom_generation_samples():
    args = SimpleNamespace(reward_key=None)
    nested_group = [
        [Sample(reward=0.0), Sample(reward=0.0)],
        [Sample(reward=1.0), Sample(reward=1.0)],
    ]

    assert bool(check_reward_nonzero_std(args, nested_group).keep)

    for sample in nested_group[1]:
        sample.reward = 0.0
    output = check_reward_nonzero_std(args, nested_group)
    assert not bool(output.keep)
    assert output.reason == "zero_std_0.0"


@pytest.mark.unit
def test_done_callback_requeues_nested_aborted_group(monkeypatch):
    data_buffer = _FakeDataBuffer([])
    worker = _make_worker(monkeypatch, data_buffer=data_buffer)
    nested_group = [
        [Sample(status=Sample.Status.COMPLETED)],
        [Sample(status=Sample.Status.ABORTED)],
    ]

    class _DoneTask:
        def result(self):
            return nested_group

    worker._make_done_cb(0)(_DoneTask())

    assert data_buffer.requeued == [nested_group]
    assert worker.queue_size() == 0


@pytest.mark.parametrize("deterministic", [False, True])
def test_nested_aborted_group_requeues_resumes_and_keeps_sixteen_rollouts(monkeypatch, deterministic):
    """真实buffer→group入口→custom generate→callback，不能只测ABORTED入队。"""

    async def exercise():
        args = SimpleNamespace(
            n_samples_per_prompt=16,
            partial_rollout=False,
            group_rm=False,
            custom_generate_function_path="test.resume_fanout",
            rollout_sample_hook_path=None,
            sglang_enable_deterministic_inference=deterministic,
        )
        state = SimpleNamespace(
            aborted=False,
            semaphore=asyncio.Semaphore(16),
            generation_pacer=AsyncPacer(),
            dp_rank_context=lambda: nullcontext(None),
            active_server_generations=0,
            group_sampling_seeds=list(range(700, 716)),
        )
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _: state)
        monkeypatch.setattr(fa, "GenerateState", lambda _: state)
        source = RolloutDataSourceWithBuffer.__new__(RolloutDataSourceWithBuffer)
        source.args, source.buffer, source.buffer_filter = args, [], pop_first
        worker = fa.AsyncRolloutWorker(args, source, concurrency=16)
        group = [
            [
                Sample(
                    index=i,
                    group_index=3,
                    response="done",
                    status=Sample.Status.COMPLETED,
                    reward=float(i % 2),
                    tokens=[1, 2],
                    response_length=1,
                    loss_mask=[1],
                    rollout_log_probs=[-0.25],
                    session_id=f"sid-{i}",
                )
                for _ in range(2)
            ]
            for i in range(16)
        ]
        aborted = Sample(index=7, group_index=3, prompt="original", status=Sample.Status.ABORTED)
        group[7] = [aborted]
        original = [
            (leaf, list(leaf.tokens), leaf.reward, list(leaf.loss_mask), list(leaf.rollout_log_probs), leaf.session_id)
            for branch in group
            if branch is not group[7]
            for leaf in branch
        ]
        attempts = []

        async def generate(_args, sample, params):
            assert sample is aborted
            assert sample.session_id is not None
            attempts.append(params.copy())
            assert params.get("sampling_seed") == (707 if deterministic else None)
            params["max_new_tokens"] = 0
            if len(attempts) < 3:
                return [sample]  # 连续ABORTED不能不断增加列表深度。
            return [
                Sample(
                    index=sample.index,
                    group_index=sample.group_index,
                    response="new",
                    reward=1.0,
                    status=Sample.Status.COMPLETED,
                    tokens=[1, token],
                    response_length=1,
                    loss_mask=[1],
                    rollout_log_probs=[-0.5],
                )
                for token in (3, 4)
            ]

        async def reward(_args, samples):
            assert samples == [], "已带reward的段不能重打分"
            return []

        monkeypatch.setattr(sglang_rollout, "load_function", lambda _: generate)
        monkeypatch.setattr(sglang_rollout, "batched_async_rm", reward)
        worker._make_done_cb(0)(SimpleNamespace(result=lambda: group))
        assert worker.queue_size() == 0
        params = {"max_new_tokens": 8}
        for attempt in range(3):
            resumed = source.get_samples(1)
            assert len(resumed) == 1 and len(resumed[0]) == 16
            result = await sglang_rollout.generate_and_rm_group(args, resumed[0], params)
            assert len(result) == 16 and all(isinstance(branch, list) for branch in result)
            assert all(isinstance(leaf, Sample) for branch in result for leaf in branch)
            worker._make_done_cb(attempt + 1)(SimpleNamespace(result=lambda result=result: result))
        assert not source.buffer and worker.queue_size() == 1
        delivered = worker.get_completed_groups()[0][1]
        assert len(delivered) == 16 and len(delivered[7]) == 2
        assert [leaf.index for leaf in delivered[7]] == [7, 7]
        assert len(attempts) == 3 and all(p["max_new_tokens"] == 8 for p in attempts)
        assert params == {"max_new_tokens": 8}
        for leaf, tokens, reward_value, mask, log_probs, sid in original:
            assert leaf in list(iter_samples(delivered))
            assert (leaf.tokens, leaf.reward, leaf.loss_mask, leaf.rollout_log_probs, leaf.session_id) == (
                tokens,
                reward_value,
                mask,
                log_probs,
                sid,
            )

    asyncio.run(exercise())


@pytest.mark.unit
@pytest.mark.parametrize(
    "versions,expected",
    [(["10"], 0), (["8"], 2), (["3", "10"], 7), ([], None), (["bad"], None), (["11"], None), (["²"], None)],
)
def test_sample_staleness_uses_oldest_valid_version(versions, expected):
    assert sample_staleness(SimpleNamespace(weight_versions=versions), 10) == expected


@pytest.mark.unit
def test_staleness_metrics_use_serving_snapshot():
    samples = [
        SimpleNamespace(weight_versions=["2"]),
        SimpleNamespace(weight_versions=["3"]),
        SimpleNamespace(weight_versions=[]),
    ]
    assert compute_staleness_metrics(samples, 10) == {
        "staleness/unknown_count": 1,
        "staleness/mean": 7.5,
        "staleness/max": 8,
    }
    assert compute_staleness_metrics(samples, None) == {}


@pytest.mark.unit
def test_done_callback_never_blocks_event_loop_thread(monkeypatch):
    """The callback runs on the loop thread; blocking there freezes every
    in-flight generation. Push more results than the old bounded-queue cap
    (1000) through it and require completion."""
    worker = _make_worker(monkeypatch)

    class _DoneTask:
        def __init__(self, gid):
            self._result = _make_group(gid)

        def result(self):
            return self._result

    def _push_all():
        for gid in range(1001):
            worker._make_done_cb(gid)(_DoneTask(gid))

    pusher = threading.Thread(target=_push_all, daemon=True)
    pusher.start()
    pusher.join(timeout=30)

    assert not pusher.is_alive(), "done-callback blocked on a full output queue"
    assert worker.queue_size() == 1001


@pytest.mark.unit
def test_loop_backpressure_stops_topping_up_when_queue_is_full(monkeypatch):
    """With instantly-completing generations and plenty of fuel, the queue must
    plateau around ``concurrency`` instead of absorbing the whole dataset."""
    concurrency = 3
    fuel = 60
    data_buffer = _FakeDataBuffer([_make_group(i) for i in range(fuel)])

    async def _instant_generate(args, group, sampling_params, evaluation):
        return group

    monkeypatch.setattr(fa, "generate_and_rm_group", _instant_generate)
    worker = _make_worker(monkeypatch, data_buffer=data_buffer, concurrency=concurrency)
    worker.poll_interval = 0.01

    worker.start()
    try:
        # Give the loop ample iterations to overshoot if it is going to.
        deadline = time.time() + 3.0
        max_seen = 0
        while time.time() < deadline:
            max_seen = max(max_seen, worker.queue_size())
            if max_seen > 2 * concurrency:
                break
            time.sleep(0.02)
    finally:
        worker.stop()

    # In-flight tasks may still land after the gate check, so allow one pool
    # beyond the gate — but nothing near the unthrottled fuel size.
    assert 0 < max_seen <= 2 * concurrency, f"queue grew to {max_seen} with concurrency={concurrency}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
