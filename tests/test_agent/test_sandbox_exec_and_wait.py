"""CPU tests for ``slime.agent.sandbox.exec_and_wait``'s spawn-lock contract.

The detached spawn is guarded by ``mkdir {lock_dir} || exit 0`` so that a
transport-level retry of the *same* spawn RPC (a severed response replayed by
``E2BSandbox._rpc_retry``) cannot double-execute the command. That guard must
not leak into the next *logical* invocation of the same tag: the lock dir used
to survive forever, so a second ``exec_and_wait`` with the same tag skipped the
spawn at the guard (before the stale-marker cleanup, which sat behind it) and
``_await_done_marker`` immediately read the previous run's exit-code marker.

Concrete victim: ``harness.common.install_npm_cli`` retries a failed npm
install three times with ``tag="harness-npm-install"`` — attempts 2 and 3 ran
nothing and returned attempt 1's exit code and log verbatim.

The fake sandbox here interprets the actual command strings ``exec_and_wait``
issues (mkdir guard short-circuit, rm cleanup, setsid launch, marker polls), so
these tests hold for any command layout that keeps the semantics right and fail
for one that reuses stale state.
"""

from __future__ import annotations

import asyncio
import re
import shlex
from types import SimpleNamespace

import httpx
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

import slime.agent.sandbox as sandbox_mod
from slime.agent.sandbox import E2BSandbox, exec_and_wait

NUM_GPUS = 0


_POLL_RE = re.compile(r"test -f (\S+) && cat \1")
_SPAWN_RE = re.compile(r"mkdir (\S+) 2>/dev/null \|\| exit 0; (.*)$")
_LAUNCH_RE = re.compile(r"setsid bash (\S+) ")
_TAIL_RE = re.compile(r"tail -c \d+ (\S+)")


class ShellFakeSandbox:
    """Interprets exec_and_wait's shell commands against an in-memory FS.

    ``run_script`` is called once per *actual* launch with the launcher path;
    it returns ``(exit_code, output)`` which land in the done/out marker files
    exactly like the real detached command would write them.
    """

    sandbox_id = "shell-fake"

    def __init__(self, run_script):
        self.run_script = run_script
        self.files: dict[str, str] = {}
        self.dirs: set[str] = set()
        self.launches = 0
        self.exec_log: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def write_file(self, path, content, *, user="root"):
        self.files[path] = content

    async def read_file(self, path, *, user="root"):
        return self.files.get(path, "")

    async def exec(self, cmd, *, user="root", env=None, timeout=120, check=False, idempotent=True):
        self.exec_log.append(cmd)

        poll = _POLL_RE.search(cmd)
        if poll:
            path = poll.group(1)
            if path in self.files:
                return 0, self.files[path], ""
            return 1, "", ""

        tail = _TAIL_RE.search(cmd)
        if tail:
            return 0, self.files.get(tail.group(1), "")[-512:], ""

        spawn = _SPAWN_RE.search(cmd)
        if spawn:
            lock_dir, rest = spawn.groups()
            if lock_dir in self.dirs:
                return 0, "", ""  # guard hit: nothing after it runs
            self.dirs.add(lock_dir)
            self._run_shell_fragment(rest)
            return 0, "", ""

        # Plain cleanup command(s): rm -rf / rm -f sequences.
        self._run_shell_fragment(cmd)
        return 0, "", ""

    def _run_shell_fragment(self, fragment):
        for part in fragment.split(";"):
            part = part.strip().rstrip("&").strip()
            if part.startswith(("rm -rf", "rm -f")):
                for token in shlex.split(part)[2:]:
                    self.files.pop(token, None)
                    self.dirs.discard(token)
            launch = _LAUNCH_RE.search(part + " ")
            if launch:
                launcher = launch.group(1)
                assert launcher in self.files, "launcher must be written before the spawn"
                self.launches += 1
                exit_code, output = self.run_script(self.launches)
                out_file = re.search(r"> (\S+) 2>&1", part).group(1)
                done_file = launcher.replace(".sh", ".done")
                self.files[out_file] = output
                self.files[done_file] = f"{exit_code}\n"


@pytest.fixture
def fast_marker_polls(monkeypatch):
    """_await_done_marker sleeps 5s between polls; make that instant."""

    async def _instant(_seconds):
        return None

    monkeypatch.setattr(sandbox_mod, "asyncio", SimpleNamespace(sleep=_instant))


@pytest.mark.unit
def test_same_tag_reinvocation_actually_reruns(fast_marker_polls):
    """A retry loop (e.g. install_npm_cli) must re-run the command, not be fed
    the previous attempt's stale exit code and log."""
    outcomes = {1: (1, "attempt-1 failed"), 2: (0, "attempt-2 ok")}
    sb = ShellFakeSandbox(run_script=lambda n: outcomes[n])

    async def _two_attempts():
        first = await exec_and_wait(sb, cmd="npm install", time_budget_sec=60, tag="npm", want_output=True)
        second = await exec_and_wait(sb, cmd="npm install", time_budget_sec=60, tag="npm", want_output=True)
        return first, second

    (first_code, first_out), (second_code, second_out) = asyncio.run(_two_attempts())

    assert (first_code, first_out) == (1, "attempt-1 failed")
    assert sb.launches == 2, "second invocation must actually spawn the command"
    assert (second_code, second_out) == (0, "attempt-2 ok")


@pytest.mark.unit
def test_transport_retry_of_the_spawn_stays_deduped(fast_marker_polls):
    """Replaying the spawn RPC itself (what the mkdir guard is *for*) must not
    double-execute the command."""
    sb = ShellFakeSandbox(run_script=lambda n: (0, "ok"))

    asyncio.run(exec_and_wait(sb, cmd="true", time_budget_sec=60, tag="job"))

    spawn_cmds = [c for c in sb.exec_log if "setsid" in c]
    assert len(spawn_cmds) == 1
    # Replay the identical spawn RPC, as _rpc_retry would after a severed
    # response: the guard must swallow it.
    asyncio.run(sb.exec(spawn_cmds[0]))
    assert sb.launches == 1

    # The per-invocation cleanup must NOT ride inside the guarded spawn —
    # behind the guard it never runs on a replayed tag.
    assert not any("rm -" in c and "setsid" in c for c in sb.exec_log)


@pytest.mark.unit
def test_closed_resource_error_retries_idempotent_rpc(monkeypatch):
    class ClosedResourceError(Exception):
        pass

    attempts = 0
    sandbox = E2BSandbox("unused", image_metadata_key="image", rpc_retries=2)

    async def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ClosedResourceError("stream closed")
        return "ok"

    async def no_backoff(_seconds):
        return None

    monkeypatch.setattr(sandbox_mod.asyncio, "sleep", no_backoff)
    assert asyncio.run(sandbox._rpc_retry("read", operation)) == "ok"
    assert attempts == 2


@pytest.mark.unit
def test_closed_resource_error_does_not_replay_non_idempotent_rpc():
    class ClosedResourceError(Exception):
        pass

    attempts = 0
    sandbox = E2BSandbox("unused", image_metadata_key="image", rpc_retries=6)

    async def operation():
        nonlocal attempts
        attempts += 1
        raise ClosedResourceError("stream closed")

    with pytest.raises(ClosedResourceError):
        asyncio.run(sandbox._rpc_retry("spawn", operation, idempotent=False))
    assert attempts == 1


@pytest.mark.parametrize("same_sandbox", [True, False])
def test_retry_does_not_close_pool_used_by_another_inflight_rpc(same_sandbox):
    async def run_case():
        closed = False

        async def close_pool():
            nonlocal closed
            closed = True

        shared_sdk = SimpleNamespace(_transport=SimpleNamespace(pool=SimpleNamespace(aclose=close_pool)))
        retrying = E2BSandbox("unused", rpc_retries=2)
        retrying._sb = shared_sdk
        retrying.rpc_backoff_base_sec = 0
        retrying.rpc_backoff_cap_sec = 0
        other = retrying if same_sandbox else E2BSandbox("other", rpc_retries=2)
        other._sb = shared_sdk
        started = asyncio.Event()
        release = asyncio.Event()
        attempts = 0
        other_calls = 0

        async def inflight_operation():
            nonlocal other_calls
            other_calls += 1
            started.set()
            await release.wait()
            if closed:
                raise httpx.ReadError("shared pool closed under an unrelated RPC")
            return "uninterrupted"

        async def failing_operation():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.ReadError("one RPC failed")
            return "recovered"

        task = asyncio.create_task(other._rpc_retry("spawn", inflight_operation, idempotent=False))
        try:
            await started.wait()
            assert await retrying._rpc_retry("publish", failing_operation) == "recovered"
        finally:
            release.set()
        assert await task == "uninterrupted"
        assert attempts == 2 and other_calls == 1
        assert not closed

    asyncio.run(run_case())


def test_exhausted_rpc_retries_raise_without_closing_shared_pool():
    attempts = 0
    closures = 0
    sandbox = E2BSandbox("unused", rpc_retries=3)
    sandbox.rpc_backoff_base_sec = 0
    sandbox.rpc_backoff_cap_sec = 0

    async def close_pool():
        nonlocal closures
        closures += 1

    sandbox._sb = SimpleNamespace(_transport=SimpleNamespace(pool=SimpleNamespace(aclose=close_pool)))

    async def operation():
        nonlocal attempts
        attempts += 1
        raise httpx.ReadError("permanent transport failure")

    with pytest.raises(httpx.ReadError, match="permanent transport failure"):
        asyncio.run(sandbox._rpc_retry("read", operation))
    assert attempts == 3 and closures == 0


def test_retry_recovers_disconnected_connection_without_interrupting_other_request():
    async def run_case():
        inflight_started = asyncio.Event()
        release = asyncio.Event()
        requests = 0

        async def inflight(_request):
            inflight_started.set()
            await release.wait()
            return web.Response(text="uninterrupted")

        async def disconnect_once(request):
            nonlocal requests
            requests += 1
            if requests == 1:
                request.transport.close()
            return web.Response(text="reconnected")

        app = web.Application()
        app.router.add_get("/inflight", inflight)
        app.router.add_get("/flaky", disconnect_once)
        sandbox = E2BSandbox("unused", rpc_retries=2)
        sandbox.rpc_backoff_base_sec = 0
        sandbox.rpc_backoff_cap_sec = 0
        async with TestServer(app, shutdown_timeout=0.1) as server, httpx.AsyncClient(trust_env=False) as client:
            sandbox._sb = SimpleNamespace(_transport=SimpleNamespace(pool=client._transport._pool))
            other = asyncio.create_task(sandbox._rpc_retry(
                "spawn", lambda: client.get(str(server.make_url("/inflight"))), idempotent=False,
            ))
            try:
                await inflight_started.wait()
                recovered = await sandbox._rpc_retry("read", lambda: client.get(str(server.make_url("/flaky"))))
                assert recovered.status_code == 200 and recovered.text == "reconnected"
            finally:
                release.set()
                outcomes = await asyncio.gather(other, return_exceptions=True)
            assert not isinstance(outcomes[0], BaseException)
            assert outcomes[0].text == "uninterrupted"
        assert requests == 2

    asyncio.run(asyncio.wait_for(run_case(), timeout=10))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
