"""Exact-template E2B sandbox used by terminal-agent rollouts."""

from __future__ import annotations

import os

from slime.agent.sandbox import E2BSandbox


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def resolve_e2b_connection() -> tuple[str, str | None, str | None]:
    api_key = _first_env(
        "NEXTASK_BENCH_E2B_API_KEY",
        "NEXTASK_SOLVE_E2B_API_KEY",
        "NEXTASK_GEN_E2B_API_KEY",
        "E2B_API_KEY",
        "E2B_API_KEY_CODEX",
    )
    if not api_key:
        raise RuntimeError("E2B API key is not configured")
    api_url = _first_env(
        "NEXTASK_BENCH_E2B_API_URL",
        "NEXTASK_SOLVE_E2B_API_URL",
        "NEXTASK_GEN_E2B_API_URL",
        "E2B_API_URL",
    )
    domain = _first_env(
        "NEXTASK_BENCH_E2B_DOMAIN",
        "NEXTASK_SOLVE_E2B_DOMAIN",
        "NEXTASK_GEN_E2B_DOMAIN",
        "E2B_DOMAIN",
    )
    return api_key, api_url, domain


class NextaskE2BSandbox(E2BSandbox):
    """Create directly from the package's frozen ``template_identity``.

    The generic slime E2B backend supports metadata-based image routing for
    internal SWE gateways.  NexTask v3 packages instead freeze a concrete E2B
    template identity, so using the SDK's ``template=`` field is the auditable
    and cross-checked path.
    """

    async def __aenter__(self) -> NextaskE2BSandbox:
        from e2b import AsyncSandbox

        api_key, api_url, domain = resolve_e2b_connection()
        options = {"api_key": api_key}
        if api_url:
            options["api_url"] = api_url
        if domain:
            options["domain"] = domain
        self._sb = await self._rpc_retry(
            "create",
            lambda: AsyncSandbox.create(
                template=self.image,
                timeout=self.timeout,
                allow_internet_access=True,
                **options,
            ),
        )
        self.sandbox_id = self._sb.sandbox_id
        return self
