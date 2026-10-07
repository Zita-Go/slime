"""Exact-template E2B sandbox used by terminal-agent rollouts."""

from __future__ import annotations

from slime.agent.sandbox import E2BSandbox

from .e2b_config import resolve_e2b_connection


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
