"""Exact-template E2B sandbox used by terminal-agent rollouts."""

from __future__ import annotations

import logging

from slime.agent.sandbox import E2BSandbox

from .e2b_config import resolve_e2b_connection
from .e2b_network import resolve_e2b_proxy

logger = logging.getLogger(__name__)


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
        options = {"api_key": api_key, "proxy": resolve_e2b_proxy()}
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
        self.cleanup_confirmed = False
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._sb is None:
            self.cleanup_confirmed = True
            return
        try:
            # Both True (killed) and False (already absent) are authoritative
            # absence states in the E2B SDK.
            await self._rpc_retry("kill", lambda: self._sb.kill())
            self.cleanup_confirmed = True
        except Exception as cleanup_error:
            self.cleanup_confirmed = False
            logger.error("[terminal_agent_rl] live sandbox cleanup failed: %s", self.sandbox_id)
            raise RuntimeError(f"live E2B sandbox cleanup was not confirmed: {self.sandbox_id}") from cleanup_error
