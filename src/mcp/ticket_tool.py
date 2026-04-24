"""Ticket-management tool back-end — wraps CRM / helpdesk REST APIs.

Implements the :class:`~src.mcp.tool_backend.ToolBackend` protocol by
querying a helpdesk system (Zendesk, Intercom, Freshdesk, Jira Service
Management, etc.) for tickets related to a search query.

Environment variables
---------------------
``TICKET_TOOL_URL``
    Base URL of the helpdesk REST API, e.g.
    ``https://yourcompany.zendesk.com/api/v2``.
``TICKET_TOOL_API_KEY``
    API key or Bearer token for the helpdesk API.
``TICKET_TOOL_PROVIDER``
    One of ``"zendesk"``, ``"intercom"``, ``"jira"``, ``"mock"`` (default).

When no URL is configured the tool falls back to an empty mock result so that
the agent pipeline degrades gracefully and uses its LLM knowledge alone.

Usage
-----
.. code-block:: python

    from src.mcp.ticket_tool import TicketToolBackend

    tool = TicketToolBackend()
    results = tool.search("payment failed credit card error")
"""

from __future__ import annotations

import os
from typing import Any

from src.models import SearchResult


class TicketToolBackend:
    """Query a helpdesk / ticket-management system.

    Parameters
    ----------
    base_url:
        Helpdesk API base URL.  Falls back to ``TICKET_TOOL_URL`` env var.
    api_key:
        Authentication token.  Falls back to ``TICKET_TOOL_API_KEY`` env var.
    provider:
        One of ``"zendesk"``, ``"intercom"``, ``"jira"``, ``"mock"``.
        Falls back to ``TICKET_TOOL_PROVIDER`` env var.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        provider: str | None = None,
    ) -> None:
        self._base_url = base_url or os.environ.get("TICKET_TOOL_URL", "")
        self._api_key = api_key or os.environ.get("TICKET_TOOL_API_KEY", "")
        self._provider = (provider or os.environ.get("TICKET_TOOL_PROVIDER", "mock")).lower()

    def search(
        self,
        query: str,
        *,
        num_results: int = 5,
        tenant_id: str = "",
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Search the ticket system for issues related to *query*.

        Returns recent/similar tickets as :class:`~src.models.SearchResult`
        objects so the agent can use them as context when drafting a response.
        """
        if not self._base_url or self._provider == "mock":
            return self._mock_results(query, num_results)
        return self._api_search(query, num_results=num_results, tenant_id=tenant_id)

    # ------------------------------------------------------------------
    # Provider-specific implementations
    # ------------------------------------------------------------------

    def _api_search(
        self, query: str, *, num_results: int, tenant_id: str
    ) -> list[SearchResult]:  # pragma: no cover
        """Call the configured helpdesk REST API.

        Replace this stub body with provider-specific HTTP calls.
        The Zendesk search API endpoint is shown as an example.
        """
        import httpx  # noqa: PLC0415

        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            if self._provider == "zendesk":
                resp = httpx.get(
                    f"{self._base_url}/search.json",
                    params={"query": f"type:ticket {query}", "per_page": num_results},
                    headers=headers,
                    timeout=10.0,
                )
                resp.raise_for_status()
                items = resp.json().get("results", [])
                return [
                    SearchResult(
                        title=f"Ticket #{item.get('id')}: {item.get('subject', '')}",
                        url=f"{self._base_url}/tickets/{item.get('id')}",
                        snippet=item.get("description", "")[:300],
                    )
                    for item in items[:num_results]
                ]
            # Add other providers (intercom, jira, freshdesk) here
            return []
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Mock fallback
    # ------------------------------------------------------------------

    def _mock_results(self, query: str, num_results: int) -> list[SearchResult]:
        """Return a small set of mock ticket results for offline testing."""
        snippet = query[:80]
        return [
            SearchResult(
                title=f"[mock] Ticket #{i+1}: Related to '{snippet}'",
                url=f"https://support.example.com/tickets/{1000 + i}",
                snippet=(
                    f"Customer reported an issue similar to: {snippet}. "
                    "Status: resolved. Resolution: escalated to engineering."
                ),
            )
            for i in range(min(num_results, 3))
        ]
