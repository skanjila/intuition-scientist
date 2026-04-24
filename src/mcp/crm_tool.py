"""CRM tool back-end — wraps Salesforce / HubSpot REST APIs.

Implements the :class:`~src.mcp.tool_backend.ToolBackend` protocol by
querying a CRM system for account / opportunity / contact data related to a
search query.

Environment variables
---------------------
``CRM_TOOL_URL``
    CRM instance URL, e.g. ``https://yourorg.my.salesforce.com``.
``CRM_TOOL_API_KEY``
    OAuth access token or API key.
``CRM_TOOL_PROVIDER``
    One of ``"salesforce"``, ``"hubspot"``, ``"mock"`` (default).

Usage
-----
.. code-block:: python

    from src.mcp.crm_tool import CRMToolBackend

    crm = CRMToolBackend()
    results = crm.search("Acme Corp Q3 renewal", num_results=3)
"""

from __future__ import annotations

import os
from typing import Any

from src.models import SearchResult


class CRMToolBackend:
    """Query a CRM system for account and opportunity intelligence.

    Parameters
    ----------
    base_url:
        CRM instance base URL.  Falls back to ``CRM_TOOL_URL`` env var.
    api_key:
        Access token.  Falls back to ``CRM_TOOL_API_KEY`` env var.
    provider:
        One of ``"salesforce"``, ``"hubspot"``, ``"mock"``.
        Falls back to ``CRM_TOOL_PROVIDER`` env var.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        provider: str | None = None,
    ) -> None:
        self._base_url = base_url or os.environ.get("CRM_TOOL_URL", "")
        self._api_key = api_key or os.environ.get("CRM_TOOL_API_KEY", "")
        self._provider = (provider or os.environ.get("CRM_TOOL_PROVIDER", "mock")).lower()

    def search(
        self,
        query: str,
        *,
        num_results: int = 5,
        tenant_id: str = "",
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Search the CRM for accounts, contacts, or opportunities matching *query*.

        Returns CRM records as :class:`~src.models.SearchResult` objects.
        """
        if not self._base_url or self._provider == "mock":
            return self._mock_results(query, num_results)
        return self._api_search(query, num_results=num_results)

    # ------------------------------------------------------------------
    # Provider-specific implementations
    # ------------------------------------------------------------------

    def _api_search(self, query: str, *, num_results: int) -> list[SearchResult]:  # pragma: no cover
        """Call the configured CRM REST API.

        Replace this stub body with provider-specific HTTP calls.
        Salesforce SOSL search is shown as an example.
        """
        import httpx  # noqa: PLC0415

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            if self._provider == "salesforce":
                sosl = f"FIND {{{query}}} IN ALL FIELDS RETURNING Account(Id, Name, Industry, AnnualRevenue), Opportunity(Id, Name, StageName, Amount)"
                resp = httpx.get(
                    f"{self._base_url}/services/data/v58.0/search",
                    params={"q": sosl},
                    headers=headers,
                    timeout=10.0,
                )
                resp.raise_for_status()
                records = resp.json().get("searchRecords", [])
                return [
                    SearchResult(
                        title=r.get("Name", "Record"),
                        url=f"{self._base_url}/{r.get('Id', '')}",
                        snippet=str(r)[:300],
                    )
                    for r in records[:num_results]
                ]
            return []
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Mock fallback
    # ------------------------------------------------------------------

    def _mock_results(self, query: str, num_results: int) -> list[SearchResult]:
        snippet = query[:80]
        return [
            SearchResult(
                title=f"[mock] CRM: Account related to '{snippet}'",
                url="https://crm.example.com/accounts/001mock",
                snippet=(
                    f"Account: {snippet[:40]}. "
                    "Industry: Technology. ARR: $500K. "
                    "Last activity: 14 days ago. Open opportunity: Q3 renewal ($120K, Negotiation)."
                ),
            )
            for _ in range(min(num_results, 2))
        ]
