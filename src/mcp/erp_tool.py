"""ERP tool back-end — wraps SAP / Oracle REST APIs for supply-chain data.

Implements the :class:`~src.mcp.tool_backend.ToolBackend` protocol by
querying an ERP system for live inventory levels, open purchase orders,
supplier lead times, and alternative sourcing options.

Environment variables
---------------------
``ERP_TOOL_URL``
    Base URL of the ERP REST API, e.g.
    ``https://yourorg.s4hana.cloud.sap/sap/opu/odata/sap``.
``ERP_TOOL_API_KEY``
    Bearer token or basic-auth credentials (base64-encoded ``user:pass``).
``ERP_TOOL_PROVIDER``
    One of ``"sap"``, ``"oracle"``, ``"netsuite"``, ``"mock"`` (default).

Usage
-----
.. code-block:: python

    from src.mcp.erp_tool import ERPToolBackend

    erp = ERPToolBackend()
    results = erp.search("SKU-12345 inventory and open POs")
"""

from __future__ import annotations

import os
from typing import Any

from src.models import SearchResult


class ERPToolBackend:
    """Query an ERP system for inventory, PO, and supplier data.

    Parameters
    ----------
    base_url:
        ERP API base URL.  Falls back to ``ERP_TOOL_URL`` env var.
    api_key:
        Auth token.  Falls back to ``ERP_TOOL_API_KEY`` env var.
    provider:
        One of ``"sap"``, ``"oracle"``, ``"netsuite"``, ``"mock"``.
        Falls back to ``ERP_TOOL_PROVIDER`` env var.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        provider: str | None = None,
    ) -> None:
        self._base_url = base_url or os.environ.get("ERP_TOOL_URL", "")
        self._api_key = api_key or os.environ.get("ERP_TOOL_API_KEY", "")
        self._provider = (provider or os.environ.get("ERP_TOOL_PROVIDER", "mock")).lower()

    def search(
        self,
        query: str,
        *,
        num_results: int = 5,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Query the ERP system for records related to *query*.

        Typical queries include SKU numbers, supplier names, PO numbers,
        or natural-language descriptions like ``"late deliveries this week"``.
        """
        if not self._base_url or self._provider == "mock":
            return self._mock_results(query, num_results)
        return self._api_search(query, num_results=num_results)

    def _api_search(self, query: str, *, num_results: int) -> list[SearchResult]:  # pragma: no cover
        """Call the configured ERP REST API (SAP OData shown as example)."""
        import httpx  # noqa: PLC0415

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }
        try:
            if self._provider == "sap":
                resp = httpx.get(
                    f"{self._base_url}/API_MATERIAL_STOCK_SRV/A_MatlStkInAcctMod",
                    params={"$filter": f"contains(Material, '{query[:20]}')", "$top": num_results},
                    headers=headers,
                    timeout=10.0,
                )
                resp.raise_for_status()
                items = resp.json().get("value", [])
                return [
                    SearchResult(
                        title=f"Stock: {item.get('Material', '')}",
                        url=f"{self._base_url}/materials/{item.get('Material', '')}",
                        snippet=str(item)[:300],
                    )
                    for item in items
                ]
            return []
        except Exception:
            return []

    def _mock_results(self, query: str, num_results: int) -> list[SearchResult]:
        snippet = query[:60]
        return [
            SearchResult(
                title=f"[mock] ERP record for '{snippet[:30]}'",
                url="https://erp.example.com/inventory/mock",
                snippet=(
                    f"Material: {snippet[:20]}. "
                    "On-hand stock: 145 units. "
                    "Safety stock: 200 units (BELOW MINIMUM). "
                    "Open POs: 2 (expected delivery: +18 days). "
                    "Alternative suppliers: SupplierB (lead time: 7 days), SupplierC (lead time: 12 days)."
                ),
            )
            for _ in range(min(num_results, 2))
        ]
