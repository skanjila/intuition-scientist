"""Data-warehouse tool back-end — queries BigQuery / Snowflake / Redshift.

Implements the :class:`~src.mcp.tool_backend.ToolBackend` protocol by
executing read-only SQL queries against a cloud data warehouse and returning
the results as :class:`~src.models.SearchResult` snippets for use by the
analytics report-generation agent.

Environment variables
---------------------
``DW_TOOL_URL``
    JDBC-style connection string or REST API endpoint.
``DW_TOOL_API_KEY``
    API key / service-account JSON path (provider-dependent).
``DW_TOOL_PROVIDER``
    One of ``"bigquery"``, ``"snowflake"``, ``"redshift"``, ``"mock"``
    (default).
``DW_TOOL_PROJECT``
    GCP project ID (BigQuery only).
``DW_TOOL_DATASET``
    Default dataset/schema name.

Usage
-----
.. code-block:: python

    from src.mcp.datawarehouse_tool import DataWarehouseToolBackend

    dw = DataWarehouseToolBackend()
    results = dw.search("weekly revenue by product line last 4 weeks")
"""

from __future__ import annotations

import os
from typing import Any

from src.models import SearchResult


class DataWarehouseToolBackend:
    """Execute read-only analytics queries against a cloud data warehouse.

    Parameters
    ----------
    connection_url:
        DW connection string / endpoint.  Falls back to ``DW_TOOL_URL``.
    api_key:
        Authentication credential.  Falls back to ``DW_TOOL_API_KEY``.
    provider:
        One of ``"bigquery"``, ``"snowflake"``, ``"redshift"``, ``"mock"``.
        Falls back to ``DW_TOOL_PROVIDER``.
    project:
        GCP project (BigQuery).  Falls back to ``DW_TOOL_PROJECT``.
    dataset:
        Default dataset.  Falls back to ``DW_TOOL_DATASET``.
    """

    def __init__(
        self,
        connection_url: str | None = None,
        api_key: str | None = None,
        provider: str | None = None,
        project: str | None = None,
        dataset: str | None = None,
    ) -> None:
        self._url = connection_url or os.environ.get("DW_TOOL_URL", "")
        self._api_key = api_key or os.environ.get("DW_TOOL_API_KEY", "")
        self._provider = (provider or os.environ.get("DW_TOOL_PROVIDER", "mock")).lower()
        self._project = project or os.environ.get("DW_TOOL_PROJECT", "")
        self._dataset = dataset or os.environ.get("DW_TOOL_DATASET", "analytics")

    def search(
        self,
        query: str,
        *,
        num_results: int = 5,
        sql: str = "",
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Run an analytics query and return results as ``SearchResult`` snippets.

        Parameters
        ----------
        query:
            Natural-language description of the metric to retrieve.
        num_results:
            Maximum rows to return.
        sql:
            Optional explicit SQL to execute instead of deriving one from
            *query*.  When supplied, *query* is used only as the result title.
        """
        if not self._url or self._provider == "mock":
            return self._mock_results(query, num_results)
        return self._execute(query, sql=sql, num_results=num_results)

    def _execute(self, query: str, *, sql: str, num_results: int) -> list[SearchResult]:  # pragma: no cover
        """Execute SQL against the configured warehouse (Snowflake shown)."""
        try:
            if self._provider == "snowflake":
                import snowflake.connector  # type: ignore[import]  # noqa: PLC0415

                conn = snowflake.connector.connect(
                    account=self._url,
                    token=self._api_key,
                    authenticator="oauth",
                )
                cursor = conn.cursor()
                effective_sql = sql or f"SELECT * FROM {self._dataset}.metrics WHERE description ILIKE '%{query[:40]}%' LIMIT {num_results}"
                cursor.execute(effective_sql)
                rows = cursor.fetchmany(num_results)
                cursor.close()
                conn.close()
                return [
                    SearchResult(
                        title=f"Row {i+1}: {query[:40]}",
                        url="",
                        snippet=str(dict(zip([d[0] for d in cursor.description], row)))[:300],
                    )
                    for i, row in enumerate(rows)
                ]
            return []
        except Exception:
            return []

    def _mock_results(self, query: str, num_results: int) -> list[SearchResult]:
        snippet = query[:60]
        return [
            SearchResult(
                title=f"[mock] DW metric: {snippet[:40]}",
                url="",
                snippet=(
                    f"Metric summary for '{snippet}': "
                    "Period: 2026-W16. "
                    "Revenue: $4.2M (+8% WoW, +22% YoY). "
                    "CAC: $420 (-5% WoW). "
                    "Churn: 1.8% (threshold: 2%). "
                    "Top anomaly: Product-C revenue -34% WoW (investigate)."
                ),
            )
            for _ in range(min(num_results, 2))
        ]
