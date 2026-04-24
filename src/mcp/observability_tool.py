"""Observability tool back-end — queries Datadog / Prometheus / Grafana APIs.

Implements the :class:`~src.mcp.tool_backend.ToolBackend` protocol by
retrieving live metric snapshots, alert details, and log excerpts from a
monitoring/observability platform.

Environment variables
---------------------
``OBS_TOOL_URL``
    Base URL of the observability API, e.g.
    ``https://api.datadoghq.com`` or ``http://prometheus:9090``.
``OBS_TOOL_API_KEY``
    API key or Bearer token.
``OBS_TOOL_APP_KEY``
    Datadog application key (Datadog requires both ``api_key`` and
    ``app_key``).  Ignored for non-Datadog providers.
``OBS_TOOL_PROVIDER``
    One of ``"datadog"``, ``"prometheus"``, ``"grafana"``, ``"mock"``
    (default).

Usage
-----
.. code-block:: python

    from src.mcp.observability_tool import ObservabilityToolBackend

    obs = ObservabilityToolBackend()
    results = obs.search("high CPU on api-gateway service last 30min")
"""

from __future__ import annotations

import os
from typing import Any

from src.models import SearchResult


class ObservabilityToolBackend:
    """Retrieve live metrics, alerts, and logs from a monitoring platform.

    Parameters
    ----------
    base_url:
        Observability API base URL.  Falls back to ``OBS_TOOL_URL`` env var.
    api_key:
        API key.  Falls back to ``OBS_TOOL_API_KEY`` env var.
    app_key:
        Datadog application key.  Falls back to ``OBS_TOOL_APP_KEY`` env var.
    provider:
        ``"datadog"`` | ``"prometheus"`` | ``"grafana"`` | ``"mock"``.
        Falls back to ``OBS_TOOL_PROVIDER`` env var.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        app_key: str | None = None,
        provider: str | None = None,
    ) -> None:
        self._base_url = base_url or os.environ.get("OBS_TOOL_URL", "")
        self._api_key = api_key or os.environ.get("OBS_TOOL_API_KEY", "")
        self._app_key = app_key or os.environ.get("OBS_TOOL_APP_KEY", "")
        self._provider = (provider or os.environ.get("OBS_TOOL_PROVIDER", "mock")).lower()

    def search(
        self,
        query: str,
        *,
        num_results: int = 5,
        time_range_minutes: int = 60,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Query the observability platform for signals related to *query*.

        Parameters
        ----------
        query:
            Natural-language description of the incident or service to probe.
        num_results:
            Maximum number of result snippets.
        time_range_minutes:
            Look-back window in minutes (default: 60).
        """
        if not self._base_url or self._provider == "mock":
            return self._mock_results(query, num_results)
        return self._api_search(query, num_results=num_results, minutes=time_range_minutes)

    def _api_search(
        self, query: str, *, num_results: int, minutes: int
    ) -> list[SearchResult]:  # pragma: no cover
        """Call the configured observability REST API (Datadog shown as example)."""
        import httpx  # noqa: PLC0415
        import time  # noqa: PLC0415

        headers = {
            "DD-API-KEY": self._api_key,
            "DD-APPLICATION-KEY": self._app_key,
        }
        now = int(time.time())
        start = now - minutes * 60
        try:
            if self._provider == "datadog":
                # Query recent alerts/events
                resp = httpx.get(
                    f"{self._base_url}/api/v1/events",
                    params={"start": start, "end": now, "tags": query[:50]},
                    headers=headers,
                    timeout=10.0,
                )
                resp.raise_for_status()
                events = resp.json().get("events", [])
                return [
                    SearchResult(
                        title=ev.get("title", "Event"),
                        url=ev.get("url", ""),
                        snippet=ev.get("text", "")[:300],
                        relevance_score=1.0,
                    )
                    for ev in events[:num_results]
                ]
            return []
        except Exception:
            return []

    def _mock_results(self, query: str, num_results: int) -> list[SearchResult]:
        snippet = query[:80]
        return [
            SearchResult(
                title=f"[mock] Alert: Elevated error rate — '{snippet[:40]}'",
                url="https://monitoring.example.com/alerts/mock-001",
                snippet=(
                    f"Service affected: api-gateway. "
                    "Error rate: 12% (threshold: 2%). "
                    "CPU: 94%. Memory: 78%. "
                    "Upstream dependency: payments-service returning 503. "
                    "Started: 14 minutes ago."
                ),
            )
            for _ in range(min(num_results, 2))
        ]
