"""ToolBackend protocol — uniform interface for all data-retrieval back-ends.

Every tool back-end (web search, vector store, CRM, ERP, observability, etc.)
must satisfy this protocol so that :class:`~src.agents.base_agent.BaseAgent`
can call them interchangeably without knowing which data source is behind them.

The interface is intentionally minimal — a single :meth:`search` method that
accepts a query string plus optional keyword arguments and returns a list of
:class:`~src.models.SearchResult` objects.  Backends that connect to
structured databases or REST APIs may use the ``**kwargs`` to pass additional
typed parameters (e.g. ``tenant_id``, ``filters``, ``table_name``).

Usage example
-------------
.. code-block:: python

    from src.mcp.tool_backend import ToolBackend
    from src.mcp.vector_store_backend import VectorStoreBackend

    store = VectorStoreBackend(corpus_dir="policies/")
    results = store.search("GDPR data retention requirements", tenant_id="acme")
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from src.models import SearchResult


@runtime_checkable
class ToolBackend(Protocol):
    """Minimal interface that every tool/data back-end must satisfy.

    Back-ends are free to add extra constructor parameters (API keys, URLs,
    embedding models, etc.) as long as they implement this single method.
    """

    def search(self, query: str, *, num_results: int = 5, **kwargs: Any) -> list[SearchResult]:
        """Search the back-end and return up to *num_results* results.

        Parameters
        ----------
        query:
            Free-text search query.
        num_results:
            Maximum number of results to return.
        **kwargs:
            Backend-specific parameters (e.g. ``tenant_id``, ``filters``).

        Returns
        -------
        list[SearchResult]
            Ordered by relevance (most relevant first).  May be empty when
            no results match or when the back-end is unavailable.
        """
        ...
