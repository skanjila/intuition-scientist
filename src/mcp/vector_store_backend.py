"""Vector-store tool back-end — retrieval-augmented generation over internal docs.

Implements the :class:`~src.mcp.tool_backend.ToolBackend` protocol using a
simple in-memory TF-IDF index so the module works offline without any
external dependencies.  For production use, swap the ``_build_index`` method
for a real vector-store (Chroma, pgvector, Weaviate, Pinecone, etc.).

Environment variables
---------------------
``VECTOR_STORE_URL``
    Optional URL for a remote vector-store REST API (Chroma HTTP client, etc.).
    When absent the local in-memory index is used.
``VECTOR_STORE_API_KEY``
    API key for the remote vector store (if required).

Usage
-----
.. code-block:: python

    from src.mcp.vector_store_backend import VectorStoreBackend

    store = VectorStoreBackend()
    store.add_documents([
        {"id": "pol-001", "text": "GDPR Art.17 right-to-erasure policy...", "source": "gdpr_policy.pdf"},
    ])
    results = store.search("data deletion request", tenant_id="acme")
"""

from __future__ import annotations

import math
import os
import re
from typing import Any

from src.models import SearchResult


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]{3,}", text.lower())


def _tf_vec(tokens: list[str]) -> dict[str, float]:
    vec: dict[str, float] = {}
    for t in tokens:
        vec[t] = vec.get(t, 0.0) + 1.0
    for t in vec:
        vec[t] = 1.0 + math.log(vec[t])
    return vec


def _cosine(va: dict[str, float], vb: dict[str, float]) -> float:
    keys = set(va) | set(vb)
    dot = sum(va.get(k, 0.0) * vb.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(v ** 2 for v in va.values()))
    nb = math.sqrt(sum(v ** 2 for v in vb.values()))
    return dot / (na * nb) if na and nb else 0.0


class VectorStoreBackend:
    """In-memory TF-IDF document store with a ``search`` interface.

    For production deployments replace :meth:`_similarity` and the internal
    ``_docs`` list with calls to a real embedding-based vector store.

    Parameters
    ----------
    remote_url:
        Optional remote vector-store URL.  Falls back to
        ``VECTOR_STORE_URL`` env var, then uses the local in-memory index.
    api_key:
        API key for the remote store.  Falls back to
        ``VECTOR_STORE_API_KEY`` env var.
    """

    def __init__(
        self,
        remote_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._remote_url = remote_url or os.environ.get("VECTOR_STORE_URL", "")
        self._api_key = api_key or os.environ.get("VECTOR_STORE_API_KEY", "")
        # Each doc: {"id": str, "text": str, "source": str, "tenant_id": str}
        self._docs: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_documents(self, docs: list[dict[str, str]]) -> None:
        """Add documents to the in-memory index.

        Parameters
        ----------
        docs:
            List of dicts with keys ``"id"``, ``"text"``, ``"source"``,
            and optionally ``"tenant_id"``.
        """
        self._docs.extend(docs)

    def search(
        self,
        query: str,
        *,
        num_results: int = 5,
        tenant_id: str = "",
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Search the document store and return ranked ``SearchResult`` objects.

        When a ``tenant_id`` is supplied only documents belonging to that
        tenant (or documents with no tenant) are considered.
        """
        if self._remote_url:
            return self._remote_search(query, num_results=num_results, tenant_id=tenant_id)
        return self._local_search(query, num_results=num_results, tenant_id=tenant_id)

    # ------------------------------------------------------------------
    # Local (in-memory) implementation
    # ------------------------------------------------------------------

    def _local_search(
        self, query: str, *, num_results: int, tenant_id: str
    ) -> list[SearchResult]:
        q_vec = _tf_vec(_tokenize(query))
        scored: list[tuple[float, dict[str, str]]] = []
        for doc in self._docs:
            doc_tenant = doc.get("tenant_id", "")
            if tenant_id and doc_tenant and doc_tenant != tenant_id:
                continue
            d_vec = _tf_vec(_tokenize(doc.get("text", "")))
            score = _cosine(q_vec, d_vec)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in scored[:num_results]:
            results.append(SearchResult(
                title=doc.get("id", "doc"),
                url=doc.get("source", ""),
                snippet=doc.get("text", "")[:300],
                relevance_score=round(score, 4),
            ))
        return results

    # ------------------------------------------------------------------
    # Remote implementation (stub — replace with real HTTP client)
    # ------------------------------------------------------------------

    def _remote_search(
        self, query: str, *, num_results: int, tenant_id: str
    ) -> list[SearchResult]:  # pragma: no cover
        """Call a remote vector-store REST API.

        This stub is intentionally not implemented — replace the body with
        real HTTP calls to your chosen vector-store provider.
        """
        import httpx  # noqa: PLC0415

        try:
            resp = httpx.post(
                f"{self._remote_url}/search",
                json={"query": query, "top_k": num_results, "tenant_id": tenant_id},
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=10.0,
            )
            resp.raise_for_status()
            items = resp.json().get("results", [])
            return [
                SearchResult(
                    title=item.get("id", ""),
                    url=item.get("source", ""),
                    snippet=item.get("text", "")[:300],
                    relevance_score=item.get("score"),
                )
                for item in items[:num_results]
            ]
        except Exception:
            return []
