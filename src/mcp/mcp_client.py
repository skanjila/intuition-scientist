"""MCP (Model Context Protocol) client for internet search and retrieval.

This module provides a thin wrapper around web-search APIs so that domain
agents can enrich their answers with real-world, up-to-date information.

The implementation uses DuckDuckGo's HTML interface as a zero-authentication
search back-end. In production environments this can be swapped for any MCP-
compatible server or a paid search API by setting the appropriate environment
variables.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
from typing import Optional

import httpx

from src.models import SearchResult


class MCPClient:
    """Thin MCP-style client for querying the internet.

    In a full MCP deployment the server would be a separate process exposing
    standardised tools (``search``, ``fetch``, ``summarize``, …).  Here we
    implement the same interface directly so that the rest of the system is
    decoupled from the transport layer.
    """

    # DuckDuckGo instant-answer API (no key required)
    _DDG_API = "https://api.duckduckgo.com/"
    # Fallback: DuckDuckGo HTML search
    _DDG_HTML = "https://html.duckduckgo.com/html/"

    def __init__(
        self,
        timeout: float = 10.0,
        max_results: int = 5,
    ) -> None:
        self.timeout = timeout
        self.max_results = max_results
        self._client = httpx.Client(
            headers={
                "User-Agent": (
                    "IntuitionScientist/1.0 (educational research agent)"
                )
            },
            timeout=timeout,
            follow_redirects=True,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(self, query: str, *, num_results: Optional[int] = None) -> list[SearchResult]:
        """Run a web search and return structured results.

        Tries the DDG Instant-Answer JSON API first; falls back to HTML
        scraping if the JSON API returns no useful data.
        """
        n = num_results or self.max_results
        results = self._search_ddg_json(query, n)
        if not results:
            results = self._search_ddg_html(query, n)
        return results[:n]

    def fetch_page(self, url: str) -> str:
        """Fetch the text content of a single URL.

        Returns the first 4 000 characters of the response body stripped of
        HTML tags, or an empty string on error.
        """
        try:
            resp = self._client.get(url)
            resp.raise_for_status()
            text = re.sub(r"<[^>]+>", " ", resp.text)
            text = re.sub(r"\s{2,}", " ", text).strip()
            return text[:4000]
        except Exception:
            return ""

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "MCPClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _search_ddg_json(self, query: str, n: int) -> list[SearchResult]:
        """Query DuckDuckGo Instant Answer API."""
        try:
            resp = self._client.get(
                self._DDG_API,
                params={"q": query, "format": "json", "no_html": "1"},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []

        results: list[SearchResult] = []

        # RelatedTopics contains topic cards
        for topic in data.get("RelatedTopics", []):
            if len(results) >= n:
                break
            if isinstance(topic, dict) and "Text" in topic:
                results.append(
                    SearchResult(
                        title=topic.get("Text", "")[:100],
                        url=topic.get("FirstURL", ""),
                        snippet=topic.get("Text", ""),
                    )
                )

        # Abstract result
        if data.get("AbstractText") and len(results) < n:
            results.insert(
                0,
                SearchResult(
                    title=data.get("Heading", query),
                    url=data.get("AbstractURL", ""),
                    snippet=data.get("AbstractText", ""),
                    relevance_score=1.0,
                ),
            )

        return results

    def _search_ddg_html(self, query: str, n: int) -> list[SearchResult]:
        """Fall back to scraping DuckDuckGo HTML results."""
        try:
            resp = self._client.post(
                self._DDG_HTML,
                data={"q": query},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            resp.raise_for_status()
            html = resp.text
        except Exception:
            return []

        results: list[SearchResult] = []
        # Simple regex-based extraction (avoids a heavyweight HTML parser dep)
        anchors = re.findall(
            r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>',
            html,
        )
        snippets = re.findall(r'<a[^>]+class="result__snippet"[^>]*>([^<]+)</a>', html)

        for i, (url, title) in enumerate(anchors):
            if len(results) >= n:
                break
            snippet = snippets[i] if i < len(snippets) else ""
            results.append(
                SearchResult(
                    title=title.strip(),
                    url=url,
                    snippet=snippet.strip(),
                )
            )

        return results
