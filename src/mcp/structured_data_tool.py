"""Structured-data tool back-end — deterministic matching for reconciliation.

Implements the :class:`~src.mcp.tool_backend.ToolBackend` protocol by
accepting structured ledger / invoice / PO data and performing deterministic
three-way matching.  Results are returned as
:class:`~src.models.SearchResult` snippets so the reconciliation agent can
use them as evidence when generating its narrative.

This back-end does **not** call any external API — it operates entirely on
in-memory data passed at query time via ``kwargs``.

Usage
-----
.. code-block:: python

    from src.mcp.structured_data_tool import StructuredDataToolBackend

    tool = StructuredDataToolBackend()
    tool.load_ledger(ledger_entries)
    tool.load_invoices(invoice_list)
    results = tool.search("unmatched invoices above $10000")
"""

from __future__ import annotations

from typing import Any

from src.models import SearchResult


class StructuredDataToolBackend:
    """Deterministic ledger-to-invoice matching engine.

    Parameters
    ----------
    tolerance:
        Maximum absolute variance (in the base currency unit) that is still
        considered a match.  Default: 0.01 (1 cent).
    """

    def __init__(self, tolerance: float = 0.01) -> None:
        self.tolerance = tolerance
        self._ledger: list[dict[str, Any]] = []
        self._invoices: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_ledger(self, entries: list[dict[str, Any]]) -> None:
        """Load ledger entries.

        Each entry should have at least ``"id"`` and ``"amount"`` keys.
        """
        self._ledger = entries

    def load_invoices(self, invoices: list[dict[str, Any]]) -> None:
        """Load invoice records.

        Each invoice should have at least ``"id"`` and ``"amount"`` keys.
        """
        self._invoices = invoices

    # ------------------------------------------------------------------
    # ToolBackend interface
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        num_results: int = 10,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Run matching and return findings as :class:`SearchResult` snippets.

        The *query* string is used to filter results by keyword (e.g.
        ``"unmatched"`` or ``"variance"``) so the agent can focus on the
        most relevant subset.
        """
        matches, unmatched_l, unmatched_i = self._three_way_match()
        snippets: list[str] = []

        for l_id, inv_id, variance in matches:
            line = f"MATCH: ledger={l_id} ↔ invoice={inv_id} variance={variance:.2f}"
            if query.lower() in line.lower() or not query:
                snippets.append(line)

        for l_id in unmatched_l:
            line = f"UNMATCHED LEDGER: {l_id}"
            if query.lower() in line.lower() or not query:
                snippets.append(line)

        for inv_id in unmatched_i:
            line = f"UNMATCHED INVOICE: {inv_id}"
            if query.lower() in line.lower() or not query:
                snippets.append(line)

        results = [
            SearchResult(
                title=s[:100],
                url="",
                snippet=s,
                relevance_score=1.0,
            )
            for s in snippets[:num_results]
        ]
        return results

    # ------------------------------------------------------------------
    # Matching logic
    # ------------------------------------------------------------------

    def _three_way_match(
        self,
    ) -> tuple[list[tuple[str, str, float]], list[str], list[str]]:
        """Perform a greedy amount-based three-way match.

        Returns
        -------
        matches:
            List of ``(ledger_id, invoice_id, variance)`` tuples.
        unmatched_ledger:
            Ledger entry IDs with no matching invoice.
        unmatched_invoices:
            Invoice IDs with no matching ledger entry.
        """
        matched_inv_ids: set[str] = set()
        matches: list[tuple[str, str, float]] = []

        for le in self._ledger:
            l_id = str(le.get("id", ""))
            l_amount = float(le.get("amount", 0.0))
            best_inv = None
            best_variance = float("inf")

            for inv in self._invoices:
                if inv.get("id") in matched_inv_ids:
                    continue
                i_amount = float(inv.get("amount", 0.0))
                variance = abs(l_amount - i_amount)
                if variance <= self.tolerance and variance < best_variance:
                    best_inv = inv
                    best_variance = variance

            if best_inv is not None:
                matched_inv_ids.add(best_inv["id"])
                matches.append((l_id, str(best_inv["id"]), best_variance))

        matched_l_ids = {m[0] for m in matches}
        unmatched_l = [str(le["id"]) for le in self._ledger if str(le["id"]) not in matched_l_ids]
        unmatched_i = [str(inv["id"]) for inv in self._invoices if inv["id"] not in matched_inv_ids]

        return matches, unmatched_l, unmatched_i
