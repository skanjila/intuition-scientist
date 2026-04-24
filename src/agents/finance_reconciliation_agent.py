"""Finance Reconciliation agent (Proposal 4).

Specialises in three-way match (PO → GR → Invoice), GL reconciliation,
variance explanation, and audit-narrative generation.

Market context
--------------
Finance teams at mid-size enterprises spend 15–25 person-days per month-end
close on manual reconciliation.  AI-assisted reconciliation reduces this to
2–4 days and catches anomalies that rule-based systems miss.
"""

from src.agents.base_agent import BaseAgent
from src.models import Domain, SearchResult


class FinanceReconciliationAgent(BaseAgent):
    """Expert in financial reconciliation, audit, and closing processes."""

    domain = Domain.FINANCE_RECONCILIATION

    def _build_system_prompt(self) -> str:
        return (
            "You are a world-class finance controller and chartered accountant with\n"
            "deep expertise in:\n"
            "- Three-way match: purchase order (PO) ↔ goods receipt (GR) ↔ invoice\n"
            "  matching; tolerance policies; variance classification\n"
            "- General ledger reconciliation: account-level tie-out, suspense\n"
            "  clearing, intercompany eliminations, FX revaluation\n"
            "- Anomaly detection: duplicate payments, round-number bias, Benford's\n"
            "  Law analysis, velocity outliers, ghost vendors\n"
            "- Audit trail generation: GAAP/IFRS compliant narratives, supporting\n"
            "  schedules, journal entry documentation, SOX evidence packs\n"
            "- Month-end close: close calendar management, accruals, prepayments,\n"
            "  cut-off testing, materiality thresholds\n"
            "- ERP systems: SAP FI/CO, Oracle Fusion Financials, NetSuite, Xero —\n"
            "  transaction codes, document types, posting keys\n"
            "- Regulatory: SOX 302/404, PCAOB AS2201, IFRS 9/15/16, ASC 606\n\n"
            "=== RECONCILIATION PROTOCOL ===\n"
            "1. Identify the matching dimension (amount, date, vendor, PO number).\n"
            "2. Flag all items outside the tolerance band.\n"
            "3. Classify variances: timing, pricing, quantity, currency, tax.\n"
            "4. Propose correcting journal entries for each unmatched item.\n"
            "5. Generate an audit-ready narrative suitable for the CFO pack.\n\n"
            "Be precise with numbers. Flag items requiring controller sign-off.\n"
            "Respond only with the requested JSON structure."
        )

    def _compute_weights(
        self,
        question: str,
        mcp_results: list[SearchResult],
    ) -> tuple[float, float]:
        """Override: reconciliation is primarily data-driven, not web-search-driven."""
        # Use a moderate tool weight — structured-data results are highly relevant
        # but the agent's accounting knowledge is the primary reasoning engine.
        mcp_quality = min(1.0, len(mcp_results) / 4)
        tool_w = max(0.35, min(0.65, 0.45 + mcp_quality * 0.20))
        return round(1.0 - tool_w, 3), round(tool_w, 3)
