"""Shared data models for the Business Agent Platform.

Architecture overview
---------------------
This module defines every data contract used across the 12 business use
cases.  There are four conceptual layers:

1. **Domain registry** — :class:`Domain` enum listing every agent domain
   that targets a real business problem.

2. **Human-AI balance primitives** — :class:`HumanJudgment`,
   :class:`AutonomyLevel`, and :class:`EscalationDecision` express *how*
   human judgment is blended with AI recommendations at each decision point.
   See ``docs/BUSINESS_USE_CASES.md`` §Human-AI Balance for the design
   rationale.

3. **Core exchange types** — :class:`AgentResponse` and
   :class:`SearchResult` are the atomic outputs produced by every agent and
   tool backend.

4. **Per-use-case result types** — one typed dataclass per business use
   case so callers always get strongly-typed, predictable outputs.

Human-AI Balance
----------------
Every orchestrator entry point accepts an optional :class:`HumanJudgment`
and an :class:`AutonomyLevel`.  The blend algorithm is:

    human_weight = _AUTONOMY_BASE_WEIGHT[autonomy_level]
    if human_judgment.override:
        human_weight = 1.0                # human fully overrides AI
    elif ai_confidence < ESCALATION_THRESHOLD:
        needs_escalation = True           # request human input
    final = human_weight * human + (1 - human_weight) * ai

Default autonomy levels by use case:

    ========================  =================  =============================
    Use Case                  Default Autonomy   Escalation Triggers
    ========================  =================  =============================
    Customer Support Triage   AI_PROPOSES        P1/P2, "lawsuit", "breach"
    Compliance Q&A            AI_ASSISTS         any legal interpretation
    Incident Response         AI_PROPOSES        P1, >50 users impacted
    Finance Reconciliation    AI_PROPOSES        variance > materiality
    Sales Outreach            AI_ASSISTS         deal > $500K
    Analytics Reporting       FULL_AUTO          anomaly > 3σ
    Code Review               AI_PROPOSES        critical/high severity
    Supply Chain Exception    AI_PROPOSES        cost > budget, no alt supplier
    RFP Drafting              AI_ASSISTS         unlimited liability clause
    ========================  =================  =============================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------


class Domain(str, Enum):
    """Agent domains — every value maps to exactly one business use case or
    a supporting capability used by multiple use cases.

    Primary use-case domains
    ------------------------
    CUSTOMER_SUPPORT        → Use case 1: Customer Support Triage
    LEGAL_COMPLIANCE        → Use case 2: Compliance Policy Q&A (+ RFP support)
    INCIDENT_RESPONSE       → Use case 3: Incident Response / On-Call
    FINANCE_RECONCILIATION  → Use case 4: Finance Reconciliation
    MARKETING_GROWTH        → Use case 5: Sales Research & Outreach (support)
    ANALYTICS               → Use case 6: Analytics / Report Generation
    CODE_REVIEW             → Use case 7: Code Review / PR Assistant
    SUPPLY_CHAIN            → Use case 8: Supply Chain Exception Management
    RFP_RESPONSE            → Use case 9: RFP Response Drafting

    Supporting domains used across multiple use cases
    -------------------------------------------------
    FINANCE_ECONOMICS       → Use cases 4, 5, 8 (financial analysis)
    CYBERSECURITY           → Use cases 3, 7 (security analysis)
    ENTERPRISE_ARCHITECTURE → Use case 3 (system design context)
    STRATEGY_INTELLIGENCE   → Use cases 5, 9 (competitive intelligence)
    ORGANIZATIONAL_BEHAVIOR → Use case 1 (de-escalation, tone)
    """

    # ── Primary use-case domains ──────────────────────────────────────
    CUSTOMER_SUPPORT = "customer_support"
    INCIDENT_RESPONSE = "incident_response"
    FINANCE_RECONCILIATION = "finance_reconciliation"
    CODE_REVIEW = "code_review"
    ANALYTICS = "analytics"
    RFP_RESPONSE = "rfp_response"

    # ── Supporting domains ────────────────────────────────────────────
    LEGAL_COMPLIANCE = "legal_compliance"
    ENTERPRISE_ARCHITECTURE = "enterprise_architecture"
    MARKETING_GROWTH = "marketing_growth"
    ORGANIZATIONAL_BEHAVIOR = "organizational_behavior"
    STRATEGY_INTELLIGENCE = "strategy_intelligence"
    FINANCE_ECONOMICS = "finance_economics"
    CYBERSECURITY = "cybersecurity"
    SUPPLY_CHAIN = "supply_chain"


# ---------------------------------------------------------------------------
# Human-AI balance primitives
# ---------------------------------------------------------------------------


class AutonomyLevel(str, Enum):
    """Controls how much weight the AI recommendation carries vs. human judgment.

    Use this to configure the risk tolerance for each business use case or
    individual invocation.

    FULL_AUTO
        AI decides and acts; human receives a summary notification.
        Suitable for low-stakes, high-volume decisions (P4 tickets, style
        comments in code review, informational report sections).

    AI_PROPOSES
        AI makes a concrete recommendation; human approves or rejects before
        any action is taken.  Suitable for medium-stakes decisions (P3 ticket
        routing, reconciliation journal entries, supply-chain substitutions).
        *Default for most use cases.*

    AI_ASSISTS
        Human makes the primary decision with AI-generated analysis as
        supporting material.  Suitable for high-stakes or judgment-heavy
        decisions (legal interpretations, large-deal outreach, RFP strategy).

    HUMAN_FIRST
        Human judges first; AI validates, flags gaps, and adds evidence.
        Suitable for the highest-stakes decisions (P1 incident escalation,
        unlimited-liability contract clauses, regulatory enforcement actions).
    """

    FULL_AUTO = "full_auto"
    AI_PROPOSES = "ai_proposes"
    AI_ASSISTS = "ai_assists"
    HUMAN_FIRST = "human_first"


#: Base human-weight by autonomy level (before confidence adjustment).
#: At FULL_AUTO the human weight is 0 — AI acts alone.
#: At HUMAN_FIRST the human weight is 0.80 — human's judgment dominates.
AUTONOMY_BASE_WEIGHT: dict[AutonomyLevel, float] = {
    AutonomyLevel.FULL_AUTO: 0.00,
    AutonomyLevel.AI_PROPOSES: 0.20,
    AutonomyLevel.AI_ASSISTS: 0.50,
    AutonomyLevel.HUMAN_FIRST: 0.80,
}

#: AI confidence below this threshold always triggers an escalation request,
#: regardless of autonomy level.
ESCALATION_CONFIDENCE_THRESHOLD: float = 0.55


@dataclass
class HumanJudgment:
    """Structured human input captured at a decision checkpoint.

    When supplied to an orchestrator method the system blends it with the
    AI recommendation according to the configured :class:`AutonomyLevel`.

    Parameters
    ----------
    context:
        A brief description of what the human is being asked to judge
        (e.g. ``"Urgency classification for ticket #45821"``).
    judgment:
        The human's actual decision, opinion, or correction in plain text.
    confidence:
        How confident the human is in their judgment (0.0–1.0).  A lower
        confidence raises the effective weight of the AI recommendation.
    override:
        When ``True`` the human completely overrides the AI; the blend
        weight becomes 1.0 regardless of ``AutonomyLevel``.
    notes:
        Any additional context the human wants to record (e.g. rationale,
        constraints, caveats).

    Usage example
    -------------
    .. code-block:: python

        from src.models import HumanJudgment, AutonomyLevel
        from src.orchestrator.business_orchestrator import BusinessOrchestrator

        orch = BusinessOrchestrator()
        judgment = HumanJudgment(
            context="Ticket urgency classification",
            judgment="This is P2 — customer is on an enterprise SLA",
            confidence=0.9,
        )
        result = orch.triage(
            ticket_text="Payment API is returning 500 errors for all enterprise users",
            human_judgment=judgment,
            autonomy=AutonomyLevel.AI_PROPOSES,
        )
    """

    context: str
    judgment: str
    confidence: float = 0.5
    override: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("HumanJudgment.confidence must be between 0.0 and 1.0")


@dataclass
class EscalationDecision:
    """Describes whether and why a decision needs human review.

    Produced automatically by each orchestrator method based on the AI
    confidence score and use-case-specific escalation rules.

    Parameters
    ----------
    needs_escalation:
        ``True`` when human review is recommended before acting on the
        AI result.
    reason:
        Plain-English explanation of why escalation was triggered
        (e.g. ``"P1 urgency detected — SLA breach imminent"``).
    urgency:
        One of ``"immediate"`` (act within minutes), ``"review"`` (act
        within hours), or ``"informational"`` (FYI, no action required).
    checkpoint:
        The specific decision or action that needs human input
        (e.g. ``"Approve routing to Security team"``).
    """

    needs_escalation: bool
    reason: str
    urgency: str = "review"          # "immediate" | "review" | "informational"
    checkpoint: str = ""


# ---------------------------------------------------------------------------
# Core exchange types
# ---------------------------------------------------------------------------


@dataclass
class AgentResponse:
    """A domain-specific agent's answer to any query.

    This is the atomic output produced by every :class:`~src.agents.base_agent.BaseAgent`
    subclass.  The orchestrator collects a list of these and synthesises them
    into a use-case-specific typed result.

    Parameters
    ----------
    domain:
        Which agent produced this response.
    answer:
        The agent's final blended answer.
    reasoning:
        Step-by-step reasoning chain (may be truncated for display).
    confidence:
        Blended confidence score (0.0–1.0).
    sources:
        URLs or references cited by the tool-grounded pipeline.
    mcp_context:
        Raw tool/MCP context used in the tool-grounded pipeline.
        Empty when the tool pipeline was skipped or returned nothing.
    intuition_weight:
        Fraction of the final answer drawn from the knowledge-only pipeline.
    tool_weight:
        Fraction drawn from the tool-grounded pipeline.
    """

    domain: Domain
    answer: str
    reasoning: str
    confidence: float
    sources: list[str] = field(default_factory=list)
    mcp_context: str = ""
    intuition_weight: float = 0.5
    tool_weight: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("AgentResponse.confidence must be between 0.0 and 1.0")


@dataclass
class SearchResult:
    """A single result returned by any :class:`~src.mcp.tool_backend.ToolBackend`.

    Parameters
    ----------
    title:
        Short title or identifier for the result.
    url:
        Source URL or API endpoint path.  May be empty for internal data.
    snippet:
        Relevant excerpt (≤ 300 chars) shown to the agent as context.
    relevance_score:
        Backend-reported relevance (0.0–1.0).  ``None`` when unavailable.
    """

    title: str
    url: str
    snippet: str
    relevance_score: Optional[float] = None


# ---------------------------------------------------------------------------
# Use case 1 — Customer Support Triage
# ---------------------------------------------------------------------------


@dataclass
class TriageResult:
    """Customer support triage output (Use case 1).

    Produced by :meth:`~src.orchestrator.business_orchestrator.BusinessOrchestrator.triage`.

    How to run
    ----------
    .. code-block:: python

        from src.orchestrator.business_orchestrator import BusinessOrchestrator
        from src.models import HumanJudgment, AutonomyLevel

        orch = BusinessOrchestrator()
        result = orch.triage(
            "Payment API returning 500 errors for all enterprise users",
            human_judgment=HumanJudgment(
                context="Urgency check",
                judgment="Definitely P1 — this is revenue-impacting",
                confidence=0.95,
            ),
            autonomy=AutonomyLevel.AI_PROPOSES,
        )
        print(result.urgency, result.routing_department)
        print(result.draft_response)

    See ``docs/use_cases/01_customer_support_triage.md`` for full documentation.
    """

    ticket_text: str
    urgency: str                      # "P1" | "P2" | "P3" | "P4"
    routing_department: str
    draft_response: str
    kb_articles: list[str] = field(default_factory=list)
    escalation: EscalationDecision = field(
        default_factory=lambda: EscalationDecision(needs_escalation=False, reason="")
    )
    human_judgment: Optional[HumanJudgment] = None
    autonomy_used: AutonomyLevel = AutonomyLevel.AI_PROPOSES
    ai_confidence: float = 0.5
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Use case 2 — Compliance Policy Q&A (result is AgentResponse + escalation)
# ---------------------------------------------------------------------------


@dataclass
class ComplianceAnswer:
    """Compliance Q&A output (Use case 2).

    Produced by :meth:`~src.orchestrator.business_orchestrator.BusinessOrchestrator.compliance_qa`.

    How to run
    ----------
    .. code-block:: python

        from src.orchestrator.business_orchestrator import BusinessOrchestrator
        from src.mcp.vector_store_backend import VectorStoreBackend

        store = VectorStoreBackend()
        store.add_documents([
            {"id": "gdpr-art17", "text": "GDPR Art.17: Right to erasure...", "source": "gdpr.pdf"},
        ])
        orch = BusinessOrchestrator()
        result = orch.compliance_qa(
            "Do we need to delete all user data on account closure under GDPR?",
            tool_backend=store,
        )
        print(result.answer)
        print(result.cited_sections)

    See ``docs/use_cases/02_compliance_qa.md`` for full documentation.
    """

    question: str
    answer: str
    cited_sections: list[str] = field(default_factory=list)
    escalation: EscalationDecision = field(
        default_factory=lambda: EscalationDecision(needs_escalation=False, reason="")
    )
    human_judgment: Optional[HumanJudgment] = None
    autonomy_used: AutonomyLevel = AutonomyLevel.AI_ASSISTS
    ai_confidence: float = 0.5
    reasoning: str = ""
    sources: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Use case 3 — Incident Response
# ---------------------------------------------------------------------------


@dataclass
class IncidentContext:
    """Input context for incident-response triage (Use case 3).

    Parameters
    ----------
    alert_payload:
        Raw alert text or JSON string from PagerDuty / Datadog / Opsgenie.
    log_lines:
        Recent log excerpts relevant to the alert (≤ 50 lines recommended).
    service_graph:
        Optional description of upstream/downstream service dependencies.
    past_incidents:
        Optional list of similar past incident summaries for pattern matching.
    """

    alert_payload: str
    log_lines: list[str] = field(default_factory=list)
    service_graph: str = ""
    past_incidents: list[str] = field(default_factory=list)


@dataclass
class IncidentResponse:
    """Structured incident-response recommendation (Use case 3).

    Produced by :meth:`~src.orchestrator.business_orchestrator.BusinessOrchestrator.respond_to_incident`.

    How to run
    ----------
    .. code-block:: python

        from src.orchestrator.business_orchestrator import BusinessOrchestrator
        from src.models import IncidentContext

        ctx = IncidentContext(
            alert_payload="[P1] api-gateway error rate 45% (threshold: 2%)",
            log_lines=["WARN payments-service: connection refused", "..."],
            service_graph="api-gateway → payments-service → postgres-primary",
        )
        orch = BusinessOrchestrator()
        result = orch.respond_to_incident(ctx)
        for step in result.mitigation_steps:
            print(step)

    See ``docs/use_cases/03_incident_response.md`` for full documentation.
    """

    alert_payload: str
    root_cause_hypotheses: list[str] = field(default_factory=list)
    mitigation_steps: list[str] = field(default_factory=list)
    runbook_links: list[str] = field(default_factory=list)
    severity: str = "P3"
    escalation: EscalationDecision = field(
        default_factory=lambda: EscalationDecision(needs_escalation=False, reason="")
    )
    human_judgment: Optional[HumanJudgment] = None
    autonomy_used: AutonomyLevel = AutonomyLevel.AI_PROPOSES
    ai_confidence: float = 0.5
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Use case 4 — Finance Reconciliation
# ---------------------------------------------------------------------------


@dataclass
class ReconciliationMatch:
    """One matched ledger ↔ invoice pair (Use case 4)."""

    ledger_id: str
    invoice_id: str
    match_confidence: float
    variance: float = 0.0
    note: str = ""


@dataclass
class ReconciliationResult:
    """Finance reconciliation output (Use case 4).

    Produced by :meth:`~src.orchestrator.business_orchestrator.BusinessOrchestrator.reconcile`.

    How to run
    ----------
    .. code-block:: python

        from src.orchestrator.business_orchestrator import BusinessOrchestrator
        from src.mcp.structured_data_tool import StructuredDataToolBackend

        tool = StructuredDataToolBackend(tolerance=0.01)
        tool.load_ledger([{"id": "L001", "amount": 1000.00}])
        tool.load_invoices([{"id": "INV-42", "amount": 1000.00}])

        orch = BusinessOrchestrator()
        result = orch.reconcile(
            ledger=[{"id": "L001", "amount": 1000.00}],
            invoices=[{"id": "INV-42", "amount": 1000.00}],
            tool_backend=tool,
        )
        print(result.audit_narrative)

    See ``docs/use_cases/04_finance_reconciliation.md`` for full documentation.
    """

    matched_pairs: list[ReconciliationMatch] = field(default_factory=list)
    unmatched_ledger_ids: list[str] = field(default_factory=list)
    unmatched_invoice_ids: list[str] = field(default_factory=list)
    anomaly_scores: dict[str, float] = field(default_factory=dict)
    audit_narrative: str = ""
    suggested_journals: list[str] = field(default_factory=list)
    escalation: EscalationDecision = field(
        default_factory=lambda: EscalationDecision(needs_escalation=False, reason="")
    )
    human_judgment: Optional[HumanJudgment] = None
    autonomy_used: AutonomyLevel = AutonomyLevel.AI_PROPOSES
    ai_confidence: float = 0.5


# ---------------------------------------------------------------------------
# Use case 5 — Sales Research & Outreach
# ---------------------------------------------------------------------------


@dataclass
class OutreachResult:
    """Sales research and outreach output (Use case 5).

    Produced by :meth:`~src.orchestrator.business_orchestrator.BusinessOrchestrator.outreach`.

    How to run
    ----------
    .. code-block:: python

        from src.orchestrator.business_orchestrator import BusinessOrchestrator

        orch = BusinessOrchestrator()
        result = orch.outreach(
            company="Acme Corp",
            product="DataPlatform Pro",
            deal_value=250_000,
        )
        print(result.email_draft)
        for q in result.discovery_questions:
            print(" •", q)

    See ``docs/use_cases/05_sales_outreach.md`` for full documentation.
    """

    company_name: str
    company_summary: str
    pain_points: list[str] = field(default_factory=list)
    email_draft: str = ""
    meddic_score: float = 0.0
    discovery_questions: list[str] = field(default_factory=list)
    competitive_notes: str = ""
    escalation: EscalationDecision = field(
        default_factory=lambda: EscalationDecision(needs_escalation=False, reason="")
    )
    human_judgment: Optional[HumanJudgment] = None
    autonomy_used: AutonomyLevel = AutonomyLevel.AI_ASSISTS
    ai_confidence: float = 0.5


# ---------------------------------------------------------------------------
# Use case 6 — Analytics / Report Generation
# ---------------------------------------------------------------------------


@dataclass
class ReportContext:
    """Input specification for analytics report generation (Use case 6).

    Parameters
    ----------
    metrics:
        Dictionary of metric name → current value (numeric or string).
        Example: ``{"revenue_usd": 4_200_000, "churn_pct": 1.8}``.
    kpi_targets:
        Optional dictionary of metric name → target value for comparison.
    reporting_period:
        Human-readable period label, e.g. ``"2026-W16"`` or ``"Q1 2026"``.
    audience:
        Report audience — ``"exec"`` (1-page brief), ``"ops"`` (operational
        detail), or ``"tech"`` (technical deep-dive with SQL).
    """

    metrics: dict[str, object]
    kpi_targets: dict[str, object] = field(default_factory=dict)
    reporting_period: str = ""
    audience: str = "ops"            # "exec" | "ops" | "tech"


@dataclass
class ReportResult:
    """Analytics report output (Use case 6).

    Produced by :meth:`~src.orchestrator.business_orchestrator.BusinessOrchestrator.generate_report`.

    How to run
    ----------
    .. code-block:: python

        from src.orchestrator.business_orchestrator import BusinessOrchestrator
        from src.models import ReportContext

        ctx = ReportContext(
            metrics={"revenue_usd": 4_200_000, "churn_pct": 1.8, "cac_usd": 420},
            kpi_targets={"revenue_usd": 4_000_000, "churn_pct": 2.0},
            reporting_period="2026-W16",
            audience="exec",
        )
        orch = BusinessOrchestrator()
        result = orch.generate_report(ctx)
        print(result.headline)
        print(result.trend_analysis)

    See ``docs/use_cases/06_analytics_reporting.md`` for full documentation.
    """

    headline: str
    trend_analysis: str
    anomalies: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    chart_recommendations: list[str] = field(default_factory=list)
    sql_queries: list[str] = field(default_factory=list)
    escalation: EscalationDecision = field(
        default_factory=lambda: EscalationDecision(needs_escalation=False, reason="")
    )
    human_judgment: Optional[HumanJudgment] = None
    autonomy_used: AutonomyLevel = AutonomyLevel.FULL_AUTO
    ai_confidence: float = 0.5


# ---------------------------------------------------------------------------
# Use case 7 — Code Review / PR Assistant
# ---------------------------------------------------------------------------


@dataclass
class CodeReviewComment:
    """One inline code-review comment (Use case 7).

    Parameters
    ----------
    file_path:
        Relative path to the reviewed file.
    line:
        Line number (1-indexed).  Use 0 for a file-level comment.
    severity:
        ``"critical"`` | ``"high"`` | ``"medium"`` | ``"low"`` | ``"info"``
    category:
        ``"security"`` | ``"logic"`` | ``"performance"`` | ``"style"``
        | ``"test"`` | ``"docs"``
    message:
        Concise description of the issue.
    suggestion:
        Concrete fix or recommended change.
    """

    file_path: str
    line: int
    severity: str
    category: str
    message: str
    suggestion: str = ""


@dataclass
class CodeReviewResult:
    """Structured code-review output (Use case 7).

    Produced by :meth:`~src.orchestrator.business_orchestrator.BusinessOrchestrator.review_pr`.

    How to run
    ----------
    .. code-block:: python

        from src.orchestrator.business_orchestrator import BusinessOrchestrator

        diff = open("my_feature.diff").read()
        orch = BusinessOrchestrator()
        result = orch.review_pr(
            diff=diff,
            description="Add OAuth2 login endpoint",
        )
        for comment in result.comments:
            print(f"[{comment.severity}] {comment.file_path}:{comment.line}")
            print(f"  {comment.message}")
        print("Risk score:", result.risk_score)
        print("Recommendation:", result.approval_recommendation)

    See ``docs/use_cases/07_code_review.md`` for full documentation.
    """

    diff_summary: str
    comments: list[CodeReviewComment] = field(default_factory=list)
    risk_score: float = 0.0
    security_flags: list[str] = field(default_factory=list)
    approval_recommendation: str = "needs_review"  # "approve"|"request_changes"|"needs_review"
    overall_reasoning: str = ""
    escalation: EscalationDecision = field(
        default_factory=lambda: EscalationDecision(needs_escalation=False, reason="")
    )
    human_judgment: Optional[HumanJudgment] = None
    autonomy_used: AutonomyLevel = AutonomyLevel.AI_PROPOSES
    ai_confidence: float = 0.5


# ---------------------------------------------------------------------------
# Use case 8 — Supply Chain Exception Management
# ---------------------------------------------------------------------------


@dataclass
class ExceptionEvent:
    """Supply-chain exception input (Use case 8).

    Parameters
    ----------
    exception_type:
        One of ``"late_delivery"``, ``"stockout"``, ``"supplier_failure"``,
        ``"quality_hold"``, ``"demand_spike"``.
    sku:
        SKU or product identifier affected.
    supplier:
        Primary supplier name or ID.
    severity:
        ``"low"`` | ``"medium"`` | ``"high"`` | ``"critical"``
    eta_days_late:
        How many days late the shipment / replenishment is expected to be.
    current_inventory:
        Current on-hand stock quantity.
    cost_to_expedite:
        Estimated cost (base currency) to expedite the order.
    notes:
        Any free-text context from the supply chain team.
    """

    exception_type: str
    sku: str
    supplier: str
    severity: str = "medium"
    eta_days_late: int = 0
    current_inventory: float = 0.0
    cost_to_expedite: float = 0.0
    notes: str = ""


@dataclass
class ExceptionResponse:
    """Supply-chain exception management output (Use case 8).

    Produced by :meth:`~src.orchestrator.business_orchestrator.BusinessOrchestrator.handle_exception`.

    How to run
    ----------
    .. code-block:: python

        from src.orchestrator.business_orchestrator import BusinessOrchestrator
        from src.models import ExceptionEvent

        event = ExceptionEvent(
            exception_type="late_delivery",
            sku="SKU-12345",
            supplier="SupplierA",
            severity="high",
            eta_days_late=14,
            current_inventory=145,
            cost_to_expedite=8500.0,
        )
        orch = BusinessOrchestrator()
        result = orch.handle_exception(event)
        print(result.recommended_action)
        print(f"Financial impact: ${result.financial_impact_estimate:,.0f}")

    See ``docs/use_cases/08_supply_chain_exceptions.md`` for full documentation.
    """

    exception_type: str
    sku: str
    recommended_action: str          # "expedite"|"substitute"|"backorder"|"cancel"|"escalate"
    financial_impact_estimate: float = 0.0
    alternative_suppliers: list[str] = field(default_factory=list)
    draft_po_lines: list[str] = field(default_factory=list)
    escalation: EscalationDecision = field(
        default_factory=lambda: EscalationDecision(needs_escalation=False, reason="")
    )
    human_judgment: Optional[HumanJudgment] = None
    autonomy_used: AutonomyLevel = AutonomyLevel.AI_PROPOSES
    ai_confidence: float = 0.5
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Use case 9 — RFP Response Drafting
# ---------------------------------------------------------------------------


@dataclass
class RFPResult:
    """RFP response drafting output (Use case 9).

    Produced by :meth:`~src.orchestrator.business_orchestrator.BusinessOrchestrator.draft_rfp`.

    How to run
    ----------
    .. code-block:: python

        from src.orchestrator.business_orchestrator import BusinessOrchestrator

        rfp_text = open("rfp_document.txt").read()
        orch = BusinessOrchestrator()
        result = orch.draft_rfp(
            rfp_text=rfp_text,
            rfp_title="Enterprise Data Platform RFP — Acme Corp",
        )
        for section, draft in result.section_drafts.items():
            print(f"## {section}\\n{draft}\\n")
        print("Risk flags:", result.risk_flags)

    See ``docs/use_cases/09_rfp_drafting.md`` for full documentation.
    """

    rfp_title: str
    section_drafts: dict[str, str] = field(default_factory=dict)
    compliance_matrix: dict[str, str] = field(default_factory=dict)
    win_themes: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    overall_strategy: str = ""
    escalation: EscalationDecision = field(
        default_factory=lambda: EscalationDecision(needs_escalation=False, reason="")
    )
    human_judgment: Optional[HumanJudgment] = None
    autonomy_used: AutonomyLevel = AutonomyLevel.AI_ASSISTS
    ai_confidence: float = 0.5
