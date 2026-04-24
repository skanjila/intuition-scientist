"""Shared data models for the AI Agent Platform.

Architecture overview
---------------------
This module defines every data contract used across business and medical
AI agent use cases. There are four conceptual layers:

1. **Domain registry** — :class:`Domain` enum listing every agent domain.

2. **Human-AI balance primitives** — :class:`HumanJudgment`,
   :class:`AutonomyLevel`, and :class:`EscalationDecision` express *how*
   human judgment is blended with AI recommendations at each decision point.

3. **Core exchange types** — :class:`AgentResponse` and
   :class:`SearchResult` are the atomic outputs produced by every agent and
   tool backend.

4. **Per-use-case result types** — one typed dataclass per business or medical
   use case so callers always get strongly-typed, predictable outputs.

Human-AI Balance
----------------
Every orchestrator entry point accepts an optional :class:`HumanJudgment`
and an :class:`AutonomyLevel`.  The blend algorithm is::

    human_weight = AUTONOMY_BASE_WEIGHT[autonomy_level]
    if human_judgment.override:
        human_weight = 1.0                # human fully overrides AI
    elif ai_confidence < ESCALATION_CONFIDENCE_THRESHOLD:
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
    Clinical Decision         HUMAN_FIRST        always — physician must validate
    Drug Interaction          AI_PROPOSES        always — pharmacist must validate
    Mental Health Triage      HUMAN_FIRST        always — clinician must review
    Genomic Risk              HUMAN_FIRST        always — genetic counselor required
    ========================  =================  =============================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Optional


# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------


class Domain(str, Enum):
    """Agent domains — every value maps to exactly one business or medical use
    case, or a supporting capability used by multiple use cases.

    New business domains
    --------------------
    CUSTOMER_SUPPORT        → Use case 1: Customer Support Triage
    INCIDENT_RESPONSE       → Use case 2: Incident Response / On-Call
    FINANCE_RECONCILIATION  → Use case 3: Finance Reconciliation
    CODE_REVIEW             → Use case 4: Code Review / PR Assistant
    ANALYTICS               → Use case 5: Analytics / Report Generation
    RFP_RESPONSE            → Use case 6: RFP Response Drafting
    LEGAL_COMPLIANCE        → Use case 7: Compliance Policy Q&A
    ENTERPRISE_ARCHITECTURE → Use case 8: System design context
    MARKETING_GROWTH        → Use case 9: Sales Research & Outreach
    ORGANIZATIONAL_BEHAVIOR → Use case 10: Workforce / de-escalation
    STRATEGY_INTELLIGENCE   → Use case 11: Competitive intelligence
    FINANCE_ECONOMICS       → Use case 12: Financial analysis
    CYBERSECURITY           → Use case 13: Security analysis
    SUPPLY_CHAIN            → Use case 14: Supply Chain Exception Management

    New medical domains
    -------------------
    CLINICAL_DECISION_SUPPORT  → Medical use case 1
    DRUG_INTERACTION           → Medical use case 2
    MEDICAL_LITERATURE         → Medical use case 3
    PATIENT_RISK               → Medical use case 4
    HEALTHCARE_ACCESS          → Medical use case 5
    GENOMICS_MEDICINE          → Medical use case 6
    MENTAL_HEALTH_TRIAGE       → Medical use case 7
    CLINICAL_TRIALS            → Medical use case 8
    """

    # ── New business domains ──────────────────────────────────────────────
    CUSTOMER_SUPPORT = "customer_support"
    INCIDENT_RESPONSE = "incident_response"
    FINANCE_RECONCILIATION = "finance_reconciliation"
    CODE_REVIEW = "code_review"
    ANALYTICS = "analytics"
    RFP_RESPONSE = "rfp_response"
    LEGAL_COMPLIANCE = "legal_compliance"
    ENTERPRISE_ARCHITECTURE = "enterprise_architecture"
    MARKETING_GROWTH = "marketing_growth"
    ORGANIZATIONAL_BEHAVIOR = "organizational_behavior"
    STRATEGY_INTELLIGENCE = "strategy_intelligence"
    FINANCE_ECONOMICS = "finance_economics"
    CYBERSECURITY = "cybersecurity"
    SUPPLY_CHAIN = "supply_chain"

    # ── New medical domains ───────────────────────────────────────────────
    CLINICAL_DECISION_SUPPORT = "clinical_decision_support"
    DRUG_INTERACTION = "drug_interaction"
    MEDICAL_LITERATURE = "medical_literature"
    PATIENT_RISK = "patient_risk"
    HEALTHCARE_ACCESS = "healthcare_access"
    GENOMICS_MEDICINE = "genomics_medicine"
    MENTAL_HEALTH_TRIAGE = "mental_health_triage"
    CLINICAL_TRIALS = "clinical_trials"

    # ── Legacy domains required by src/agents/base_agent.py ──────────────
    SOCIAL_SCIENCE = "social_science"           # legacy
    INTERVIEW_PREP = "interview_prep"           # legacy
    ALGORITHMS_PROGRAMMING = "algorithms_programming"  # legacy
    EE_LLM_RESEARCH = "ee_llm_research"         # legacy
    PHYSICS = "physics"                         # legacy
    NEURAL_NETWORKS = "neural_networks"         # legacy
    DEEP_LEARNING = "deep_learning"             # legacy
    SIGNAL_PROCESSING = "signal_processing"     # legacy
    EXPERIMENT_RUNNER = "experiment_runner"     # legacy
    HEALTHCARE = "healthcare"                   # legacy
    CLIMATE_ENERGY = "climate_energy"           # legacy
    BIOTECH_GENOMICS = "biotech_genomics"       # legacy


# ---------------------------------------------------------------------------
# Human-AI balance primitives
# ---------------------------------------------------------------------------


class AutonomyLevel(str, Enum):
    """Controls how much weight the AI recommendation carries vs. human judgment.

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
        unlimited-liability contract clauses, all clinical decisions).
    """

    FULL_AUTO = "full_auto"
    AI_PROPOSES = "ai_proposes"
    AI_ASSISTS = "ai_assists"
    HUMAN_FIRST = "human_first"


#: Base human-weight by autonomy level (before confidence adjustment).
#: At FULL_AUTO the human weight is 0.0 — AI acts alone.
#: At HUMAN_FIRST the human weight is 0.80 — human's judgment dominates.
AUTONOMY_BASE_WEIGHT: dict[AutonomyLevel, float] = {
    AutonomyLevel.FULL_AUTO: 0.0,
    AutonomyLevel.AI_PROPOSES: 0.20,
    AutonomyLevel.AI_ASSISTS: 0.50,
    AutonomyLevel.HUMAN_FIRST: 0.80,
}

#: Confidence below this threshold triggers escalation for standard domains.
ESCALATION_CONFIDENCE_THRESHOLD: float = 0.55

#: Confidence below this threshold triggers escalation for medical domains.
MEDICAL_ESCALATION_THRESHOLD: float = 0.70


@dataclass
class HumanJudgment:
    """A human reviewer's judgment injected into the AI recommendation pipeline.

    This dataclass carries the human's assessment of a specific decision
    context. The orchestrator blends it with the AI output according to the
    configured :class:`AutonomyLevel`.

    Usage example
    -------------
    .. code-block:: python

        from src.models import HumanJudgment

        judgment = HumanJudgment(
            context="Ticket urgency classification for #45821",
            judgment="This is P1 — customer is churning",
            confidence=0.95,
            override=True,
            notes="Customer explicitly mentioned cancellation",
        )
        assert judgment.override is True
    """

    context: str
    judgment: str
    confidence: float = 0.5
    override: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"HumanJudgment.confidence must be in [0, 1]; got {self.confidence}"
            )


@dataclass
class EscalationDecision:
    """Whether a result requires human review before action is taken.

    Created by the orchestrator when AI confidence is below threshold or the
    domain policy mandates human sign-off (e.g., all clinical decisions).

    Usage example
    -------------
    .. code-block:: python

        from src.models import EscalationDecision

        decision = EscalationDecision(
            needs_escalation=True,
            reason="Confidence below threshold (0.42 < 0.55)",
            urgency="immediate",
            checkpoint="On-call engineer must acknowledge within 15 min",
        )
        assert decision.needs_escalation is True
    """

    needs_escalation: bool
    reason: str
    urgency: str = "review"
    checkpoint: str = ""


# ---------------------------------------------------------------------------
# Core exchange types
# ---------------------------------------------------------------------------


@dataclass
class AgentResponse:
    """Atomic output produced by every BaseAgent subclass.

    The orchestrator collects a list of these and synthesises them into a
    use-case-specific result type. Both intuition-only and tool-grounded
    pipeline outputs are represented here.

    Usage example
    -------------
    .. code-block:: python

        from src.models import AgentResponse, Domain

        resp = AgentResponse(
            domain=Domain.CYBERSECURITY,
            answer="The log pattern suggests lateral movement.",
            reasoning="Matches MITRE ATT&CK T1021 (Remote Services).",
            confidence=0.82,
            sources=["https://attack.mitre.org/techniques/T1021/"],
        )
        assert resp.confidence == 0.82
    """

    domain: Domain
    answer: str
    reasoning: str
    confidence: float
    sources: list = field(default_factory=list)
    mcp_context: str = ""
    intuition_weight: float = 0.5
    tool_weight: float = 0.5

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"AgentResponse.confidence must be in [0, 1]; got {self.confidence}"
            )


@dataclass
class SearchResult:
    """A single result returned by any ToolBackend or MCP web-search call.

    Usage example
    -------------
    .. code-block:: python

        from src.models import SearchResult

        result = SearchResult(
            title="OWASP Top 10 2023",
            url="https://owasp.org/Top10/",
            snippet="The OWASP Top 10 is a standard awareness document...",
            relevance_score=0.91,
        )
        assert result.relevance_score == 0.91
    """

    title: str
    url: str
    snippet: str
    relevance_score: Optional[float] = None


# ---------------------------------------------------------------------------
# Business result types
# ---------------------------------------------------------------------------


@dataclass
class TriageResult:
    """Customer support triage output (Use case 1).

    Produced by the customer-support orchestrator after classifying a ticket,
    selecting a routing department, and drafting an initial response.

    Usage example
    -------------
    .. code-block:: python

        from src.models import TriageResult, EscalationDecision, AutonomyLevel

        result = TriageResult(
            ticket_text="My payment was charged twice.",
            urgency="P2",
            routing_department="billing",
            draft_response="We sincerely apologise and will investigate...",
            kb_articles=["KB-1042"],
        )
        assert result.urgency == "P2"
    """

    ticket_text: str
    urgency: str
    routing_department: str
    draft_response: str
    kb_articles: list = field(default_factory=list)
    escalation: EscalationDecision = field(
        default_factory=lambda: EscalationDecision(needs_escalation=False, reason="")
    )
    human_judgment: Optional[HumanJudgment] = None
    autonomy_used: AutonomyLevel = AutonomyLevel.AI_PROPOSES
    ai_confidence: float = 0.5
    reasoning: str = ""


@dataclass
class ComplianceAnswer:
    """Compliance policy Q&A output (Use case 2).

    Produced when a user asks a legal or regulatory policy question.
    Always includes cited document sections and escalation guidance.

    Usage example
    -------------
    .. code-block:: python

        from src.models import ComplianceAnswer

        answer = ComplianceAnswer(
            question="Does GDPR require consent for analytics cookies?",
            answer="Yes — explicit, informed, and freely given consent is required.",
            cited_sections=["GDPR Art. 6(1)(a)", "Recital 32"],
        )
        assert "GDPR Art. 6(1)(a)" in answer.cited_sections
    """

    question: str
    answer: str
    cited_sections: list = field(default_factory=list)
    escalation: EscalationDecision = field(
        default_factory=lambda: EscalationDecision(needs_escalation=False, reason="")
    )
    human_judgment: Optional[HumanJudgment] = None
    autonomy_used: AutonomyLevel = AutonomyLevel.AI_ASSISTS
    ai_confidence: float = 0.5
    reasoning: str = ""
    sources: list = field(default_factory=list)


@dataclass
class IncidentContext:
    """Input payload for the incident-response pipeline (Use case 3).

    Bundles the alert, logs, service topology, and historical incidents so the
    orchestrator can build root-cause hypotheses.

    Usage example
    -------------
    .. code-block:: python

        from src.models import IncidentContext

        ctx = IncidentContext(
            alert_payload='{"service":"payments","error_rate":0.42}',
            log_lines=["ERROR: DB connection timeout", "WARN: retry 3/3"],
            service_graph="payments -> db-primary",
        )
        assert ctx.service_graph != ""
    """

    alert_payload: str
    log_lines: list = field(default_factory=list)
    service_graph: str = ""
    past_incidents: list = field(default_factory=list)


@dataclass
class IncidentResponse:
    """Incident response orchestrator output (Use case 3).

    Contains root-cause hypotheses, mitigation steps, runbook links, and
    an escalation decision.

    Usage example
    -------------
    .. code-block:: python

        from src.models import IncidentResponse

        resp = IncidentResponse(
            alert_payload='{"service":"payments","error_rate":0.42}',
            root_cause_hypotheses=["DB connection pool exhausted"],
            mitigation_steps=["Restart connection pool", "Scale read replicas"],
            severity="P2",
        )
        assert resp.severity == "P2"
    """

    alert_payload: str
    root_cause_hypotheses: list = field(default_factory=list)
    mitigation_steps: list = field(default_factory=list)
    runbook_links: list = field(default_factory=list)
    severity: str = "P3"
    escalation: EscalationDecision = field(
        default_factory=lambda: EscalationDecision(needs_escalation=False, reason="")
    )
    human_judgment: Optional[HumanJudgment] = None
    autonomy_used: AutonomyLevel = AutonomyLevel.AI_PROPOSES
    ai_confidence: float = 0.5
    reasoning: str = ""


@dataclass
class ReconciliationMatch:
    """A single matched ledger/invoice pair from the reconciliation engine (Use case 4).

    Usage example
    -------------
    .. code-block:: python

        from src.models import ReconciliationMatch

        match = ReconciliationMatch(
            ledger_id="GL-20240301-0042",
            invoice_id="INV-88123",
            match_confidence=0.97,
            variance=0.01,
            note="Penny rounding on FX conversion",
        )
        assert match.match_confidence > 0.9
    """

    ledger_id: str
    invoice_id: str
    match_confidence: float
    variance: float = 0.0
    note: str = ""


@dataclass
class ReconciliationResult:
    """Finance reconciliation orchestrator output (Use case 4).

    Contains matched pairs, unmatched items, anomaly scores, and a suggested
    journal-entry narrative for audit purposes.

    Usage example
    -------------
    .. code-block:: python

        from src.models import ReconciliationResult, ReconciliationMatch

        result = ReconciliationResult(
            matched_pairs=[
                ReconciliationMatch("GL-001", "INV-001", 0.99)
            ],
            audit_narrative="All items reconciled within materiality threshold.",
        )
        assert len(result.matched_pairs) == 1
    """

    matched_pairs: list = field(default_factory=list)
    unmatched_ledger_ids: list = field(default_factory=list)
    unmatched_invoice_ids: list = field(default_factory=list)
    anomaly_scores: dict = field(default_factory=dict)
    audit_narrative: str = ""
    suggested_journals: list = field(default_factory=list)
    escalation: EscalationDecision = field(
        default_factory=lambda: EscalationDecision(needs_escalation=False, reason="")
    )
    human_judgment: Optional[HumanJudgment] = None
    autonomy_used: AutonomyLevel = AutonomyLevel.AI_PROPOSES
    ai_confidence: float = 0.5


@dataclass
class OutreachResult:
    """Sales research and outreach output (Use case 5).

    Contains a company summary, identified pain points, a personalised email
    draft, MEDDIC qualification score, and discovery questions.

    Usage example
    -------------
    .. code-block:: python

        from src.models import OutreachResult

        result = OutreachResult(
            company_name="Acme Corp",
            company_summary="Mid-market SaaS vendor focused on HR automation.",
            pain_points=["Manual payroll reconciliation", "High attrition"],
            email_draft="Hi Sarah, I noticed Acme recently...",
            meddic_score=0.62,
        )
        assert result.meddic_score > 0.5
    """

    company_name: str
    company_summary: str
    pain_points: list = field(default_factory=list)
    email_draft: str = ""
    meddic_score: float = 0.0
    discovery_questions: list = field(default_factory=list)
    competitive_notes: str = ""
    escalation: EscalationDecision = field(
        default_factory=lambda: EscalationDecision(needs_escalation=False, reason="")
    )
    human_judgment: Optional[HumanJudgment] = None
    autonomy_used: AutonomyLevel = AutonomyLevel.AI_ASSISTS
    ai_confidence: float = 0.5


@dataclass
class ReportContext:
    """Input payload for the analytics report generation pipeline (Use case 6).

    Usage example
    -------------
    .. code-block:: python

        from src.models import ReportContext

        ctx = ReportContext(
            metrics={"dau": 14200, "mrr": 82000, "churn_rate": 0.021},
            kpi_targets={"mrr": 90000, "churn_rate": 0.015},
            reporting_period="2024-Q1",
            audience="exec",
        )
        assert ctx.audience == "exec"
    """

    metrics: dict
    kpi_targets: dict = field(default_factory=dict)
    reporting_period: str = ""
    audience: str = "ops"


@dataclass
class ReportResult:
    """Analytics report generation output (Use case 6).

    Contains a headline summary, trend analysis, anomaly list, next actions,
    chart recommendations, and optional SQL queries for self-serve analytics.

    Usage example
    -------------
    .. code-block:: python

        from src.models import ReportResult

        result = ReportResult(
            headline="MRR grew 8% QoQ but churn exceeded target by 40%.",
            trend_analysis="Churn accelerated in the SMB segment.",
            anomalies=["SMB churn spike in week 11"],
            next_actions=["Investigate SMB churn root cause"],
        )
        assert "churn" in result.headline
    """

    headline: str
    trend_analysis: str
    anomalies: list = field(default_factory=list)
    next_actions: list = field(default_factory=list)
    chart_recommendations: list = field(default_factory=list)
    sql_queries: list = field(default_factory=list)
    escalation: EscalationDecision = field(
        default_factory=lambda: EscalationDecision(needs_escalation=False, reason="")
    )
    human_judgment: Optional[HumanJudgment] = None
    autonomy_used: AutonomyLevel = AutonomyLevel.FULL_AUTO
    ai_confidence: float = 0.5


@dataclass
class CodeReviewComment:
    """A single inline comment from the code-review agent (Use case 7).

    Usage example
    -------------
    .. code-block:: python

        from src.models import CodeReviewComment

        comment = CodeReviewComment(
            file_path="src/auth/tokens.py",
            line=42,
            severity="critical",
            category="security",
            message="JWT secret is hard-coded.",
            suggestion="Load secret from environment variable via os.getenv.",
        )
        assert comment.severity == "critical"
    """

    file_path: str
    line: int
    severity: str
    category: str
    message: str
    suggestion: str = ""


@dataclass
class CodeReviewResult:
    """Code review / PR assistant output (Use case 7).

    Contains inline comments, a risk score, security flags, and an approval
    recommendation for the pull request.

    Usage example
    -------------
    .. code-block:: python

        from src.models import CodeReviewResult, CodeReviewComment

        result = CodeReviewResult(
            diff_summary="Adds JWT auth middleware; 3 files changed.",
            comments=[
                CodeReviewComment("auth.py", 12, "high", "security",
                                  "Token not validated", "Add verify=True")
            ],
            risk_score=0.75,
            approval_recommendation="needs_review",
        )
        assert result.risk_score > 0.5
    """

    diff_summary: str
    comments: list = field(default_factory=list)
    risk_score: float = 0.0
    security_flags: list = field(default_factory=list)
    approval_recommendation: str = "needs_review"
    overall_reasoning: str = ""
    escalation: EscalationDecision = field(
        default_factory=lambda: EscalationDecision(needs_escalation=False, reason="")
    )
    human_judgment: Optional[HumanJudgment] = None
    autonomy_used: AutonomyLevel = AutonomyLevel.AI_PROPOSES
    ai_confidence: float = 0.5


@dataclass
class ExceptionEvent:
    """Input payload for the supply-chain exception management pipeline (Use case 8).

    Usage example
    -------------
    .. code-block:: python

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
        assert event.severity == "high"
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

    Contains the recommended action, financial impact estimate, alternative
    suppliers, and draft purchase-order lines.

    Usage example
    -------------
    .. code-block:: python

        from src.models import ExceptionResponse

        resp = ExceptionResponse(
            exception_type="late_delivery",
            sku="SKU-12345",
            recommended_action="expedite",
            financial_impact_estimate=8500.0,
            alternative_suppliers=["SupplierB", "SupplierC"],
        )
        assert resp.recommended_action == "expedite"
    """

    exception_type: str
    sku: str
    recommended_action: str
    financial_impact_estimate: float = 0.0
    alternative_suppliers: list = field(default_factory=list)
    draft_po_lines: list = field(default_factory=list)
    escalation: EscalationDecision = field(
        default_factory=lambda: EscalationDecision(needs_escalation=False, reason="")
    )
    human_judgment: Optional[HumanJudgment] = None
    autonomy_used: AutonomyLevel = AutonomyLevel.AI_PROPOSES
    ai_confidence: float = 0.5
    reasoning: str = ""


@dataclass
class RFPResult:
    """RFP response drafting output (Use case 9).

    Contains per-section drafts, a compliance matrix, win themes, risk flags,
    and an overall bid strategy narrative.

    Usage example
    -------------
    .. code-block:: python

        from src.models import RFPResult

        result = RFPResult(
            rfp_title="Enterprise Data Platform RFP — Acme Corp",
            section_drafts={"Executive Summary": "Our platform delivers..."},
            win_themes=["10-year track record", "SOC 2 Type II certified"],
            risk_flags=["Unlimited liability clause in §12.3"],
        )
        assert "risk_flags" in result.__dataclass_fields__
    """

    rfp_title: str
    section_drafts: dict = field(default_factory=dict)
    compliance_matrix: dict = field(default_factory=dict)
    win_themes: list = field(default_factory=list)
    risk_flags: list = field(default_factory=list)
    overall_strategy: str = ""
    escalation: EscalationDecision = field(
        default_factory=lambda: EscalationDecision(needs_escalation=False, reason="")
    )
    human_judgment: Optional[HumanJudgment] = None
    autonomy_used: AutonomyLevel = AutonomyLevel.AI_ASSISTS
    ai_confidence: float = 0.5


# ---------------------------------------------------------------------------
# Medical result types
# ---------------------------------------------------------------------------

_MEDICAL_DISCLAIMER: str = (
    "AI-generated analysis only. Always requires validation by a licensed "
    "healthcare professional before any clinical decision."
)


@dataclass
class ClinicalAssessmentInput:
    """Input payload for the clinical decision support pipeline (Medical use case 1).

    Usage example
    -------------
    .. code-block:: python

        from src.models import ClinicalAssessmentInput

        inp = ClinicalAssessmentInput(
            patient_summary="65 y/o male with T2DM and hypertension.",
            symptoms=["chest pain", "shortness of breath"],
            current_medications=["metformin 1000 mg", "lisinopril 10 mg"],
            lab_values={"HbA1c": 8.1, "BP": "142/88"},
            relevant_history="Prior MI in 2019",
        )
        assert "chest pain" in inp.symptoms
    """

    patient_summary: str
    symptoms: list = field(default_factory=list)
    current_medications: list = field(default_factory=list)
    lab_values: dict = field(default_factory=dict)
    relevant_history: str = ""


@dataclass
class ClinicalDecisionResult:
    """Clinical decision support output (Medical use case 1).

    .. warning::
        AI-generated analysis only. A licensed physician MUST validate all
        outputs before any clinical decision is made.

    Usage example
    -------------
    .. code-block:: python

        from src.models import ClinicalDecisionResult

        result = ClinicalDecisionResult(
            patient_summary="65 y/o male presenting with chest pain.",
            differential_diagnoses=["ACS", "GERD", "Costochondritis"],
            recommended_investigations=["ECG", "Troponin", "CXR"],
            treatment_options=["ASA 325 mg", "GTN 0.4 mg SL"],
            red_flags=["ST elevation on ECG"],
        )
        assert result.autonomy_used.value == "human_first"
    """

    MEDICAL_DISCLAIMER: ClassVar[str] = _MEDICAL_DISCLAIMER

    patient_summary: str
    differential_diagnoses: list = field(default_factory=list)
    recommended_investigations: list = field(default_factory=list)
    treatment_options: list = field(default_factory=list)
    red_flags: list = field(default_factory=list)
    guideline_references: list = field(default_factory=list)
    escalation: EscalationDecision = field(
        default_factory=lambda: EscalationDecision(
            needs_escalation=True,
            reason="Clinical decision requires physician review",
            urgency="immediate",
            checkpoint="Physician must validate before acting",
        )
    )
    human_judgment: Optional[HumanJudgment] = None
    autonomy_used: AutonomyLevel = AutonomyLevel.HUMAN_FIRST
    ai_confidence: float = 0.5
    disclaimer: str = field(default=_MEDICAL_DISCLAIMER)


@dataclass
class DrugInteractionResult:
    """Drug interaction analysis output (Medical use case 2).

    .. warning::
        AI-generated analysis only. A pharmacist or physician MUST validate
        all outputs before any prescribing or dispensing decision.

    Usage example
    -------------
    .. code-block:: python

        from src.models import DrugInteractionResult

        result = DrugInteractionResult(
            medications=["warfarin", "aspirin", "amiodarone"],
            interactions=["warfarin + aspirin: increased bleeding risk"],
            severity_summary="Two major interactions identified.",
        )
        assert len(result.interactions) > 0
    """

    MEDICAL_DISCLAIMER: ClassVar[str] = _MEDICAL_DISCLAIMER

    medications: list
    interactions: list = field(default_factory=list)
    contraindications: list = field(default_factory=list)
    dosing_notes: list = field(default_factory=list)
    severity_summary: str = ""
    escalation: EscalationDecision = field(
        default_factory=lambda: EscalationDecision(
            needs_escalation=True,
            reason="Drug interactions require pharmacist/physician review",
            urgency="immediate",
            checkpoint="Pharmacist must validate",
        )
    )
    human_judgment: Optional[HumanJudgment] = None
    autonomy_used: AutonomyLevel = AutonomyLevel.AI_PROPOSES
    ai_confidence: float = 0.5
    disclaimer: str = field(default=_MEDICAL_DISCLAIMER)


@dataclass
class LiteratureSynthesisResult:
    """Medical literature synthesis output (Medical use case 3).

    Synthesises evidence from research papers and clinical guidelines to
    answer a clinical or research question.

    Usage example
    -------------
    .. code-block:: python

        from src.models import LiteratureSynthesisResult

        result = LiteratureSynthesisResult(
            query="Efficacy of SGLT2 inhibitors in HFrEF",
            synthesis="SGLT2 inhibitors reduce HF hospitalisation by ~25%.",
            key_findings=["DAPA-HF trial: dapagliflozin NNT=21"],
            evidence_quality="high",
        )
        assert result.evidence_quality == "high"
    """

    query: str
    synthesis: str
    key_findings: list = field(default_factory=list)
    evidence_quality: str = ""
    conflicting_evidence: list = field(default_factory=list)
    recommended_reading: list = field(default_factory=list)
    escalation: EscalationDecision = field(
        default_factory=lambda: EscalationDecision(
            needs_escalation=False,
            reason="",
            urgency="informational",
            checkpoint="",
        )
    )
    human_judgment: Optional[HumanJudgment] = None
    autonomy_used: AutonomyLevel = AutonomyLevel.AI_ASSISTS
    ai_confidence: float = 0.5


@dataclass
class PatientRiskInput:
    """Input payload for the patient risk stratification pipeline (Medical use case 4).

    Usage example
    -------------
    .. code-block:: python

        from src.models import PatientRiskInput

        inp = PatientRiskInput(
            patient_id="PT-00421",
            age=72,
            diagnoses=["T2DM", "CKD stage 3", "hypertension"],
            medications=["metformin", "amlodipine"],
            recent_vitals={"systolic_bp": 148, "eGFR": 42},
        )
        assert inp.age == 72
    """

    patient_id: str
    age: int
    diagnoses: list = field(default_factory=list)
    medications: list = field(default_factory=list)
    recent_vitals: dict = field(default_factory=dict)
    social_determinants: dict = field(default_factory=dict)


@dataclass
class PatientRiskResult:
    """Patient risk stratification output (Medical use case 4).

    .. warning::
        AI-generated analysis only. A licensed clinician MUST review all
        outputs before initiating any care intervention.

    Usage example
    -------------
    .. code-block:: python

        from src.models import PatientRiskResult

        result = PatientRiskResult(
            patient_id="PT-00421",
            risk_level="high",
            risk_factors=["eGFR declining", "uncontrolled hypertension"],
            recommended_interventions=["Nephrology referral within 4 weeks"],
            follow_up_timeline="4 weeks",
        )
        assert result.risk_level == "high"
    """

    MEDICAL_DISCLAIMER: ClassVar[str] = _MEDICAL_DISCLAIMER

    patient_id: str
    risk_level: str = "moderate"
    risk_factors: list = field(default_factory=list)
    recommended_interventions: list = field(default_factory=list)
    follow_up_timeline: str = ""
    care_gaps: list = field(default_factory=list)
    escalation: EscalationDecision = field(
        default_factory=lambda: EscalationDecision(needs_escalation=False, reason="")
    )
    human_judgment: Optional[HumanJudgment] = None
    autonomy_used: AutonomyLevel = AutonomyLevel.AI_PROPOSES
    ai_confidence: float = 0.5
    disclaimer: str = field(default=_MEDICAL_DISCLAIMER)


@dataclass
class HealthcareGapResult:
    """Healthcare access gap analysis output (Medical use case 5).

    Identifies systemic gaps in healthcare access for a region or population
    and recommends policy interventions.

    Usage example
    -------------
    .. code-block:: python

        from src.models import HealthcareGapResult

        result = HealthcareGapResult(
            region_or_population="Rural Appalachia",
            identified_gaps=["No oncology within 90 miles", "1 primary care per 2,800"],
            affected_population_estimate="~340,000 residents",
        )
        assert result.identified_gaps != []
    """

    region_or_population: str
    identified_gaps: list = field(default_factory=list)
    affected_population_estimate: str = ""
    root_causes: list = field(default_factory=list)
    recommended_interventions: list = field(default_factory=list)
    policy_implications: list = field(default_factory=list)
    data_sources: list = field(default_factory=list)
    escalation: EscalationDecision = field(
        default_factory=lambda: EscalationDecision(needs_escalation=False, reason="")
    )
    human_judgment: Optional[HumanJudgment] = None
    autonomy_used: AutonomyLevel = AutonomyLevel.AI_ASSISTS
    ai_confidence: float = 0.5


@dataclass
class GenomicRiskResult:
    """Genomic risk analysis output (Medical use case 6).

    .. warning::
        AI-generated analysis only. A certified genetic counsellor MUST review
        all outputs before any clinical or reproductive decision is made.

    Usage example
    -------------
    .. code-block:: python

        from src.models import GenomicRiskResult

        result = GenomicRiskResult(
            sample_id="GS-20240301-0042",
            analyzed_variants=["BRCA1 c.5266dupC (pathogenic)"],
            disease_risk_scores={"breast_cancer_lifetime": 0.72},
            pharmacogenomic_notes=["CYP2D6 poor metaboliser — avoid codeine"],
            genetic_counseling_needed=True,
        )
        assert result.genetic_counseling_needed is True
    """

    MEDICAL_DISCLAIMER: ClassVar[str] = _MEDICAL_DISCLAIMER

    sample_id: str
    analyzed_variants: list = field(default_factory=list)
    disease_risk_scores: dict = field(default_factory=dict)
    pharmacogenomic_notes: list = field(default_factory=list)
    recommended_screenings: list = field(default_factory=list)
    genetic_counseling_needed: bool = True
    escalation: EscalationDecision = field(
        default_factory=lambda: EscalationDecision(
            needs_escalation=True,
            reason="Genomic results require genetic counselor review",
            urgency="review",
            checkpoint="Genetic counselor consult required",
        )
    )
    human_judgment: Optional[HumanJudgment] = None
    autonomy_used: AutonomyLevel = AutonomyLevel.HUMAN_FIRST
    ai_confidence: float = 0.5
    disclaimer: str = field(default=_MEDICAL_DISCLAIMER)


@dataclass
class MentalHealthTriageResult:
    """Mental health triage output (Medical use case 7).

    .. warning::
        AI-generated analysis only. A licensed mental health clinician MUST
        review all outputs before any intervention is initiated.

    Usage example
    -------------
    .. code-block:: python

        from src.models import MentalHealthTriageResult

        result = MentalHealthTriageResult(
            presenting_concerns="Persistent low mood, social withdrawal, sleep changes.",
            risk_level="moderate",
            recommended_resources=["PHQ-9 screening", "CBT referral"],
            follow_up_urgency="within_48h",
        )
        assert result.risk_level == "moderate"
    """

    MEDICAL_DISCLAIMER: ClassVar[str] = _MEDICAL_DISCLAIMER

    presenting_concerns: str
    risk_level: str = "low"
    recommended_resources: list = field(default_factory=list)
    crisis_indicators: list = field(default_factory=list)
    suggested_interventions: list = field(default_factory=list)
    follow_up_urgency: str = "routine"
    escalation: EscalationDecision = field(
        default_factory=lambda: EscalationDecision(
            needs_escalation=True,
            reason="Mental health triage requires clinician review",
            urgency="review",
            checkpoint="Clinician must review before any intervention",
        )
    )
    human_judgment: Optional[HumanJudgment] = None
    autonomy_used: AutonomyLevel = AutonomyLevel.HUMAN_FIRST
    ai_confidence: float = 0.5
    disclaimer: str = field(default=_MEDICAL_DISCLAIMER)


@dataclass
class ClinicalTrialMatch:
    """A single clinical trial matched to a patient profile (Medical use case 8).

    Usage example
    -------------
    .. code-block:: python

        from src.models import ClinicalTrialMatch

        match = ClinicalTrialMatch(
            trial_id="NCT05123456",
            title="Phase III SGLT2 inhibitor in CKD",
            eligibility_match="strong",
            key_criteria_met=["eGFR 20-45", "T2DM diagnosis"],
            phase="Phase III",
            sponsor="AstraZeneca",
            contact_url="https://clinicaltrials.gov/ct2/show/NCT05123456",
        )
        assert match.eligibility_match == "strong"
    """

    trial_id: str
    title: str
    eligibility_match: str
    key_criteria_met: list = field(default_factory=list)
    key_criteria_missed: list = field(default_factory=list)
    phase: str = ""
    sponsor: str = ""
    contact_url: str = ""


@dataclass
class ClinicalTrialsResult:
    """Clinical trials matching output (Medical use case 8).

    .. warning::
        AI-generated analysis only. A physician or principal investigator MUST
        confirm eligibility before any patient is enrolled in a trial.

    Usage example
    -------------
    .. code-block:: python

        from src.models import ClinicalTrialsResult, ClinicalTrialMatch

        result = ClinicalTrialsResult(
            patient_summary="72 y/o with CKD stage 3 and T2DM.",
            matched_trials=[
                ClinicalTrialMatch("NCT05123456", "SGLT2 in CKD", "strong")
            ],
            total_searched=120,
        )
        assert len(result.matched_trials) == 1
    """

    MEDICAL_DISCLAIMER: ClassVar[str] = _MEDICAL_DISCLAIMER

    patient_summary: str
    matched_trials: list = field(default_factory=list)
    total_searched: int = 0
    escalation: EscalationDecision = field(
        default_factory=lambda: EscalationDecision(
            needs_escalation=True,
            reason="Trial eligibility requires physician/PI confirmation",
            urgency="review",
            checkpoint="PI must confirm eligibility",
        )
    )
    human_judgment: Optional[HumanJudgment] = None
    autonomy_used: AutonomyLevel = AutonomyLevel.AI_PROPOSES
    ai_confidence: float = 0.5
    disclaimer: str = field(default=_MEDICAL_DISCLAIMER)
