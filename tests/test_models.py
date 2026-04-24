"""Tests for data models."""

import pytest
from src.models import (
    AgentResponse,
    AutonomyLevel,
    AUTONOMY_BASE_WEIGHT,
    ClinicalDecisionResult,
    ClinicalTrialsResult,
    Domain,
    DrugInteractionResult,
    EscalationDecision,
    ESCALATION_CONFIDENCE_THRESHOLD,
    GenomicRiskResult,
    HumanJudgment,
    MEDICAL_ESCALATION_THRESHOLD,
    MentalHealthTriageResult,
    PatientRiskResult,
    SearchResult,
    TriageResult,
    _MEDICAL_DISCLAIMER,
)


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------


class TestDomain:
    def test_business_domains_present(self):
        business = {
            "customer_support", "incident_response", "finance_reconciliation",
            "code_review", "analytics", "rfp_response", "legal_compliance",
            "enterprise_architecture", "marketing_growth", "organizational_behavior",
            "strategy_intelligence", "finance_economics", "cybersecurity", "supply_chain",
        }
        values = {d.value for d in Domain}
        assert business.issubset(values)

    def test_medical_domains_present(self):
        medical = {
            "clinical_decision_support", "drug_interaction", "medical_literature",
            "patient_risk", "healthcare_access", "genomics_medicine",
            "mental_health_triage", "clinical_trials",
        }
        values = {d.value for d in Domain}
        assert medical.issubset(values)

    def test_legacy_domains_present(self):
        """Legacy domains required by base_agent.py must not be removed."""
        legacy = {
            "social_science", "interview_prep", "algorithms_programming",
            "ee_llm_research", "physics", "neural_networks", "deep_learning",
            "signal_processing", "experiment_runner", "healthcare",
            "climate_energy", "biotech_genomics",
        }
        values = {d.value for d in Domain}
        assert legacy.issubset(values)

    def test_total_domain_count(self):
        # 14 business + 8 medical + 1 stock_market + 12 legacy = 35
        assert len(Domain) == 35


# ---------------------------------------------------------------------------
# AutonomyLevel and AUTONOMY_BASE_WEIGHT
# ---------------------------------------------------------------------------


class TestAutonomyLevel:
    def test_all_levels_in_weight_map(self):
        for level in AutonomyLevel:
            assert level in AUTONOMY_BASE_WEIGHT

    def test_weight_ordering(self):
        assert (
            AUTONOMY_BASE_WEIGHT[AutonomyLevel.FULL_AUTO]
            < AUTONOMY_BASE_WEIGHT[AutonomyLevel.AI_PROPOSES]
            < AUTONOMY_BASE_WEIGHT[AutonomyLevel.AI_ASSISTS]
            < AUTONOMY_BASE_WEIGHT[AutonomyLevel.HUMAN_FIRST]
        )

    def test_full_auto_zero_human_weight(self):
        assert AUTONOMY_BASE_WEIGHT[AutonomyLevel.FULL_AUTO] == 0.0


# ---------------------------------------------------------------------------
# EscalationDecision
# ---------------------------------------------------------------------------


class TestEscalationDecision:
    def test_defaults(self):
        d = EscalationDecision(needs_escalation=False, reason="")
        assert d.urgency == "review"
        assert d.checkpoint == ""

    def test_custom_values(self):
        d = EscalationDecision(
            needs_escalation=True,
            reason="High risk",
            urgency="immediate",
            checkpoint="On-call must ack",
        )
        assert d.needs_escalation is True
        assert d.urgency == "immediate"


# ---------------------------------------------------------------------------
# HumanJudgment
# ---------------------------------------------------------------------------


class TestHumanJudgment:
    def test_valid_creation(self):
        hj = HumanJudgment(
            context="Ticket urgency classification",
            judgment="This is P1",
            confidence=0.9,
        )
        assert hj.confidence == 0.9
        assert hj.override is False
        assert hj.notes == ""

    def test_confidence_bounds(self):
        with pytest.raises(ValueError):
            HumanJudgment(context="c", judgment="j", confidence=1.5)
        with pytest.raises(ValueError):
            HumanJudgment(context="c", judgment="j", confidence=-0.1)

    def test_boundary_confidence(self):
        hj0 = HumanJudgment(context="c", judgment="j", confidence=0.0)
        assert hj0.confidence == 0.0
        hj1 = HumanJudgment(context="c", judgment="j", confidence=1.0)
        assert hj1.confidence == 1.0


# ---------------------------------------------------------------------------
# AgentResponse
# ---------------------------------------------------------------------------


class TestAgentResponse:
    def test_valid_creation(self):
        resp = AgentResponse(
            domain=Domain.PHYSICS,
            answer="Gravity is curvature of spacetime.",
            reasoning="General Relativity.",
            confidence=0.95,
        )
        assert resp.domain == Domain.PHYSICS
        assert resp.sources == []
        assert resp.mcp_context == ""

    def test_confidence_bounds(self):
        with pytest.raises(ValueError):
            AgentResponse(
                domain=Domain.PHYSICS, answer="a", reasoning="r", confidence=1.1
            )
        with pytest.raises(ValueError):
            AgentResponse(
                domain=Domain.PHYSICS, answer="a", reasoning="r", confidence=-0.01
            )

    def test_boundary_confidence(self):
        r = AgentResponse(domain=Domain.CYBERSECURITY, answer="a", reasoning="r", confidence=0.0)
        assert r.confidence == 0.0


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_creation(self):
        sr = SearchResult(title="Test", url="https://example.com", snippet="A snippet")
        assert sr.relevance_score is None

    def test_with_relevance(self):
        sr = SearchResult(title="T", url="u", snippet="s", relevance_score=0.88)
        assert sr.relevance_score == 0.88


# ---------------------------------------------------------------------------
# TriageResult
# ---------------------------------------------------------------------------


class TestTriageResult:
    def test_defaults(self):
        t = TriageResult(
            ticket_text="Order missing",
            urgency="P3",
            routing_department="logistics",
            draft_response="We are investigating.",
        )
        assert t.kb_articles == []
        assert t.escalation.needs_escalation is False
        assert t.autonomy_used == AutonomyLevel.AI_PROPOSES
        assert t.ai_confidence == 0.5

    def test_mutable_defaults_independent(self):
        t1 = TriageResult("a", "P3", "dept", "draft")
        t2 = TriageResult("b", "P2", "dept", "draft")
        t1.kb_articles.append("KB-1")
        assert t2.kb_articles == []


# ---------------------------------------------------------------------------
# Medical result types — disclaimer and escalation defaults
# ---------------------------------------------------------------------------


class TestMedicalDisclaimer:
    def test_disclaimer_constant_not_empty(self):
        assert len(_MEDICAL_DISCLAIMER) > 0

    def test_threshold_values(self):
        assert ESCALATION_CONFIDENCE_THRESHOLD < MEDICAL_ESCALATION_THRESHOLD

    def test_clinical_decision_result_has_disclaimer(self):
        r = ClinicalDecisionResult(patient_summary="Patient X")
        assert r.disclaimer == _MEDICAL_DISCLAIMER
        assert ClinicalDecisionResult.MEDICAL_DISCLAIMER == _MEDICAL_DISCLAIMER

    def test_clinical_decision_defaults_to_human_first(self):
        r = ClinicalDecisionResult(patient_summary="Patient X")
        assert r.autonomy_used == AutonomyLevel.HUMAN_FIRST
        assert r.escalation.needs_escalation is True
        assert r.escalation.urgency == "immediate"

    def test_drug_interaction_escalated_by_default(self):
        r = DrugInteractionResult(medications=["warfarin", "aspirin"])
        assert r.escalation.needs_escalation is True
        assert r.disclaimer == _MEDICAL_DISCLAIMER

    def test_genomic_risk_requires_counselor(self):
        r = GenomicRiskResult(sample_id="GS-001")
        assert r.genetic_counseling_needed is True
        assert r.autonomy_used == AutonomyLevel.HUMAN_FIRST
        assert r.escalation.needs_escalation is True

    def test_mental_health_triage_human_first(self):
        r = MentalHealthTriageResult(presenting_concerns="Low mood")
        assert r.autonomy_used == AutonomyLevel.HUMAN_FIRST
        assert r.escalation.needs_escalation is True
        assert r.disclaimer == _MEDICAL_DISCLAIMER

    def test_patient_risk_result_has_disclaimer(self):
        r = PatientRiskResult(patient_id="PT-001")
        assert r.disclaimer == _MEDICAL_DISCLAIMER

    def test_clinical_trials_result_has_disclaimer(self):
        r = ClinicalTrialsResult(patient_summary="72 y/o with CKD")
        assert r.disclaimer == _MEDICAL_DISCLAIMER
        assert r.escalation.needs_escalation is True
