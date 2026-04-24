"""Tests for BusinessOrchestrator — all 18 use cases."""
from __future__ import annotations
import pytest
from src.llm.mock_backend import MockBackend
from src.orchestrator.business_orchestrator import BusinessOrchestrator
from src.models import (
    AutonomyLevel, HumanJudgment, IncidentContext, PatientRiskInput,
    StockPredictionInput, ReportContext, ExceptionEvent, ClinicalAssessmentInput,
    TriageResult, ComplianceAnswer, IncidentResponse, ReconciliationResult,
    OutreachResult, ReportResult, CodeReviewResult, ExceptionResponse, RFPResult,
    ClinicalDecisionResult, DrugInteractionResult, LiteratureSynthesisResult,
    PatientRiskResult, HealthcareGapResult, GenomicRiskResult,
    MentalHealthTriageResult, ClinicalTrialsResult, StockPredictionResult,
)


@pytest.fixture
def orch():
    return BusinessOrchestrator(backend=MockBackend())


class TestBusinessUseCases:
    def test_triage_returns_correct_type(self, orch):
        r = orch.triage("Payment API returning 500 for all enterprise users")
        assert isinstance(r, TriageResult)
        assert r.ticket_text
        assert r.urgency in ("P1","P2","P3","P4")

    def test_triage_p1_detection(self, orch):
        r = orch.triage("Critical outage: all users cannot login")
        assert r.urgency == "P1"

    def test_compliance_qa(self, orch):
        r = orch.compliance_qa("Must we delete user data on account closure under GDPR?")
        assert isinstance(r, ComplianceAnswer)
        assert r.question

    def test_incident_response(self, orch):
        ctx = IncidentContext(alert_payload="[P1] api-gateway 503 errors 45%", log_lines=["WARN: connection refused"])
        r = orch.respond_to_incident(ctx)
        assert isinstance(r, IncidentResponse)
        assert r.severity in ("P1","P2","P3","P4")

    def test_reconcile(self, orch):
        ledger = [{"id":"L1","amount":1000}, {"id":"L2","amount":500}]
        invoices = [{"id":"I1","amount":1000}, {"id":"I2","amount":502}]
        r = orch.reconcile(ledger, invoices)
        assert isinstance(r, ReconciliationResult)
        assert len(r.matched_pairs) == 2

    def test_outreach(self, orch):
        r = orch.outreach("Acme Corp", "DataPlatform", 100_000)
        assert isinstance(r, OutreachResult)
        assert r.company_name == "Acme Corp"

    def test_outreach_large_deal_escalates(self, orch):
        r = orch.outreach("BigCo", "Platform", 600_000)
        assert r.escalation.needs_escalation is True

    def test_report(self, orch):
        ctx = ReportContext(metrics={"revenue": 4_200_000, "churn": 1.8}, kpi_targets={"revenue": 4_000_000})
        r = orch.generate_report(ctx)
        assert isinstance(r, ReportResult)
        assert r.headline

    def test_review_pr(self, orch):
        r = orch.review_pr("def login(u, p): pass", "Add login endpoint")
        assert isinstance(r, CodeReviewResult)
        assert 0.0 <= r.risk_score <= 1.0

    def test_review_pr_sql_injection_high_risk(self, orch):
        r = orch.review_pr("db.raw(f'SELECT * FROM users WHERE id={user_id}')", "")
        assert r.risk_score >= 0.6

    def test_handle_exception(self, orch):
        evt = ExceptionEvent(exception_type="late_delivery", sku="SKU-1", supplier="SupA", eta_days_late=10, cost_to_expedite=5000)
        r = orch.handle_exception(evt)
        assert isinstance(r, ExceptionResponse)
        assert r.recommended_action in ("expedite","substitute","backorder","cancel","escalate")

    def test_draft_rfp(self, orch):
        r = orch.draft_rfp("We require 99.99% uptime. Vendor must accept unlimited liability.", "Cloud Platform RFP")
        assert isinstance(r, RFPResult)
        assert "unlimited liability" in r.risk_flags


class TestMedicalUseCases:
    def test_clinical_decision_always_escalates(self, orch):
        a = ClinicalAssessmentInput(patient_summary="45F chest pain", symptoms=["chest pain","diaphoresis"])
        r = orch.clinical_decision(a)
        assert isinstance(r, ClinicalDecisionResult)
        assert r.escalation.needs_escalation is True

    def test_drug_interactions_always_escalates(self, orch):
        r = orch.check_drug_interactions(["warfarin","aspirin","ibuprofen"])
        assert isinstance(r, DrugInteractionResult)
        assert r.escalation.needs_escalation is True

    def test_literature_synthesis(self, orch):
        r = orch.synthesize_literature("GLP-1 agonists cardiovascular outcomes meta-analysis")
        assert isinstance(r, LiteratureSynthesisResult)
        assert r.query

    def test_patient_risk(self, orch):
        p = PatientRiskInput(patient_id="P001", age=75, diagnoses=["diabetes","hypertension","CKD"])
        r = orch.stratify_patient_risk(p)
        assert isinstance(r, PatientRiskResult)
        assert r.risk_level in ("low","moderate","high")

    def test_patient_risk_high_age_multiple_diags(self, orch):
        p = PatientRiskInput(patient_id="P002", age=80, diagnoses=["diabetes","heart failure","CKD","COPD"])
        r = orch.stratify_patient_risk(p)
        assert r.risk_level == "high"

    def test_healthcare_gaps(self, orch):
        r = orch.analyze_healthcare_gaps("rural Appalachia")
        assert isinstance(r, HealthcareGapResult)

    def test_genomic_risk_always_escalates(self, orch):
        r = orch.assess_genomic_risk("S001", ["BRCA1:c.5266dupC"])
        assert isinstance(r, GenomicRiskResult)
        assert r.escalation.needs_escalation is True
        assert r.genetic_counseling_needed is True

    def test_mental_health_always_escalates(self, orch):
        r = orch.triage_mental_health("I have been feeling hopeless and having panic attacks")
        assert isinstance(r, MentalHealthTriageResult)
        assert r.escalation.needs_escalation is True

    def test_mental_health_crisis_immediate(self, orch):
        r = orch.triage_mental_health("I want to hurt myself and feel suicidal")
        assert r.escalation.urgency == "immediate"
        assert r.risk_level == "high"

    def test_clinical_trials(self, orch):
        p = PatientRiskInput(patient_id="P003", age=45, diagnoses=["breast cancer"])
        r = orch.match_clinical_trials(p, "HER2+ breast cancer")
        assert isinstance(r, ClinicalTrialsResult)
        assert r.escalation.needs_escalation is True


class TestStockMarket:
    def test_stock_prediction(self, orch):
        inp = StockPredictionInput(ticker="NVDA", horizon="1m")
        r = orch.predict_stock(inp)
        assert isinstance(r, StockPredictionResult)
        assert r.ticker == "NVDA"
        assert r.direction in ("bullish","bearish","neutral")
        assert 0 <= r.confidence_pct <= 100

    def test_stock_with_human_thesis(self, orch):
        inp = StockPredictionInput(ticker="AAPL", horizon="1w", human_thesis="Bullish — services revenue growing")
        j = HumanJudgment(context="AAPL outlook", judgment="Strong buy — cash position healthy", confidence=0.8)
        r = orch.predict_stock(inp, human_judgment=j, autonomy=AutonomyLevel.AI_ASSISTS)
        assert isinstance(r, StockPredictionResult)
        assert r.human_judgment is not None

    def test_stock_disclaimer(self, orch):
        inp = StockPredictionInput(ticker="SPY")
        r = orch.predict_stock(inp)
        assert "not investment advice" in r.disclaimer.lower()


class TestContextManager:
    def test_context_manager(self):
        with BusinessOrchestrator(backend=MockBackend()) as orch:
            r = orch.triage("Test ticket")
            assert isinstance(r, TriageResult)


class TestHumanAIBalance:
    def test_human_override(self, orch):
        j = HumanJudgment(context="test", judgment="This is P1 — I override AI", confidence=1.0, override=True)
        r = orch.triage("some ticket", human_judgment=j)
        assert "I override AI" in r.draft_response

    def test_autonomy_full_auto_no_human_weight(self, orch):
        from src.models import AUTONOMY_BASE_WEIGHT
        assert AUTONOMY_BASE_WEIGHT[AutonomyLevel.FULL_AUTO] == 0.0

    def test_autonomy_human_first_high_weight(self, orch):
        from src.models import AUTONOMY_BASE_WEIGHT
        assert AUTONOMY_BASE_WEIGHT[AutonomyLevel.HUMAN_FIRST] == 0.80
