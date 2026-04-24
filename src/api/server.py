"""FastAPI REST server exposing all 18 use-case endpoints."""
from __future__ import annotations
import dataclasses
from typing import Optional, Any

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False


def _require_fastapi():
    if not _FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")


if _FASTAPI_AVAILABLE:
    class TriageRequest(BaseModel):
        ticket_text: str
        autonomy: str = "ai_proposes"

    class ComplianceRequest(BaseModel):
        question: str
        autonomy: str = "ai_assists"

    class IncidentRequest(BaseModel):
        alert_payload: str
        log_lines: list[str] = []
        service_graph: str = ""
        autonomy: str = "ai_proposes"

    class ReconcileRequest(BaseModel):
        ledger: list[dict]
        invoices: list[dict]
        materiality_threshold: float = 1000.0

    class OutreachRequest(BaseModel):
        company: str
        product: str = ""
        deal_value: float = 0.0

    class ReportRequest(BaseModel):
        metrics: dict
        kpi_targets: dict = {}
        reporting_period: str = ""
        audience: str = "ops"

    class ReviewPRRequest(BaseModel):
        diff: str
        description: str = ""

    class ExceptionRequest(BaseModel):
        exception_type: str
        sku: str
        supplier: str
        severity: str = "medium"
        eta_days_late: int = 0
        current_inventory: float = 0.0
        cost_to_expedite: float = 0.0

    class RFPRequest(BaseModel):
        rfp_text: str
        rfp_title: str = ""

    class ClinicalRequest(BaseModel):
        patient_summary: str
        symptoms: list[str] = []
        current_medications: list[str] = []
        lab_values: dict = {}
        relevant_history: str = ""

    class DrugInteractionRequest(BaseModel):
        medications: list[str]
        patient_context: str = ""

    class LiteratureRequest(BaseModel):
        query: str

    class PatientRiskRequest(BaseModel):
        patient_id: str
        age: int
        diagnoses: list[str] = []
        medications: list[str] = []

    class HealthcareGapsRequest(BaseModel):
        region_or_population: str

    class GenomicRequest(BaseModel):
        sample_id: str
        variants: list[str]
        patient_context: str = ""

    class MentalHealthRequest(BaseModel):
        presenting_concerns: str

    class ClinicalTrialsRequest(BaseModel):
        patient_id: str
        age: int
        diagnoses: list[str] = []
        condition: str = ""

    class StockRequest(BaseModel):
        ticker: str
        horizon: str = "1m"
        news_headlines: list[str] = []
        macro_context: str = ""
        human_thesis: str = ""

    def _to_dict(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj):
            return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, list):
            return [_to_dict(i) for i in obj]
        if isinstance(obj, dict):
            return {k: _to_dict(v) for k, v in obj.items()}
        if hasattr(obj, 'value'):
            return obj.value
        return obj

    def create_app(backend_spec: str = "", model_profile: str = "balanced") -> "FastAPI":
        from src.orchestrator.business_orchestrator import BusinessOrchestrator
        from src.models import (
            IncidentContext, ClinicalAssessmentInput, ExceptionEvent,
            PatientRiskInput, StockPredictionInput, ReportContext,
        )

        app = FastAPI(title="Business Agent Platform", version="1.0.0")
        orch = BusinessOrchestrator(backend_spec=backend_spec, model_profile=model_profile)

        @app.get("/health")
        def health():
            return {"status": "ok", "model_profile": model_profile}

        @app.get("/models")
        def models():
            from src.llm.model_registry import list_free_models
            return {"models": [_to_dict(m) for m in list_free_models()]}

        @app.post("/triage")
        def triage(req: TriageRequest):
            return _to_dict(orch.triage(req.ticket_text))

        @app.post("/compliance_qa")
        def compliance_qa(req: ComplianceRequest):
            return _to_dict(orch.compliance_qa(req.question))

        @app.post("/incident")
        def incident(req: IncidentRequest):
            ctx = IncidentContext(alert_payload=req.alert_payload, log_lines=req.log_lines, service_graph=req.service_graph)
            return _to_dict(orch.respond_to_incident(ctx))

        @app.post("/reconcile")
        def reconcile(req: ReconcileRequest):
            return _to_dict(orch.reconcile(req.ledger, req.invoices, materiality_threshold=req.materiality_threshold))

        @app.post("/outreach")
        def outreach(req: OutreachRequest):
            return _to_dict(orch.outreach(req.company, req.product, req.deal_value))

        @app.post("/report")
        def report(req: ReportRequest):
            ctx = ReportContext(metrics=req.metrics, kpi_targets=req.kpi_targets, reporting_period=req.reporting_period, audience=req.audience)
            return _to_dict(orch.generate_report(ctx))

        @app.post("/review_pr")
        def review_pr(req: ReviewPRRequest):
            return _to_dict(orch.review_pr(req.diff, req.description))

        @app.post("/exception")
        def exception(req: ExceptionRequest):
            evt = ExceptionEvent(
                exception_type=req.exception_type, sku=req.sku, supplier=req.supplier,
                severity=req.severity, eta_days_late=req.eta_days_late,
                current_inventory=req.current_inventory, cost_to_expedite=req.cost_to_expedite,
            )
            return _to_dict(orch.handle_exception(evt))

        @app.post("/rfp")
        def rfp(req: RFPRequest):
            return _to_dict(orch.draft_rfp(req.rfp_text, req.rfp_title))

        @app.post("/clinical_decision")
        def clinical_decision(req: ClinicalRequest):
            assessment = ClinicalAssessmentInput(
                patient_summary=req.patient_summary, symptoms=req.symptoms,
                current_medications=req.current_medications, lab_values=req.lab_values,
                relevant_history=req.relevant_history,
            )
            return _to_dict(orch.clinical_decision(assessment))

        @app.post("/drug_interactions")
        def drug_interactions(req: DrugInteractionRequest):
            return _to_dict(orch.check_drug_interactions(req.medications, req.patient_context))

        @app.post("/literature")
        def literature(req: LiteratureRequest):
            return _to_dict(orch.synthesize_literature(req.query))

        @app.post("/patient_risk")
        def patient_risk(req: PatientRiskRequest):
            p = PatientRiskInput(patient_id=req.patient_id, age=req.age, diagnoses=req.diagnoses, medications=req.medications)
            return _to_dict(orch.stratify_patient_risk(p))

        @app.post("/healthcare_gaps")
        def healthcare_gaps(req: HealthcareGapsRequest):
            return _to_dict(orch.analyze_healthcare_gaps(req.region_or_population))

        @app.post("/genomic_risk")
        def genomic_risk(req: GenomicRequest):
            return _to_dict(orch.assess_genomic_risk(req.sample_id, req.variants, req.patient_context))

        @app.post("/mental_health")
        def mental_health(req: MentalHealthRequest):
            return _to_dict(orch.triage_mental_health(req.presenting_concerns))

        @app.post("/clinical_trials")
        def clinical_trials(req: ClinicalTrialsRequest):
            p = PatientRiskInput(patient_id=req.patient_id, age=req.age, diagnoses=req.diagnoses)
            return _to_dict(orch.match_clinical_trials(p, req.condition))

        @app.post("/stock")
        def stock(req: StockRequest):
            inp = StockPredictionInput(
                ticker=req.ticker, horizon=req.horizon, news_headlines=req.news_headlines,
                macro_context=req.macro_context, human_thesis=req.human_thesis,
            )
            return _to_dict(orch.predict_stock(inp))

        return app

    app = create_app()
