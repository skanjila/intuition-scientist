#!/usr/bin/env python3
"""Business, Medical & Stock Market Agent Platform — CLI"""
from __future__ import annotations
import argparse
import json
import sys


def _make_orch(args):
    from src.orchestrator.business_orchestrator import BusinessOrchestrator
    return BusinessOrchestrator(
        backend_spec=getattr(args, "model", "") or "",
        model_profile=getattr(args, "profile", "balanced") or "balanced",
        verbose=getattr(args, "verbose", False),
    )


def _autonomy(args):
    from src.models import AutonomyLevel
    val = getattr(args, "autonomy", None)
    if val:
        return AutonomyLevel(val)
    return None


def cmd_demo(args):
    """Run mock demos of all 18 use cases."""
    from src.llm.mock_backend import MockBackend
    from src.orchestrator.business_orchestrator import BusinessOrchestrator
    from src.models import (
        IncidentContext, ReportContext, ExceptionEvent, ClinicalAssessmentInput,
        PatientRiskInput, StockPredictionInput,
    )
    orch = BusinessOrchestrator(backend=MockBackend())
    demos = [
        ("triage", lambda: orch.triage("Payment API returning 500 for all enterprise users")),
        ("compliance_qa", lambda: orch.compliance_qa("Do we need to delete user data under GDPR on account closure?")),
        ("incident", lambda: orch.respond_to_incident(IncidentContext(alert_payload="[P1] api-gateway 503 45%", log_lines=["ERROR: timeout"]))),
        ("reconcile", lambda: orch.reconcile([{"id":"L1","amount":1000}], [{"id":"I1","amount":1000}])),
        ("outreach", lambda: orch.outreach("Acme Corp", "DataPlatform", 100_000)),
        ("report", lambda: orch.generate_report(ReportContext(metrics={"revenue": 4_200_000}, kpi_targets={"revenue": 4_000_000}))),
        ("review_pr", lambda: orch.review_pr("def login(u,p): return db.query(u,p)", "Add login")),
        ("exception", lambda: orch.handle_exception(ExceptionEvent("late_delivery","SKU-1","SupplierA",eta_days_late=10,cost_to_expedite=5000))),
        ("draft_rfp", lambda: orch.draft_rfp("Vendor must provide 99.99% uptime.", "Cloud Platform RFP")),
        ("clinical_decision", lambda: orch.clinical_decision(ClinicalAssessmentInput("45F chest pain", symptoms=["chest pain"]))),
        ("drug_interactions", lambda: orch.check_drug_interactions(["warfarin","aspirin","ibuprofen"])),
        ("literature", lambda: orch.synthesize_literature("GLP-1 agonists cardiovascular outcomes")),
        ("patient_risk", lambda: orch.stratify_patient_risk(PatientRiskInput("P001", 75, ["diabetes","hypertension","CKD"]))),
        ("healthcare_gaps", lambda: orch.analyze_healthcare_gaps("rural Appalachia")),
        ("genomic_risk", lambda: orch.assess_genomic_risk("S001", ["BRCA1:c.5266dupC"])),
        ("mental_health", lambda: orch.triage_mental_health("panic attacks daily, not sleeping")),
        ("clinical_trials", lambda: orch.match_clinical_trials(PatientRiskInput("P002", 45, ["breast cancer"]), "HER2+ breast cancer")),
        ("stock", lambda: orch.predict_stock(StockPredictionInput("NVDA", "1m"))),
    ]
    for name, fn in demos:
        try:
            result = fn()
            print(f"✓ {name}: {type(result).__name__}")
        except Exception as e:
            print(f"✗ {name}: {e}")


def cmd_models(args):
    from src.llm.model_registry import list_free_models
    models = list_free_models()
    print(f"{'Spec':<55} {'Profile':<12} {'Tier':<12} {'Context':<10}")
    print("-" * 90)
    for m in models:
        spec = f"{m.provider}:{m.model_id}"
        print(f"{spec:<55} {m.profile.value:<12} {m.cost_tier.value:<12} {m.context_window_k}k")


def cmd_serve(args):
    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn not installed. Run: pip install fastapi uvicorn", file=sys.stderr)
        sys.exit(1)
    host = getattr(args, "host", "0.0.0.0")
    port = getattr(args, "port", 8080)
    uvicorn.run("src.api.server:app", host=host, port=port, reload=False)


def cmd_triage(args):
    orch = _make_orch(args)
    r = orch.triage(args.ticket, autonomy=_autonomy(args) or __import__("src.models", fromlist=["AutonomyLevel"]).AutonomyLevel.AI_PROPOSES)
    print(f"Urgency: {r.urgency} | Department: {r.routing_department}")
    print(f"Escalate: {r.escalation.needs_escalation} ({r.escalation.reason})")
    print(f"\nDraft Response:\n{r.draft_response}")


def cmd_incident(args):
    from src.models import IncidentContext
    orch = _make_orch(args)
    ctx = IncidentContext(alert_payload=args.alert)
    r = orch.respond_to_incident(ctx)
    print(f"Severity: {r.severity}")
    print(f"Root causes: {r.root_cause_hypotheses}")
    print(f"Steps: {r.mitigation_steps}")


def cmd_compliance(args):
    orch = _make_orch(args)
    r = orch.compliance_qa(args.question)
    print(f"Answer: {r.answer}")
    print(f"Escalate: {r.escalation.needs_escalation}")


def cmd_reconcile(args):
    orch = _make_orch(args)
    ledger = [{"id":"L1","amount":1000},{"id":"L2","amount":500}]
    invoices = [{"id":"I1","amount":1000},{"id":"I2","amount":502}]
    r = orch.reconcile(ledger, invoices)
    print(f"Matched: {len(r.matched_pairs)} pairs")
    print(f"Audit: {r.audit_narrative[:200]}")


def cmd_report(args):
    from src.models import ReportContext
    metrics = json.loads(args.metrics) if args.metrics else {"revenue": 1_000_000}
    orch = _make_orch(args)
    ctx = ReportContext(metrics=metrics)
    r = orch.generate_report(ctx)
    print(f"Headline: {r.headline}")
    if r.anomalies:
        print(f"Anomalies: {r.anomalies}")


def cmd_review(args):
    orch = _make_orch(args)
    r = orch.review_pr(args.diff)
    print(f"Risk: {r.risk_score:.2f} | Recommendation: {r.approval_recommendation}")
    print(f"Reasoning: {r.overall_reasoning[:300]}")


def cmd_exception(args):
    from src.models import ExceptionEvent
    orch = _make_orch(args)
    evt = ExceptionEvent(exception_type="late_delivery", sku=args.sku, supplier=args.supplier, eta_days_late=args.days_late)
    r = orch.handle_exception(evt)
    print(f"Action: {r.recommended_action}")
    print(f"Reasoning: {r.reasoning[:200]}")


def cmd_rfp(args):
    orch = _make_orch(args)
    r = orch.draft_rfp(args.text, args.title)
    print(f"Risk flags: {r.risk_flags}")
    print(f"Strategy: {r.overall_strategy[:200]}")


def cmd_outreach(args):
    orch = _make_orch(args)
    r = orch.outreach(args.company, getattr(args, "product", ""))
    print(f"Company: {r.company_name}")
    print(f"Email draft: {r.email_draft[:300]}")


def cmd_clinical(args):
    from src.models import ClinicalAssessmentInput
    orch = _make_orch(args)
    symptoms = [s.strip() for s in args.symptoms.split(",")] if args.symptoms else []
    a = ClinicalAssessmentInput(patient_summary=args.symptoms or "Patient assessment", symptoms=symptoms)
    r = orch.clinical_decision(a)
    print(f"Escalate: {r.escalation.needs_escalation} ({r.escalation.urgency})")
    print(f"Differential: {r.differential_diagnoses}")
    print(f"Red flags: {r.red_flags}")


def cmd_drugs(args):
    orch = _make_orch(args)
    meds = [m.strip() for m in args.medications.split(",")]
    r = orch.check_drug_interactions(meds)
    print(f"Severity summary: {r.severity_summary[:300]}")


def cmd_literature(args):
    orch = _make_orch(args)
    r = orch.synthesize_literature(args.query)
    print(f"Synthesis: {r.synthesis[:400]}")


def cmd_risk(args):
    from src.models import PatientRiskInput
    orch = _make_orch(args)
    diags = [d.strip() for d in args.diagnoses.split(",")] if args.diagnoses else []
    p = PatientRiskInput(patient_id=args.patient_id, age=args.age, diagnoses=diags)
    r = orch.stratify_patient_risk(p)
    print(f"Risk level: {r.risk_level}")
    print(f"Interventions: {r.recommended_interventions}")


def cmd_gaps(args):
    orch = _make_orch(args)
    r = orch.analyze_healthcare_gaps(args.population)
    print(f"Gaps: {r.identified_gaps}")


def cmd_genomics(args):
    orch = _make_orch(args)
    variants = [v.strip() for v in args.variants.split(",")]
    r = orch.assess_genomic_risk(args.sample_id, variants)
    print(f"Counseling needed: {r.genetic_counseling_needed}")
    print(f"Screenings: {r.recommended_screenings}")


def cmd_mental(args):
    orch = _make_orch(args)
    r = orch.triage_mental_health(args.concerns)
    print(f"Risk level: {r.risk_level}")
    print(f"Crisis indicators: {r.crisis_indicators}")
    print(f"Resources: {r.recommended_resources}")


def cmd_trials(args):
    from src.models import PatientRiskInput
    orch = _make_orch(args)
    diags = [d.strip() for d in args.diagnoses.split(",")] if hasattr(args, "diagnoses") and args.diagnoses else []
    p = PatientRiskInput(patient_id=args.patient_id, age=args.age, diagnoses=diags)
    r = orch.match_clinical_trials(p, getattr(args, "condition", ""))
    print(f"Matched trials: {len(r.matched_trials)}")
    for t in r.matched_trials:
        print(f"  - {t.trial_id}: {t.title[:60]} ({t.eligibility_match})")


def cmd_stock(args):
    from src.models import StockPredictionInput
    orch = _make_orch(args)
    headlines = [args.headline] if getattr(args, "headline", None) else []
    inp = StockPredictionInput(ticker=args.ticker, horizon=getattr(args, "horizon", "1m"), news_headlines=headlines)
    r = orch.predict_stock(inp)
    print(f"{r.ticker} ({r.direction.upper()}) — confidence: {r.confidence_pct:.1f}%")
    print(f"Thesis: {r.thesis[:300]}")
    print(f"Position sizing: {r.suggested_position_sizing}")
    print(f"\n⚠ {r.disclaimer}")


def main():
    parser = argparse.ArgumentParser(description="Business Agent Platform CLI")
    parser.add_argument("--model", default="", help="Backend spec e.g. ollama:llama3.1:8b")
    parser.add_argument("--profile", default="balanced", choices=["fast","balanced","quality"])
    parser.add_argument("--autonomy", default=None, choices=["full_auto","ai_proposes","ai_assists","human_first"])
    parser.add_argument("--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("demo")
    sub.add_parser("models")

    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8080)

    p_triage = sub.add_parser("triage")
    p_triage.add_argument("--ticket", required=True)

    p_incident = sub.add_parser("incident")
    p_incident.add_argument("--alert", required=True)

    p_compliance = sub.add_parser("compliance")
    p_compliance.add_argument("--question", required=True)

    sub.add_parser("reconcile")

    p_report = sub.add_parser("report")
    p_report.add_argument("--metrics", default=None)

    p_review = sub.add_parser("review")
    p_review.add_argument("--diff", required=True)

    p_exception = sub.add_parser("exception")
    p_exception.add_argument("--sku", required=True)
    p_exception.add_argument("--supplier", required=True)
    p_exception.add_argument("--days-late", type=int, default=0, dest="days_late")

    p_rfp = sub.add_parser("rfp")
    p_rfp.add_argument("--title", default="")
    p_rfp.add_argument("--text", required=True)

    p_outreach = sub.add_parser("outreach")
    p_outreach.add_argument("--company", required=True)
    p_outreach.add_argument("--product", default="")

    p_clinical = sub.add_parser("clinical")
    p_clinical.add_argument("--symptoms", default="")

    p_drugs = sub.add_parser("drugs")
    p_drugs.add_argument("--medications", required=True)

    p_lit = sub.add_parser("literature")
    p_lit.add_argument("--query", required=True)

    p_risk = sub.add_parser("risk")
    p_risk.add_argument("--patient-id", default="P001", dest="patient_id")
    p_risk.add_argument("--age", type=int, default=50)
    p_risk.add_argument("--diagnoses", default="")

    p_gaps = sub.add_parser("gaps")
    p_gaps.add_argument("--population", required=True)

    p_genomics = sub.add_parser("genomics")
    p_genomics.add_argument("--sample-id", required=True, dest="sample_id")
    p_genomics.add_argument("--variants", required=True)

    p_mental = sub.add_parser("mental")
    p_mental.add_argument("--concerns", required=True)

    p_trials = sub.add_parser("trials")
    p_trials.add_argument("--patient-id", default="P001", dest="patient_id")
    p_trials.add_argument("--age", type=int, default=45)
    p_trials.add_argument("--diagnoses", default="")
    p_trials.add_argument("--condition", default="")

    p_stock = sub.add_parser("stock")
    p_stock.add_argument("--ticker", required=True)
    p_stock.add_argument("--horizon", default="1m")
    p_stock.add_argument("--headline", default=None)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    dispatch = {
        "demo": cmd_demo, "models": cmd_models, "serve": cmd_serve,
        "triage": cmd_triage, "incident": cmd_incident, "compliance": cmd_compliance,
        "reconcile": cmd_reconcile, "report": cmd_report, "review": cmd_review,
        "exception": cmd_exception, "rfp": cmd_rfp, "outreach": cmd_outreach,
        "clinical": cmd_clinical, "drugs": cmd_drugs, "literature": cmd_literature,
        "risk": cmd_risk, "gaps": cmd_gaps, "genomics": cmd_genomics,
        "mental": cmd_mental, "trials": cmd_trials, "stock": cmd_stock,
    }
    fn = dispatch.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
