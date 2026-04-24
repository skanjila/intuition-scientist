# Business Agent Platform — Use Case Index

This platform provides 18 AI-assisted use cases across three domains: **Business Operations**, **Healthcare/Medical**, and **Financial Markets**. All use cases follow a consistent human-AI collaboration model with configurable autonomy levels.

---

## Architecture Overview

```
CLI / REST API
      │
      ▼
BusinessOrchestrator
      │
      ├── ThreadPoolExecutor (parallel agent queries)
      │         │
      │         ├── Agent 1 (domain specialist)
      │         ├── Agent 2 (supporting specialist)
      │         └── Agent N ...
      │
      ├── Human-AI Blending (_blend)
      ├── Escalation Engine (_escalate)
      └── Structured Result dataclass
```

---

## Business Use Cases

| # | Use Case | Method | Default Autonomy | Result Type |
|---|----------|--------|-----------------|-------------|
| 1 | Customer Support Triage | `triage()` | `AI_PROPOSES` | `TriageResult` |
| 2 | Compliance Q&A | `compliance_qa()` | `AI_ASSISTS` | `ComplianceAnswer` |
| 3 | Incident Response | `respond_to_incident()` | `AI_PROPOSES` | `IncidentResponse` |
| 4 | Finance Reconciliation | `reconcile()` | `AI_PROPOSES` | `ReconciliationResult` |
| 5 | Sales Outreach | `outreach()` | `AI_ASSISTS` | `OutreachResult` |
| 6 | Analytics Report | `generate_report()` | `FULL_AUTO` | `ReportResult` |
| 7 | Code Review | `review_pr()` | `AI_PROPOSES` | `CodeReviewResult` |
| 8 | Supply Chain Exception | `handle_exception()` | `AI_PROPOSES` | `ExceptionResponse` |
| 9 | RFP Response | `draft_rfp()` | `AI_ASSISTS` | `RFPResult` |

## Healthcare Use Cases

| # | Use Case | Method | Default Autonomy | Result Type |
|---|----------|--------|-----------------|-------------|
| 10 | Clinical Decision Support | `clinical_decision()` | `HUMAN_FIRST` | `ClinicalDecisionResult` |
| 11 | Drug Interaction Check | `check_drug_interactions()` | `AI_PROPOSES` | `DrugInteractionResult` |
| 12 | Medical Literature Synthesis | `synthesize_literature()` | `AI_ASSISTS` | `LiteratureSynthesisResult` |
| 13 | Patient Risk Stratification | `stratify_patient_risk()` | `AI_PROPOSES` | `PatientRiskResult` |
| 14 | Healthcare Gap Analysis | `analyze_healthcare_gaps()` | `AI_ASSISTS` | `HealthcareGapResult` |
| 15 | Genomic Risk Assessment | `assess_genomic_risk()` | `HUMAN_FIRST` | `GenomicRiskResult` |
| 16 | Mental Health Triage | `triage_mental_health()` | `HUMAN_FIRST` | `MentalHealthTriageResult` |
| 17 | Clinical Trial Matching | `match_clinical_trials()` | `AI_PROPOSES` | `ClinicalTrialsResult` |

## Financial Markets

| # | Use Case | Method | Default Autonomy | Result Type |
|---|----------|--------|-----------------|-------------|
| 18 | Stock Market Prediction | `predict_stock()` | `AI_ASSISTS` | `StockPredictionResult` |

---

## Quick Start

```python
from src.orchestrator.business_orchestrator import BusinessOrchestrator
from src.llm.mock_backend import MockBackend

orch = BusinessOrchestrator(backend=MockBackend())

# Business
result = orch.triage("Payment API down for all users")
print(result.urgency, result.draft_response)

# Medical
from src.models import ClinicalAssessmentInput
result = orch.clinical_decision(
    ClinicalAssessmentInput("45F chest pain", symptoms=["chest pain"])
)
print(result.escalation.needs_escalation)  # Always True

# Financial
from src.models import StockPredictionInput
result = orch.predict_stock(StockPredictionInput("NVDA", "1m"))
print(result.direction, result.confidence_pct)
```

## CLI Usage

```bash
# Run all 18 use case demos
python main.py demo

# Triage a ticket
python main.py triage --ticket "API is down for enterprise customers"

# Stock analysis
python main.py stock --ticker AAPL --horizon 1w

# Start REST API server
python main.py serve --port 8080
```

## REST API

Start the server with `python main.py serve`, then:

```bash
curl -X POST http://localhost:8080/triage \
  -H 'Content-Type: application/json' \
  -d '{"ticket_text": "Payment API returning 500 errors"}'

curl -X POST http://localhost:8080/stock \
  -H 'Content-Type: application/json' \
  -d '{"ticker": "NVDA", "horizon": "1m"}'
```

---

## Autonomy Levels

| Level | Human Weight | Description |
|-------|-------------|-------------|
| `FULL_AUTO` | 0% | AI acts fully autonomously |
| `AI_PROPOSES` | 20% | AI proposes, human can override |
| `AI_ASSISTS` | 50% | Equal AI/human collaboration |
| `HUMAN_FIRST` | 80% | Human judgment dominant, AI provides analysis |

## Escalation

Every result includes an `EscalationDecision` with:
- `needs_escalation`: bool
- `reason`: explanation
- `urgency`: `"informational"` | `"review"` | `"immediate"`
- `checkpoint`: optional review step description

Medical use cases **always** set `needs_escalation=True`.

---

## Detailed Use Case Documentation

- [01 — Customer Support Triage](use_cases/01_customer_support_triage.md)
- [02 — Compliance Q&A](use_cases/02_compliance_qa.md)
- [03 — Incident Response](use_cases/03_incident_response.md)
- [04 — Finance Reconciliation](use_cases/04_finance_reconciliation.md)
- [05 — Sales Outreach](use_cases/05_sales_outreach.md)
- [06 — Analytics Report](use_cases/06_analytics_report.md)
- [07 — Code Review](use_cases/07_code_review.md)
- [08 — Supply Chain Exception](use_cases/08_supply_chain_exception.md)
- [09 — RFP Response](use_cases/09_rfp_response.md)
- [10 — Clinical Decision Support](use_cases/10_clinical_decision_support.md)
- [11 — Drug Interaction Check](use_cases/11_drug_interaction.md)
- [12 — Medical Literature Synthesis](use_cases/12_medical_literature.md)
- [13 — Patient Risk Stratification](use_cases/13_patient_risk.md)
- [14 — Healthcare Gap Analysis](use_cases/14_healthcare_gaps.md)
- [15 — Genomic Risk Assessment](use_cases/15_genomic_risk.md)
- [16 — Mental Health Triage](use_cases/16_mental_health_triage.md)
- [17 — Clinical Trial Matching](use_cases/17_clinical_trials.md)
- [18 — Stock Market Prediction](use_cases/18_stock_market_prediction.md)
