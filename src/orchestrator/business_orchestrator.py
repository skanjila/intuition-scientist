"""Business and Scientific/Medical Agent Orchestrator."""
from __future__ import annotations
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Optional

from src.llm.base import LLMBackend
from src.mcp.mcp_client import MCPClient
from src.models import (
    AgentResponse, AutonomyLevel, AUTONOMY_BASE_WEIGHT, ESCALATION_CONFIDENCE_THRESHOLD,
    ClinicalDecisionResult, ClinicalTrialMatch, ClinicalTrialsResult, ComplianceAnswer,
    Domain, DrugInteractionResult, EscalationDecision, ExceptionEvent, ExceptionResponse,
    GenomicRiskResult, HealthcareGapResult, HumanJudgment, IncidentContext, IncidentResponse,
    LiteratureSynthesisResult, MentalHealthTriageResult, OutreachResult, PatientRiskInput,
    PatientRiskResult, ReconciliationMatch, ReconciliationResult, ReportContext, ReportResult,
    RFPResult, StockPredictionInput, StockPredictionResult, TriageResult, CodeReviewResult,
)

logger = logging.getLogger(__name__)

_DEFAULT_AGENT_RESPONSE = AgentResponse(
    domain=Domain.CUSTOMER_SUPPORT,
    answer="[placeholder]",
    reasoning="",
    confidence=0.1,
    sources=[],
)


class BusinessOrchestrator:
    """Central entry point for all 18 use cases."""

    def __init__(
        self,
        backend_spec: str = "",
        backend=None,
        model_profile: str = "balanced",
        use_web_search: bool = False,
        max_workers: int = 4,
        agent_timeout_seconds: float = 30.0,
        verbose: bool = False,
    ):
        if backend is not None:
            self._backend = backend
        elif backend_spec:
            from src.llm.registry import get_backend
            self._backend = get_backend(backend_spec)
        else:
            from src.llm.model_registry import get_backend_with_fallback
            self._backend = get_backend_with_fallback("general", model_profile)

        self._mcp_client: Optional[MCPClient] = MCPClient() if use_web_search else None
        self._model_profile = model_profile
        self._max_workers = max_workers
        self._timeout = agent_timeout_seconds
        self._verbose = verbose

    def _make_agent(self, cls, tool_backend=None):
        return cls(backend=self._backend, mcp_client=self._mcp_client, tool_backend=tool_backend)

    def _query_parallel(self, agents, question, extra_context=""):
        results = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(agent.answer, question, extra_context=extra_context): agent
                for agent in agents
            }
            for future in futures:
                try:
                    resp = future.result(timeout=self._timeout)
                    results.append(resp)
                except (FuturesTimeout, Exception) as exc:
                    logger.warning("Agent call failed: %s", exc)
                    results.append(AgentResponse(
                        domain=Domain.CUSTOMER_SUPPORT,
                        answer=f"[agent unavailable: {exc}]",
                        reasoning="",
                        confidence=0.1,
                        sources=[],
                    ))
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    def _best(self, responses):
        if not responses:
            return _DEFAULT_AGENT_RESPONSE
        return max(responses, key=lambda r: r.confidence)

    def _blend(self, ai_text, human_judgment, autonomy, ai_confidence):
        if human_judgment is None:
            return ai_text, 0.0
        if human_judgment.override:
            return human_judgment.judgment, 1.0
        hw = AUTONOMY_BASE_WEIGHT[autonomy] * human_judgment.confidence
        if hw >= 0.5:
            blended = f"{human_judgment.judgment}\n\n[AI analysis ({1-hw:.0%} weight): {ai_text[:400]}]"
        else:
            blended = f"{ai_text}\n\n[Human judgment ({hw:.0%} weight): {human_judgment.judgment[:200]}]"
        return blended, hw

    def _escalate(self, ai_confidence, keywords, text, threshold=None):
        if threshold is None:
            threshold = ESCALATION_CONFIDENCE_THRESHOLD
        kw_hit = any(k.lower() in text.lower() for k in keywords)
        if ai_confidence < threshold or kw_hit:
            reason = (
                f"AI confidence {ai_confidence:.0%} below threshold"
                if ai_confidence < threshold
                else "Escalation keyword detected"
            )
            return EscalationDecision(needs_escalation=True, reason=reason, urgency="review")
        return EscalationDecision(needs_escalation=False, reason="")

    # ── Business use cases ──────────────────────────────────────────────────

    def triage(self, ticket_text: str, *, human_judgment=None, autonomy=AutonomyLevel.AI_PROPOSES, tool_backend=None) -> TriageResult:
        from src.agents.customer_support_agent import CustomerSupportAgent
        from src.agents.organizational_behavior_agent import OrganizationalBehaviorAgent
        agents = [self._make_agent(CustomerSupportAgent, tool_backend), self._make_agent(OrganizationalBehaviorAgent)]
        q = f"Triage this customer ticket: {ticket_text}"
        responses = self._query_parallel(agents, q)
        best = self._best(responses)
        escalation = self._escalate(best.confidence, ["P1","lawsuit","breach","outage","down","critical"], ticket_text + best.answer)
        blended, hw = self._blend(best.answer, human_judgment, autonomy, best.confidence)
        urgency = "P1" if any(w in ticket_text.lower() for w in ["outage","down","critical","all users","revenue"]) else "P3"
        return TriageResult(
            ticket_text=ticket_text,
            urgency=urgency,
            routing_department="Engineering" if "api" in ticket_text.lower() or "error" in ticket_text.lower() else "Support",
            draft_response=blended,
            escalation=escalation,
            human_judgment=human_judgment,
            autonomy_used=autonomy,
            ai_confidence=best.confidence,
            reasoning=best.reasoning,
        )

    def compliance_qa(self, question: str, *, human_judgment=None, autonomy=AutonomyLevel.AI_ASSISTS, tool_backend=None) -> ComplianceAnswer:
        from src.agents.legal_compliance_agent import LegalComplianceAgent
        from src.agents.organizational_behavior_agent import OrganizationalBehaviorAgent
        agents = [self._make_agent(LegalComplianceAgent, tool_backend), self._make_agent(OrganizationalBehaviorAgent)]
        responses = self._query_parallel(agents, question)
        best = self._best(responses)
        escalation = self._escalate(best.confidence, ["penalty","lawsuit","violation","criminal","regulatory action"], question + best.answer, threshold=0.60)
        blended, _ = self._blend(best.answer, human_judgment, autonomy, best.confidence)
        return ComplianceAnswer(
            question=question,
            answer=blended,
            escalation=escalation,
            human_judgment=human_judgment,
            autonomy_used=autonomy,
            ai_confidence=best.confidence,
            reasoning=best.reasoning,
            sources=best.sources,
        )

    def respond_to_incident(self, context, *, human_judgment=None, autonomy=AutonomyLevel.AI_PROPOSES, tool_backend=None) -> IncidentResponse:
        from src.agents.incident_response_agent import IncidentResponseAgent
        from src.agents.cybersecurity_agent import CybersecurityAgent
        from src.agents.enterprise_architecture_agent import EnterpriseArchitectureAgent
        alert = context.alert_payload if hasattr(context, "alert_payload") else str(context)
        logs = "\n".join(context.log_lines[:20]) if hasattr(context, "log_lines") else ""
        q = f"Incident alert: {alert}\nRecent logs:\n{logs}\nService graph: {getattr(context,'service_graph','')}"
        agents = [self._make_agent(IncidentResponseAgent, tool_backend), self._make_agent(CybersecurityAgent), self._make_agent(EnterpriseArchitectureAgent)]
        responses = self._query_parallel(agents, q)
        best = self._best(responses)
        severity = "P1" if any(w in alert.lower() for w in ["p1","critical","down","outage","all"]) else "P3"
        escalation = self._escalate(best.confidence, ["P1","critical","breach","data loss","security"], alert)
        blended, _ = self._blend(best.answer, human_judgment, autonomy, best.confidence)
        steps = [s.strip() for s in blended.split("\n") if s.strip() and (s.strip()[0].isdigit() or s.strip().startswith("-"))][:8]
        return IncidentResponse(
            alert_payload=alert,
            root_cause_hypotheses=[best.reasoning[:200]] if best.reasoning else ["Investigation in progress"],
            mitigation_steps=steps or [blended[:300]],
            severity=severity,
            escalation=escalation,
            human_judgment=human_judgment,
            autonomy_used=autonomy,
            ai_confidence=best.confidence,
            reasoning=best.reasoning,
        )

    def reconcile(self, ledger: list, invoices: list, *, human_judgment=None, autonomy=AutonomyLevel.AI_PROPOSES, tool_backend=None, materiality_threshold: float = 1000.0) -> ReconciliationResult:
        from src.agents.finance_reconciliation_agent import FinanceReconciliationAgent
        from src.agents.finance_economics_agent import FinanceEconomicsAgent
        q = f"Reconcile {len(ledger)} ledger entries against {len(invoices)} invoices. Materiality threshold: ${materiality_threshold}. Ledger sample: {str(ledger[:3])}. Invoice sample: {str(invoices[:3])}."
        agents = [self._make_agent(FinanceReconciliationAgent, tool_backend), self._make_agent(FinanceEconomicsAgent)]
        responses = self._query_parallel(agents, q)
        best = self._best(responses)
        escalation = self._escalate(best.confidence, ["fraud","error","variance","unmatched","discrepancy"], best.answer)
        blended, _ = self._blend(best.answer, human_judgment, autonomy, best.confidence)
        matched = []
        for i, (l, inv) in enumerate(zip(ledger, invoices)):
            lid = str(l.get("id", f"L{i}"))
            iid = str(inv.get("id", f"I{i}"))
            la = float(l.get("amount", 0))
            ia = float(inv.get("amount", 0))
            variance = abs(la - ia)
            matched.append(ReconciliationMatch(ledger_id=lid, invoice_id=iid, match_confidence=1.0 if variance == 0 else max(0.5, 1.0 - variance/max(la,ia,1)), variance=variance))
        unmatched_l = [str(l.get("id", f"L{i}")) for i, l in enumerate(ledger[len(invoices):])]
        unmatched_i = [str(inv.get("id", f"I{i}")) for i, inv in enumerate(invoices[len(ledger):])]
        return ReconciliationResult(
            matched_pairs=matched,
            unmatched_ledger_ids=unmatched_l,
            unmatched_invoice_ids=unmatched_i,
            audit_narrative=blended,
            escalation=escalation,
            human_judgment=human_judgment,
            autonomy_used=autonomy,
            ai_confidence=best.confidence,
        )

    def outreach(self, company: str, product: str = "", deal_value: float = 0.0, *, human_judgment=None, autonomy=AutonomyLevel.AI_ASSISTS, tool_backend=None) -> OutreachResult:
        from src.agents.marketing_growth_agent import MarketingGrowthAgent
        from src.agents.strategy_intelligence_agent import StrategyIntelligenceAgent
        q = f"Research {company} and draft a sales outreach email for product '{product}'. Deal value: ${deal_value:,.0f}."
        agents = [self._make_agent(MarketingGrowthAgent, tool_backend), self._make_agent(StrategyIntelligenceAgent)]
        responses = self._query_parallel(agents, q)
        best = self._best(responses)
        escalation = self._escalate(best.confidence, [], "", threshold=0.99)
        if deal_value > 500_000:
            escalation = EscalationDecision(needs_escalation=True, reason=f"Deal value ${deal_value:,.0f} exceeds $500K — requires senior sales review", urgency="review", checkpoint="VP Sales approval")
        blended, _ = self._blend(best.answer, human_judgment, autonomy, best.confidence)
        return OutreachResult(
            company_name=company,
            company_summary=best.reasoning[:300] if best.reasoning else f"AI-researched summary of {company}",
            email_draft=blended,
            escalation=escalation,
            human_judgment=human_judgment,
            autonomy_used=autonomy,
            ai_confidence=best.confidence,
        )

    def generate_report(self, context, *, human_judgment=None, autonomy=AutonomyLevel.FULL_AUTO, tool_backend=None) -> ReportResult:
        from src.agents.analytics_agent import AnalyticsAgent
        from src.agents.finance_economics_agent import FinanceEconomicsAgent
        metrics = context.metrics if hasattr(context, "metrics") else context
        q = f"Generate a {getattr(context,'audience','ops')} analytics report for period '{getattr(context,'reporting_period','')}'. Metrics: {str(metrics)[:500]}. Targets: {str(getattr(context,'kpi_targets',{}))[:300]}."
        agents = [self._make_agent(AnalyticsAgent, tool_backend), self._make_agent(FinanceEconomicsAgent)]
        responses = self._query_parallel(agents, q)
        best = self._best(responses)
        anomalies = []
        if hasattr(context, "kpi_targets") and hasattr(context, "metrics"):
            for k, v in context.metrics.items():
                t = context.kpi_targets.get(k)
                if t and isinstance(v, (int,float)) and isinstance(t, (int,float)):
                    pct = (float(v) - float(t)) / float(t) * 100
                    if abs(pct) > 20:
                        anomalies.append(f"{k}: {pct:+.1f}% vs target")
        escalation = self._escalate(best.confidence, [], " ".join(anomalies), threshold=0.99)
        if anomalies:
            escalation = EscalationDecision(needs_escalation=True, reason=f"KPI anomaly detected: {anomalies[0]}", urgency="review", checkpoint="Manager review of anomalies")
        blended, _ = self._blend(best.answer, human_judgment, autonomy, best.confidence)
        return ReportResult(
            headline=blended[:120].split("\n")[0] if blended else "Analytics Report",
            trend_analysis=blended,
            anomalies=anomalies,
            escalation=escalation,
            human_judgment=human_judgment,
            autonomy_used=autonomy,
            ai_confidence=best.confidence,
        )

    def review_pr(self, diff: str, description: str = "", *, human_judgment=None, autonomy=AutonomyLevel.AI_PROPOSES, tool_backend=None) -> CodeReviewResult:
        from src.agents.code_review_agent import CodeReviewAgent
        from src.agents.cybersecurity_agent import CybersecurityAgent
        q = f"Review this code change:\nDescription: {description}\n\nDiff:\n{diff[:3000]}"
        agents = [self._make_agent(CodeReviewAgent, tool_backend), self._make_agent(CybersecurityAgent)]
        responses = self._query_parallel(agents, q)
        best = self._best(responses)
        sec_keywords = ["sql injection","xss","csrf","hardcoded secret","password","api key","eval(","exec(","os.system"]
        escalation = self._escalate(best.confidence, sec_keywords, diff.lower() + best.answer.lower())
        blended, _ = self._blend(best.answer, human_judgment, autonomy, best.confidence)
        risk = 0.3 if best.confidence > 0.7 else 0.6
        if any(k in diff.lower() for k in sec_keywords):
            risk = 0.9
        approval = "approve" if risk < 0.3 else ("request_changes" if risk > 0.6 else "needs_review")
        return CodeReviewResult(
            diff_summary=description or diff[:100],
            overall_reasoning=blended,
            risk_score=risk,
            approval_recommendation=approval,
            escalation=escalation,
            human_judgment=human_judgment,
            autonomy_used=autonomy,
            ai_confidence=best.confidence,
        )

    def handle_exception(self, event, *, human_judgment=None, autonomy=AutonomyLevel.AI_PROPOSES, tool_backend=None) -> ExceptionResponse:
        from src.agents.supply_chain_agent import SupplyChainAgent
        from src.agents.finance_economics_agent import FinanceEconomicsAgent
        exc_type = event.exception_type if hasattr(event, "exception_type") else str(event)
        sku = getattr(event, "sku", "unknown")
        supplier = getattr(event, "supplier", "unknown")
        days_late = getattr(event, "eta_days_late", 0)
        cost_expedite = getattr(event, "cost_to_expedite", 0)
        q = f"Supply chain exception: {exc_type} for SKU {sku} from {supplier}. {days_late} days late. Cost to expedite: ${cost_expedite:,.0f}. Inventory: {getattr(event,'current_inventory',0)} units."
        agents = [self._make_agent(SupplyChainAgent, tool_backend), self._make_agent(FinanceEconomicsAgent)]
        responses = self._query_parallel(agents, q)
        best = self._best(responses)
        escalation = self._escalate(best.confidence, ["critical","no alternative","stockout","production halt"], getattr(event,"notes","") + best.answer)
        blended, _ = self._blend(best.answer, human_judgment, autonomy, best.confidence)
        action = "expedite" if days_late > 7 and cost_expedite < 50000 else ("substitute" if days_late > 14 else "backorder")
        return ExceptionResponse(
            exception_type=exc_type,
            sku=sku,
            recommended_action=action,
            financial_impact_estimate=cost_expedite,
            reasoning=blended,
            escalation=escalation,
            human_judgment=human_judgment,
            autonomy_used=autonomy,
            ai_confidence=best.confidence,
        )

    def draft_rfp(self, rfp_text: str, rfp_title: str = "", *, human_judgment=None, autonomy=AutonomyLevel.AI_ASSISTS, tool_backend=None) -> RFPResult:
        from src.agents.rfp_agent import RFPAgent
        from src.agents.strategy_intelligence_agent import StrategyIntelligenceAgent
        from src.agents.legal_compliance_agent import LegalComplianceAgent
        q = f"Draft an RFP response for: '{rfp_title}'\n\nRFP text:\n{rfp_text[:3000]}"
        agents = [self._make_agent(RFPAgent, tool_backend), self._make_agent(StrategyIntelligenceAgent), self._make_agent(LegalComplianceAgent)]
        responses = self._query_parallel(agents, q)
        best = self._best(responses)
        escalation = self._escalate(best.confidence, ["unlimited liability","indemnification","penalty","liquidated damages"], rfp_text + best.answer, threshold=0.65)
        blended, _ = self._blend(best.answer, human_judgment, autonomy, best.confidence)
        risk_flags = [w for w in ["unlimited liability","indemnification","penalty","sole discretion","liquidated damages"] if w in rfp_text.lower()]
        return RFPResult(
            rfp_title=rfp_title,
            section_drafts={"executive_summary": blended[:500], "technical_approach": blended[500:1000], "pricing": "See commercial section"},
            overall_strategy=best.reasoning[:300] if best.reasoning else "",
            risk_flags=risk_flags,
            escalation=escalation,
            human_judgment=human_judgment,
            autonomy_used=autonomy,
            ai_confidence=best.confidence,
        )

    # ── Medical use cases ───────────────────────────────────────────────────

    def clinical_decision(self, assessment, *, human_judgment=None, autonomy=AutonomyLevel.HUMAN_FIRST, tool_backend=None) -> ClinicalDecisionResult:
        from src.agents.clinical_decision_support_agent import ClinicalDecisionSupportAgent
        from src.agents.patient_risk_agent import PatientRiskAgent
        summary = assessment.patient_summary if hasattr(assessment, "patient_summary") else str(assessment)
        symptoms = getattr(assessment, "symptoms", [])
        meds = getattr(assessment, "current_medications", [])
        q = f"Clinical assessment for: {summary}\nSymptoms: {', '.join(symptoms)}\nMedications: {', '.join(meds)}\nLab values: {str(getattr(assessment,'lab_values',{}))[:200]}"
        agents = [self._make_agent(ClinicalDecisionSupportAgent, tool_backend), self._make_agent(PatientRiskAgent)]
        responses = self._query_parallel(agents, q)
        best = self._best(responses)
        escalation = EscalationDecision(needs_escalation=True, reason="Clinical decision requires physician validation", urgency="immediate", checkpoint="Physician must review before any clinical action")
        blended, _ = self._blend(best.answer, human_judgment, autonomy, best.confidence)
        red_flags = [w for w in ["chest pain","stroke","sepsis","anaphylaxis","suicide","overdose"] if w in summary.lower() or w in " ".join(symptoms).lower()]
        if red_flags:
            escalation.urgency = "immediate"
        return ClinicalDecisionResult(
            patient_summary=summary,
            differential_diagnoses=[best.answer[:200]] if best.answer else [],
            red_flags=red_flags,
            escalation=escalation,
            human_judgment=human_judgment,
            autonomy_used=autonomy,
            ai_confidence=best.confidence,
        )

    def check_drug_interactions(self, medications: list, patient_context: str = "", *, human_judgment=None, autonomy=AutonomyLevel.AI_PROPOSES, tool_backend=None) -> DrugInteractionResult:
        from src.agents.drug_interaction_agent import DrugInteractionAgent
        q = f"Check drug interactions for: {', '.join(medications)}. Patient context: {patient_context}"
        agents = [self._make_agent(DrugInteractionAgent, tool_backend)]
        responses = self._query_parallel(agents, q)
        best = self._best(responses)
        escalation = EscalationDecision(needs_escalation=True, reason="Drug interactions require pharmacist/physician review", urgency="review", checkpoint="Pharmacist must validate before dispensing")
        blended, _ = self._blend(best.answer, human_judgment, autonomy, best.confidence)
        return DrugInteractionResult(
            medications=medications,
            severity_summary=blended[:300],
            escalation=escalation,
            human_judgment=human_judgment,
            autonomy_used=autonomy,
            ai_confidence=best.confidence,
        )

    def synthesize_literature(self, query: str, *, human_judgment=None, autonomy=AutonomyLevel.AI_ASSISTS, tool_backend=None) -> LiteratureSynthesisResult:
        from src.agents.medical_literature_agent import MedicalLiteratureAgent
        agents = [self._make_agent(MedicalLiteratureAgent, tool_backend)]
        responses = self._query_parallel(agents, query)
        best = self._best(responses)
        escalation = self._escalate(best.confidence, [], "", threshold=0.40)
        blended, _ = self._blend(best.answer, human_judgment, autonomy, best.confidence)
        return LiteratureSynthesisResult(
            query=query,
            synthesis=blended,
            key_findings=[best.answer[:150]] if best.answer else [],
            escalation=escalation,
            human_judgment=human_judgment,
            autonomy_used=autonomy,
            ai_confidence=best.confidence,
        )

    def stratify_patient_risk(self, patient, *, human_judgment=None, autonomy=AutonomyLevel.AI_PROPOSES, tool_backend=None) -> PatientRiskResult:
        from src.agents.patient_risk_agent import PatientRiskAgent
        from src.agents.clinical_decision_support_agent import ClinicalDecisionSupportAgent
        pid = patient.patient_id if hasattr(patient, "patient_id") else "unknown"
        age = getattr(patient, "age", 0)
        diags = getattr(patient, "diagnoses", [])
        meds = getattr(patient, "medications", [])
        sdoh = getattr(patient, "social_determinants", {})
        q = f"Stratify risk for patient {pid}, age {age}. Diagnoses: {', '.join(diags)}. Medications: {', '.join(meds)}. SDOH: {str(sdoh)[:200]}."
        agents = [self._make_agent(PatientRiskAgent, tool_backend), self._make_agent(ClinicalDecisionSupportAgent)]
        responses = self._query_parallel(agents, q)
        best = self._best(responses)
        risk_level = "high" if age > 65 and len(diags) > 2 else ("moderate" if len(diags) > 0 else "low")
        escalation = EscalationDecision(
            needs_escalation=risk_level=="high",
            reason="High-risk patient requires care team review" if risk_level=="high" else "",
            urgency="review" if risk_level=="high" else "informational"
        )
        blended, _ = self._blend(best.answer, human_judgment, autonomy, best.confidence)
        return PatientRiskResult(
            patient_id=pid,
            risk_level=risk_level,
            risk_factors=diags[:5],
            recommended_interventions=[blended[:200]],
            escalation=escalation,
            human_judgment=human_judgment,
            autonomy_used=autonomy,
            ai_confidence=best.confidence,
        )

    def analyze_healthcare_gaps(self, region_or_population: str, *, human_judgment=None, autonomy=AutonomyLevel.AI_ASSISTS, tool_backend=None) -> HealthcareGapResult:
        from src.agents.healthcare_access_agent import HealthcareAccessAgent
        from src.agents.patient_risk_agent import PatientRiskAgent
        q = f"Analyze healthcare access gaps for: {region_or_population}"
        agents = [self._make_agent(HealthcareAccessAgent, tool_backend), self._make_agent(PatientRiskAgent)]
        responses = self._query_parallel(agents, q)
        best = self._best(responses)
        escalation = self._escalate(best.confidence, [], "", threshold=0.35)
        blended, _ = self._blend(best.answer, human_judgment, autonomy, best.confidence)
        return HealthcareGapResult(
            region_or_population=region_or_population,
            identified_gaps=[blended[:200]],
            recommended_interventions=[best.reasoning[:200]] if best.reasoning else [],
            escalation=escalation,
            human_judgment=human_judgment,
            autonomy_used=autonomy,
            ai_confidence=best.confidence,
        )

    def assess_genomic_risk(self, sample_id: str, variants: list, patient_context: str = "", *, human_judgment=None, autonomy=AutonomyLevel.HUMAN_FIRST, tool_backend=None) -> GenomicRiskResult:
        from src.agents.genomics_medicine_agent import GenomicsMedicineAgent
        q = f"Assess genomic risk for sample {sample_id}. Variants: {', '.join(variants[:10])}. Context: {patient_context}"
        agents = [self._make_agent(GenomicsMedicineAgent, tool_backend)]
        responses = self._query_parallel(agents, q)
        best = self._best(responses)
        escalation = EscalationDecision(needs_escalation=True, reason="Genomic results require genetic counselor review", urgency="review", checkpoint="Genetic counselor consult required before results disclosed")
        blended, _ = self._blend(best.answer, human_judgment, autonomy, best.confidence)
        return GenomicRiskResult(
            sample_id=sample_id,
            analyzed_variants=variants[:20],
            recommended_screenings=[blended[:200]],
            genetic_counseling_needed=True,
            escalation=escalation,
            human_judgment=human_judgment,
            autonomy_used=autonomy,
            ai_confidence=best.confidence,
        )

    def triage_mental_health(self, presenting_concerns: str, *, human_judgment=None, autonomy=AutonomyLevel.HUMAN_FIRST, tool_backend=None) -> MentalHealthTriageResult:
        from src.agents.mental_health_triage_agent import MentalHealthTriageAgent
        agents = [self._make_agent(MentalHealthTriageAgent, tool_backend)]
        responses = self._query_parallel(agents, presenting_concerns)
        best = self._best(responses)
        crisis_words = ["suicide","suicidal","kill myself","hurt myself","self harm","overdose","crisis"]
        crisis_indicators = [w for w in crisis_words if w in presenting_concerns.lower()]
        urgency = "immediate" if crisis_indicators else "review"
        escalation = EscalationDecision(needs_escalation=True, reason="Mental health triage requires clinician review", urgency=urgency, checkpoint="Clinician must review before any intervention")
        blended, _ = self._blend(best.answer, human_judgment, autonomy, best.confidence)
        return MentalHealthTriageResult(
            presenting_concerns=presenting_concerns,
            risk_level="high" if crisis_indicators else "moderate",
            crisis_indicators=crisis_indicators,
            recommended_resources=["988 Suicide & Crisis Lifeline", "Crisis Text Line: Text HOME to 741741"],
            suggested_interventions=[blended[:200]],
            follow_up_urgency=urgency,
            escalation=escalation,
            human_judgment=human_judgment,
            autonomy_used=autonomy,
            ai_confidence=best.confidence,
        )

    def match_clinical_trials(self, patient, condition: str = "", *, human_judgment=None, autonomy=AutonomyLevel.AI_PROPOSES, tool_backend=None) -> ClinicalTrialsResult:
        from src.agents.clinical_trials_agent import ClinicalTrialsAgent
        pid = patient.patient_id if hasattr(patient, "patient_id") else "unknown"
        age = getattr(patient, "age", 0)
        diags = getattr(patient, "diagnoses", [])
        q = f"Match clinical trials for patient {pid}, age {age}, condition: {condition or ', '.join(diags[:3])}."
        agents = [self._make_agent(ClinicalTrialsAgent, tool_backend)]
        responses = self._query_parallel(agents, q)
        best = self._best(responses)
        escalation = EscalationDecision(needs_escalation=True, reason="Trial eligibility requires physician/PI confirmation", urgency="review", checkpoint="Physician must confirm eligibility before enrollment")
        blended, _ = self._blend(best.answer, human_judgment, autonomy, best.confidence)
        return ClinicalTrialsResult(
            patient_summary=f"Patient {pid}, age {age}, {condition or ', '.join(diags[:2])}",
            matched_trials=[ClinicalTrialMatch(trial_id="NCT-MOCK-001", title=blended[:100], eligibility_match="potential")],
            escalation=escalation,
            human_judgment=human_judgment,
            autonomy_used=autonomy,
            ai_confidence=best.confidence,
        )

    # ── Stock market ────────────────────────────────────────────────────────

    def predict_stock(self, stock_input, *, human_judgment=None, autonomy=AutonomyLevel.AI_ASSISTS, tool_backend=None) -> StockPredictionResult:
        from src.agents.stock_market_agent import StockMarketAgent
        from src.agents.finance_economics_agent import FinanceEconomicsAgent
        from src.agents.strategy_intelligence_agent import StrategyIntelligenceAgent
        ticker = stock_input.ticker if hasattr(stock_input, "ticker") else str(stock_input)
        horizon = getattr(stock_input, "horizon", "1m")
        headlines = getattr(stock_input, "news_headlines", [])
        macro = getattr(stock_input, "macro_context", "")
        human_thesis = getattr(stock_input, "human_thesis", "")
        q = (
            f"Analyze {ticker} for a {horizon} investment horizon.\n"
            f"News: {'; '.join(headlines[:5])}\n"
            f"Macro context: {macro}\n"
            f"Human analyst thesis: {human_thesis}\n"
            f"Provide: direction (bullish/bearish/neutral), confidence_pct, thesis, bull_case, bear_case, catalysts, risk_factors, suggested_position_sizing, stop_loss_note."
        )
        effective_tool = tool_backend
        if effective_tool is None:
            try:
                from src.mcp.financial_data_backend import FinancialDataBackend
                effective_tool = FinancialDataBackend()
            except Exception:
                pass
        agents = [
            self._make_agent(StockMarketAgent, effective_tool),
            self._make_agent(FinanceEconomicsAgent),
            self._make_agent(StrategyIntelligenceAgent),
        ]
        responses = self._query_parallel(agents, q)
        best = self._best(responses)
        effective_judgment = human_judgment
        if human_thesis and human_judgment is None:
            effective_judgment = HumanJudgment(context=f"{ticker} analyst thesis", judgment=human_thesis, confidence=0.6)
        blended, hw = self._blend(best.answer, effective_judgment, autonomy, best.confidence)
        escalation = self._escalate(best.confidence, ["earnings","FDA","merger","delisting","investigation"], blended, threshold=0.55)
        answer_lower = blended.lower()
        direction = "bullish" if answer_lower.count("bullish") > answer_lower.count("bearish") else ("bearish" if "bearish" in answer_lower else "neutral")
        confidence_pct = round(best.confidence * 100, 1)
        sizing = "avoid" if direction == "bearish" else ("full" if confidence_pct > 75 else ("half" if confidence_pct > 60 else "starter"))
        return StockPredictionResult(
            ticker=ticker,
            direction=direction,
            confidence_pct=confidence_pct,
            thesis=blended[:400],
            bull_case=[f"AI ensemble ({1-hw:.0%}): {best.answer[:150]}"],
            bear_case=["Market uncertainty — see reasoning"],
            catalysts=[h[:80] for h in headlines[:3]],
            risk_factors=["AI analysis is not investment advice — consult a financial advisor"],
            suggested_position_sizing=sizing,
            stop_loss_note="Set stop-loss at 7–10% below entry based on your risk tolerance",
            escalation=escalation,
            human_judgment=effective_judgment,
            autonomy_used=autonomy,
            ai_confidence=best.confidence,
        )

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
