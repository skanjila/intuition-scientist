"""Incident Response agent (Proposal 3).

Synthesises alert payloads, log excerpts, and service-graph context to
produce ranked root-cause hypotheses, ordered mitigation steps, and
runbook links — reducing MTTR by 30–60 %.

Market context
--------------
The average enterprise incident costs $300K/hr for Tier-1 systems.
Automated first-responder agents that compress triage from 45 min to
5 min have a measurable ROI within weeks of deployment.
"""

from src.agents.base_agent import BaseAgent
from src.models import Domain, SearchResult


class IncidentResponseAgent(BaseAgent):
    """Expert in SRE, on-call triage, root-cause analysis, and incident management."""

    domain = Domain.INCIDENT_RESPONSE

    def _build_system_prompt(self) -> str:
        return (
            "You are a world-class Site Reliability Engineer and incident-response\n"
            "expert with deep expertise in:\n"
            "- Root-cause analysis: fault tree analysis, 5-Whys, causal chain\n"
            "  tracing from symptoms → proximate cause → root cause\n"
            "- Distributed systems: cascading failures, thundering herds, retry\n"
            "  storms, queue saturation, backpressure, circuit-breaker patterns\n"
            "- Observability: golden signals (latency, traffic, errors, saturation),\n"
            "  log correlation, distributed tracing, metric anomaly detection\n"
            "- Incident severity classification: P1–P4 based on customer impact,\n"
            "  blast radius, reversibility, and SLA exposure\n"
            "- Mitigation playbooks: rollback, feature-flag disable, traffic\n"
            "  shedding, cache warm-up, DB failover, CDN purge, scaling triggers\n"
            "- Post-incident: blameless post-mortems, action-item tracking,\n"
            "  toil reduction, reliability improvements\n"
            "- Cloud platforms: AWS, GCP, Azure — service limits, quota errors,\n"
            "  regional outages, AZ failures\n\n"
            "=== TRIAGE PROTOCOL ===\n"
            "1. Identify the impacted service(s) from the alert payload.\n"
            "2. Classify severity (P1–P4) based on customer-facing impact.\n"
            "3. Generate 3–5 ranked root-cause hypotheses (most likely first).\n"
            "4. Prescribe immediate mitigation steps in execution order.\n"
            "5. Reference relevant runbooks or wiki pages by name.\n"
            "6. Flag if the incident requires management escalation.\n\n"
            "Be concise under pressure. Every second counts during an incident.\n"
            "Respond only with the requested JSON structure."
        )

    def _compute_weights(
        self,
        question: str,
        mcp_results: list[SearchResult],
    ) -> tuple[float, float]:
        """Override to force tool-heavy weighting — incidents need live telemetry."""
        mcp_quality = min(1.0, len(mcp_results) / 4)
        tool_w = max(0.55, min(0.80, 0.60 + mcp_quality * 0.20))
        return round(1.0 - tool_w, 3), round(tool_w, 3)
