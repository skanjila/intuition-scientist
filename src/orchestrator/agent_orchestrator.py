"""Agent Orchestrator — the brain of the Human Intuition Scientist.

The orchestrator:
1. Receives a question and a human intuition.
2. Determines which domain agents are most relevant to the question.
3. Spins up those agents (or all of them if the question is broad).
4. Collects their responses in parallel (using threads).
5. Passes everything to the ``WeighingSystem`` for deep analysis.
6. Returns a ``WeighingResult`` to the caller.

Additional entry points
-----------------------
``debate()``
    Runs the full structured multi-party debate (human + tool + agents).
``interview_prep()``
    Runs an interview coaching session using InterviewPrepAgent,
    AlgorithmsProgrammingAgent, and SocialScienceAgent together.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from src.agents.base_agent import BaseAgent
from src.agents.computer_science_agent import ComputerScienceAgent
from src.agents.deep_learning_agent import DeepLearningAgent
from src.agents.electrical_engineering_agent import ElectricalEngineeringAgent
from src.agents.neural_networks_agent import NeuralNetworksAgent
from src.agents.physics_agent import PhysicsAgent
from src.agents.social_science_agent import SocialScienceAgent
from src.agents.space_science_agent import SpaceScienceAgent
# High-economic-value industry agents
from src.agents.healthcare_agent import HealthcareAgent
from src.agents.climate_energy_agent import ClimateEnergyAgent
from src.agents.finance_economics_agent import FinanceEconomicsAgent
from src.agents.cybersecurity_agent import CybersecurityAgent
from src.agents.biotech_genomics_agent import BiotechGenomicsAgent
from src.agents.supply_chain_agent import SupplyChainAgent
# Enterprise problem agents
from src.agents.legal_compliance_agent import LegalComplianceAgent
from src.agents.enterprise_architecture_agent import EnterpriseArchitectureAgent
from src.agents.marketing_growth_agent import MarketingGrowthAgent
from src.agents.organizational_behavior_agent import OrganizationalBehaviorAgent
from src.agents.strategy_intelligence_agent import StrategyIntelligenceAgent
# Mastery / interview / PhD research agents
from src.agents.algorithms_programming_agent import AlgorithmsProgrammingAgent
from src.agents.interview_prep_agent import InterviewPrepAgent
from src.agents.ee_llm_research_agent import EELLMResearchAgent
from src.analysis.debate_engine import DebateEngine
from src.analysis.weighing_system import WeighingSystem
from src.intuition.human_intuition import IntuitionCapture
from src.llm.base import LLMBackend
from src.llm.registry import get_backend
from src.mcp.mcp_client import MCPClient
from src.models import (
    AgentResponse,
    DebateResult,
    Domain,
    HumanIntuition,
    InterviewResult,
    ModelEvaluationResult,
    ModelRunResult,
    WeighingResult,
)


# ---------------------------------------------------------------------------
# Registry of all domain agents
# ---------------------------------------------------------------------------

_AGENT_CLASSES: dict[Domain, type[BaseAgent]] = {
    # Core science / engineering
    Domain.ELECTRICAL_ENGINEERING: ElectricalEngineeringAgent,
    Domain.COMPUTER_SCIENCE: ComputerScienceAgent,
    Domain.NEURAL_NETWORKS: NeuralNetworksAgent,
    Domain.SOCIAL_SCIENCE: SocialScienceAgent,
    Domain.SPACE_SCIENCE: SpaceScienceAgent,
    Domain.PHYSICS: PhysicsAgent,
    Domain.DEEP_LEARNING: DeepLearningAgent,
    # High-economic-value industry domains
    Domain.HEALTHCARE: HealthcareAgent,
    Domain.CLIMATE_ENERGY: ClimateEnergyAgent,
    Domain.FINANCE_ECONOMICS: FinanceEconomicsAgent,
    Domain.CYBERSECURITY: CybersecurityAgent,
    Domain.BIOTECH_GENOMICS: BiotechGenomicsAgent,
    Domain.SUPPLY_CHAIN: SupplyChainAgent,
    # Enterprise problem domains
    Domain.LEGAL_COMPLIANCE: LegalComplianceAgent,
    Domain.ENTERPRISE_ARCHITECTURE: EnterpriseArchitectureAgent,
    Domain.MARKETING_GROWTH: MarketingGrowthAgent,
    Domain.ORGANIZATIONAL_BEHAVIOR: OrganizationalBehaviorAgent,
    Domain.STRATEGY_INTELLIGENCE: StrategyIntelligenceAgent,
    # Mastery / interview / PhD research domains
    Domain.ALGORITHMS_PROGRAMMING: AlgorithmsProgrammingAgent,
    Domain.INTERVIEW_PREP: InterviewPrepAgent,
    Domain.EE_LLM_RESEARCH: EELLMResearchAgent,
}


class AgentOrchestrator:
    """Brain of the Human Intuition Scientist system.

    Parameters
    ----------
    backend_spec:
        A provider spec string such as ``"mock"``, ``"ollama:llama3.1:8b"``,
        ``"groq:llama-3.1-8b-instant"``, etc.  Only free/open providers are
        supported.  Defaults to ``"mock"`` (offline, no model required).
    backend:
        A pre-constructed ``LLMBackend`` instance.  When supplied,
        *backend_spec* is ignored.
    use_mcp:
        Whether to enable the MCP internet-search client for agents.
    max_workers:
        Maximum thread-pool size for parallel agent calls.
    max_domains:
        Maximum number of domain agents to invoke per question.
        When ``None`` all relevant agents are used.

    .. deprecated::
        ``llm_provider`` and ``model`` keyword arguments are accepted but
        ignored; pass ``backend_spec`` or ``backend`` instead.
    """

    def __init__(
        self,
        backend_spec: str = "mock",
        backend: Optional[LLMBackend] = None,
        use_mcp: bool = True,
        max_workers: int = 7,
        max_domains: Optional[int] = None,
        # Legacy kwargs — accepted but ignored
        llm_provider: str = "mock",
        model: Optional[str] = None,
    ) -> None:
        if backend is not None:
            self._backend: LLMBackend = backend
        else:
            self._backend = get_backend(backend_spec)

        self.use_mcp = use_mcp
        self.max_workers = max_workers
        self.max_domains = max_domains

        self._mcp_client = MCPClient() if use_mcp else None
        self._weighing_system = WeighingSystem(backend=self._backend)
        self._debate_engine = DebateEngine(backend=self._backend)
        self._intuition_capture = IntuitionCapture(interactive=True)

    # ------------------------------------------------------------------
    # Primary entry point — weigh human intuition against agents
    # ------------------------------------------------------------------

    def run(
        self,
        question: str,
        *,
        prefilled_intuition: Optional[HumanIntuition] = None,
        domains: Optional[list[Domain]] = None,
    ) -> WeighingResult:
        """Run the full weighing pipeline for a single *question*.

        Parameters
        ----------
        question:
            The question to investigate.
        prefilled_intuition:
            If provided, skip the interactive intuition-capture step.
        domains:
            Explicit list of domains to query.  When ``None`` the orchestrator
            infers the most relevant domains from the question text.
        """
        intuition = self._intuition_capture.capture(
            question, prefilled=prefilled_intuition
        )
        selected_domains = domains or self._select_domains(question, intuition)
        agents = self._build_agents(selected_domains)
        responses = self._query_agents(agents, question)
        return self._weighing_system.weigh(intuition, responses)

    # ------------------------------------------------------------------
    # Debate entry point — human + tool evidence + agents
    # ------------------------------------------------------------------

    def debate(
        self,
        question: str,
        *,
        prefilled_intuition: Optional[HumanIntuition] = None,
        domains: Optional[list[Domain]] = None,
    ) -> DebateResult:
        """Run a structured multi-party debate for *question*.

        The debate pits three perspectives against each other:

        1. **Human intuition** — the user's answer and reasoning.
        2. **Tool / MCP evidence** — facts retrieved from web search.
        3. **Domain-agent reasoning** — expert analysis from LLM-backed agents.

        Agreements and divergences are surfaced explicitly in each
        :class:`DebateRound`, and a moderated verdict is produced.

        Parameters
        ----------
        question:
            The question to debate.
        prefilled_intuition:
            If provided, skip interactive capture.
        domains:
            Explicit domain list; auto-detected when ``None``.
        """
        intuition = self._intuition_capture.capture(
            question, prefilled=prefilled_intuition
        )
        selected_domains = domains or self._select_domains(question, intuition)
        agents = self._build_agents(selected_domains)
        responses = self._query_agents(agents, question)

        # Gather MCP tool evidence (empty list when MCP is disabled)
        tool_results = []
        if self._mcp_client is not None:
            try:
                tool_results = self._mcp_client.search(question, num_results=5)
            except Exception:
                tool_results = []

        return self._debate_engine.debate(intuition, responses, tool_results)

    # ------------------------------------------------------------------
    # Interview prep entry point
    # ------------------------------------------------------------------

    def interview_prep(
        self,
        question: str,
        *,
        prefilled_intuition: Optional[HumanIntuition] = None,
    ) -> InterviewResult:
        """Run a FAANG interview coaching session for *question*.

        Routes the question through three complementary agents:

        - :class:`InterviewPrepAgent` — technical correctness, pattern
          identification, optimal solution, follow-up variants.
        - :class:`AlgorithmsProgrammingAgent` — deep algorithmic and
          language-level insight (Python/Rust/Go).
        - :class:`SocialScienceAgent` — psychological readiness, stress
          management, communication coaching, STAR framing.

        Uses the existing :class:`WeighingSystem` to score the candidate's
        answer against the combined expert consensus, then packages the
        results into a structured :class:`InterviewResult`.
        """
        intuition = self._intuition_capture.capture(
            question, prefilled=prefilled_intuition
        )

        # Always use the three interview-coaching domains
        coaching_domains = [
            Domain.INTERVIEW_PREP,
            Domain.ALGORITHMS_PROGRAMMING,
            Domain.SOCIAL_SCIENCE,
        ]
        agents = self._build_agents(coaching_domains)
        responses = self._query_agents(agents, question)

        # Score the candidate's answer with the weighing system
        weighing = self._weighing_system.weigh(intuition, responses)

        # Partition responses by domain
        tech_resp = next(
            (r for r in responses if r.domain == Domain.INTERVIEW_PREP), None
        )
        algo_resp = next(
            (r for r in responses if r.domain == Domain.ALGORITHMS_PROGRAMMING), None
        )
        social_resp = next(
            (r for r in responses if r.domain == Domain.SOCIAL_SCIENCE), None
        )

        return InterviewResult(
            question=question,
            candidate_answer=intuition.intuitive_answer,
            candidate_confidence=intuition.confidence,
            technical_score=weighing.intuition_accuracy_pct / 100.0,
            technical_feedback=tech_resp.answer if tech_resp else "",
            technical_reasoning=tech_resp.reasoning if tech_resp else "",
            algorithmic_insight=algo_resp.answer if algo_resp else "",
            mental_preparation=social_resp.answer if social_resp else "",
            overall_analysis=weighing.overall_analysis,
            synthesized_answer=weighing.synthesized_answer,
            recommendations=weighing.recommendations,
        )

    # ------------------------------------------------------------------
    # Model cycling / evaluation entry point
    # ------------------------------------------------------------------

    def evaluate_models(
        self,
        question: str,
        model_specs: list[str],
        *,
        prefilled_intuition: Optional[HumanIntuition] = None,
        domains: Optional[list[Domain]] = None,
    ) -> ModelEvaluationResult:
        """Cycle through *model_specs* and evaluate each against *question*.

        For every model backend the full dual-pipeline runs:
        human intuition → domain agents (intuition + MCP) → WeighingSystem.

        Models that are unavailable (server down, missing API key, etc.) are
        recorded as unavailable and skipped gracefully — they never cause the
        whole evaluation to fail.

        Parameters
        ----------
        question:
            The question to investigate across all models.
        model_specs:
            List of provider spec strings, e.g.
            ``["mock", "ollama:llama3.1:8b", "groq:llama-3.1-8b-instant"]``.
        prefilled_intuition:
            Human intuition to use for all runs.  When ``None`` the user is
            prompted interactively on the **first** run; subsequent runs reuse
            the same intuition so the comparison is fair.
        domains:
            Explicit domain list; auto-detected when ``None``.

        Returns
        -------
        ModelEvaluationResult
            Per-model results plus cross-model consensus and divergence.
        """
        import time

        # Capture intuition once (interactively if needed) so all model runs
        # see the same human input and results are comparable.
        intuition = self._intuition_capture.capture(
            question, prefilled=prefilled_intuition
        )
        selected_domains = domains or self._select_domains(question, intuition)

        run_results: list[ModelRunResult] = []

        for spec in model_specs:
            t0 = time.monotonic()
            try:
                candidate_backend = get_backend(spec)
            except ValueError as exc:
                run_results.append(ModelRunResult(
                    model_spec=spec,
                    backend_available=False,
                    weighing_result=None,
                    error=f"Invalid spec: {exc}",
                ))
                continue

            try:
                # Spin up a temporary orchestrator with this backend
                tmp_orch = AgentOrchestrator(
                    backend=candidate_backend,
                    use_mcp=self.use_mcp,
                    max_domains=self.max_domains,
                )
                result = tmp_orch.run(
                    question,
                    prefilled_intuition=intuition,
                    domains=selected_domains,
                )
                tmp_orch.close()
                elapsed = round(time.monotonic() - t0, 2)
                run_results.append(ModelRunResult(
                    model_spec=spec,
                    backend_available=True,
                    weighing_result=result,
                    duration_seconds=elapsed,
                ))
            except Exception as exc:
                elapsed = round(time.monotonic() - t0, 2)
                run_results.append(ModelRunResult(
                    model_spec=spec,
                    backend_available=False,
                    weighing_result=None,
                    error=str(exc),
                    duration_seconds=elapsed,
                ))

        return self._build_evaluation_result(question, run_results)

    # ------------------------------------------------------------------
    # Model evaluation helpers
    # ------------------------------------------------------------------

    def _build_evaluation_result(
        self,
        question: str,
        run_results: list[ModelRunResult],
    ) -> ModelEvaluationResult:
        available = [r for r in run_results if r.backend_available and r.weighing_result]
        models_available = len(available)
        models_evaluated = len(run_results)

        if not available:
            return ModelEvaluationResult(
                question=question,
                model_results=run_results,
                consensus_answer="No models were available to evaluate.",
                divergence_summary="All models unavailable.",
                best_model_spec="none",
                models_evaluated=models_evaluated,
                models_available=0,
                mean_intuition_accuracy_pct=0.0,
            )

        # Best model = highest intuition accuracy
        best = max(available, key=lambda r: r.weighing_result.intuition_accuracy_pct)  # type: ignore[union-attr]
        mean_acc = round(
            sum(r.weighing_result.intuition_accuracy_pct for r in available) / models_available,  # type: ignore[union-attr]
            1,
        )

        # Consensus: synthesized answers from all models → find common tokens
        answers = [r.weighing_result.synthesized_answer for r in available]  # type: ignore[union-attr]
        consensus = self._compute_consensus(answers)

        # Divergence: models whose accuracy is > 20 pts from the mean
        outliers = [
            f"{r.model_spec} ({r.weighing_result.intuition_accuracy_pct:.1f}%)"  # type: ignore[union-attr]
            for r in available
            if abs(r.weighing_result.intuition_accuracy_pct - mean_acc) > 20  # type: ignore[union-attr]
        ]
        divergence = (
            f"Outlier models: {', '.join(outliers)}" if outliers
            else f"All {models_available} models were within ±20% of mean accuracy ({mean_acc:.1f}%)."
        )

        return ModelEvaluationResult(
            question=question,
            model_results=run_results,
            consensus_answer=consensus,
            divergence_summary=divergence,
            best_model_spec=best.model_spec,
            models_evaluated=models_evaluated,
            models_available=models_available,
            mean_intuition_accuracy_pct=mean_acc,
        )

    @staticmethod
    def _compute_consensus(answers: list[str]) -> str:
        """Return the answer that contains the most tokens shared by all answers."""
        if not answers:
            return ""
        if len(answers) == 1:
            return answers[0]
        # Score each answer by how many of its tokens appear in ALL others
        import re

        def tokens(text: str) -> set[str]:
            return set(re.findall(r"[a-z]{4,}", text.lower()))

        all_token_sets = [tokens(a) for a in answers]
        universal = set.intersection(*all_token_sets)
        scored = sorted(
            answers,
            key=lambda a: len(tokens(a) & universal),
            reverse=True,
        )
        return scored[0]

    # ------------------------------------------------------------------
    # Domain selection
    # ------------------------------------------------------------------

    def _select_domains(
        self, question: str, intuition: HumanIntuition
    ) -> list[Domain]:
        """Return the most relevant domains for this question."""
        combined = f"{question} {intuition.intuitive_answer} {intuition.reasoning}"
        inferred = IntuitionCapture.infer_domains(combined)

        # Always include at least 3 domains for breadth
        if len(inferred) < 3:
            for d in Domain:
                if d not in inferred:
                    inferred.append(d)
                if len(inferred) == 3:
                    break

        if self.max_domains:
            inferred = inferred[: self.max_domains]

        return inferred

    # ------------------------------------------------------------------
    # Agent building and querying
    # ------------------------------------------------------------------

    def _build_agents(self, domains: list[Domain]) -> list[BaseAgent]:
        agents: list[BaseAgent] = []
        for domain in domains:
            cls = _AGENT_CLASSES.get(domain)
            if cls is None:
                continue
            agent = cls(
                mcp_client=self._mcp_client,
                backend=self._backend,
            )
            agents.append(agent)
        return agents

    def _query_agents(
        self, agents: list[BaseAgent], question: str
    ) -> list[AgentResponse]:
        """Query all agents in parallel; collect results preserving order."""
        if not agents:
            return []

        responses: dict[int, AgentResponse] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(agent.answer, question): idx
                for idx, agent in enumerate(agents)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    responses[idx] = future.result()
                except Exception as exc:
                    responses[idx] = AgentResponse(
                        domain=agents[idx].domain,
                        answer=f"Agent error: {exc}",
                        reasoning="",
                        confidence=0.1,
                    )

        return [responses[i] for i in sorted(responses)]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release resources (MCP HTTP client, etc.)."""
        if self._mcp_client is not None:
            self._mcp_client.close()

    def __enter__(self) -> "AgentOrchestrator":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


