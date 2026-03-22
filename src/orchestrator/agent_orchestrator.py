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
# Signal processing and experiment runner agents
from src.agents.signal_processing_agent import SignalProcessingAgent
from src.agents.experiment_runner_agent import ExperimentRunnerAgent
from src.analysis.debate_engine import DebateEngine
from src.analysis.weighing_system import WeighingSystem
from src.intuition.human_intuition import IntuitionCapture, generate_auto_intuition
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
    # Signal processing and experiment runner
    Domain.SIGNAL_PROCESSING: SignalProcessingAgent,
    Domain.EXPERIMENT_RUNNER: ExperimentRunnerAgent,
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
    agent_max_tokens:
        Token budget for each per-agent LLM call (default: 1024).
        Lower values (e.g. 256) reduce latency for local backends.
    synthesis_max_tokens:
        Token budget for synthesis and full-analysis LLM calls
        (default: 512).  Lower values (e.g. 384) reduce latency.
    auto_intuition:
        When ``True``, skip interactive prompting and auto-generate a
        lightweight "human intuition" response using keyword heuristics and
        an optional short LLM call.  The default (``False``) preserves the
        original interactive behaviour.
    adaptive_agents:
        When ``True``, use an evolving adaptive loop that starts with a small
        set of domain agents and expands only if confidence or coverage is
        deemed insufficient.  The default (``False``) queries a fixed set of
        inferred domains.  See :meth:`_adaptive_select_and_run` for details.
    target_latency_ms:
        Optional wall-clock budget (milliseconds) for the adaptive loop.
        When the budget is exceeded the loop stops expanding even if the
        confidence threshold has not been reached.  Ignored when
        ``adaptive_agents`` is ``False``.

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
        agent_max_tokens: int = 1024,
        synthesis_max_tokens: int = 512,
        auto_intuition: bool = False,
        adaptive_agents: bool = False,
        target_latency_ms: Optional[int] = None,
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
        self._agent_max_tokens = agent_max_tokens
        self.auto_intuition = auto_intuition
        self.adaptive_agents = adaptive_agents
        self.target_latency_ms = target_latency_ms

        self._mcp_client = MCPClient() if use_mcp else None
        self._weighing_system = WeighingSystem(
            backend=self._backend,
            synthesis_max_tokens=synthesis_max_tokens,
        )
        self._debate_engine = DebateEngine(backend=self._backend)
        # Interactive capture is only used when auto_intuition is False
        self._intuition_capture = IntuitionCapture(interactive=not auto_intuition)

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
            If provided, skip the interactive (or auto-generated) intuition
            step entirely and use this value directly.
        domains:
            Explicit list of domains to query.  When ``None`` the orchestrator
            infers the most relevant domains from the question text.  Note:
            when ``adaptive_agents=True`` and *domains* is ``None``, the
            adaptive loop takes over domain selection and expansion; an
            explicit *domains* list disables the adaptive loop.
        """
        # ------------------------------------------------------------------
        # Step 1 — Capture or generate human intuition
        # ------------------------------------------------------------------
        if prefilled_intuition is not None:
            intuition = prefilled_intuition
        elif self.auto_intuition:
            # Non-interactive: synthesise a lightweight "human" perspective
            # using keyword heuristics + optional short LLM quick-think pass.
            intuition = generate_auto_intuition(question, backend=self._backend)
        else:
            intuition = self._intuition_capture.capture(question)

        # ------------------------------------------------------------------
        # Step 2 — Select domains and query agents
        # ------------------------------------------------------------------
        if self.adaptive_agents and domains is None:
            # Use the evolving adaptive loop: start small, expand if needed.
            responses = self._adaptive_select_and_run(question, intuition)
        else:
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
        intuition = self._resolve_intuition(question, prefilled_intuition)
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
        intuition = self._resolve_intuition(question, prefilled_intuition)

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

        # Capture intuition once (interactively or via auto-generation if
        # auto_intuition=True) so all model runs see the same human input
        # and results are comparable.
        intuition = self._resolve_intuition(question, prefilled_intuition)
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
    # Intuition resolution helper
    # ------------------------------------------------------------------

    def _resolve_intuition(
        self,
        question: str,
        prefilled: Optional[HumanIntuition],
    ) -> HumanIntuition:
        """Return the intuition to use for *question*.

        Priority order
        --------------
        1. *prefilled* — caller-supplied value; used as-is.
        2. ``auto_intuition=True`` — generate a lightweight auto-intuition.
        3. Default interactive capture via ``_intuition_capture``.
        """
        if prefilled is not None:
            return prefilled
        if self.auto_intuition:
            return generate_auto_intuition(question, backend=self._backend)
        return self._intuition_capture.capture(question)

    # ------------------------------------------------------------------
    # Adaptive agent selection loop
    # ------------------------------------------------------------------

    def _adaptive_select_and_run(
        self,
        question: str,
        intuition: HumanIntuition,
    ) -> list["AgentResponse"]:
        """Evolving adaptive loop for intelligent agent-count selection.

        Strategy
        --------
        The loop addresses a fundamental trade-off: querying all available
        domain agents is thorough but slow; querying too few may miss critical
        perspectives.  The adaptive loop resolves this by starting small and
        expanding *only when necessary*.

        Algorithm
        ~~~~~~~~~
        1. **Rank candidates**: Score all domains by keyword relevance to the
           question + intuition text.  Ensure a minimum of
           ``_ADAPTIVE_INITIAL`` (3) candidates.
        2. **Initial batch**: Query the top ``_ADAPTIVE_INITIAL`` agents in
           parallel (same thread-pool as the non-adaptive path).
        3. **Evaluate coverage**: Compute the mean confidence across all
           collected responses.  High mean confidence (≥ ``_ADAPTIVE_CONF_THRESHOLD``
           = 0.65) signals that the current set of agents reached a coherent
           answer and no further expansion is needed.
        4. **Expand if needed**: If coverage is insufficient *and* stopping
           criteria are not met, append the next ``_ADAPTIVE_STEP`` (2)
           highest-ranked candidate domains and query *only those new agents*
           (already-queried agents are never re-queried — purely incremental).
        5. **Stopping criteria** (any one is sufficient to halt expansion):
           - Mean agent confidence ≥ ``_ADAPTIVE_CONF_THRESHOLD``.
           - No remaining candidate domains.
           - ``max_domains`` ceiling reached.
           - Wall-clock budget exceeded (``target_latency_ms`` set and elapsed
             time ≥ that value).

        Why this is safe and predictable
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        - The candidate list is computed once, deterministically, from keyword
          matching — the same heuristic used in the non-adaptive path.
        - The loop terminates in at most ``ceil(N / _ADAPTIVE_STEP)`` rounds
          where N is the number of candidate domains (bounded by ``max_domains``
          when set, or the total number of registered domains otherwise).
        - Every expansion step is logged at INFO level so behaviour is fully
          observable without stepping through source code.
        - The incremental design means a partial result (from the initial batch)
          is always available even if the loop is interrupted by a time budget.

        Parameters
        ----------
        question:
            The question under investigation.
        intuition:
            The (possibly auto-generated) human intuition for the question.
        """
        import logging
        import time

        logger = logging.getLogger(__name__)

        # --- Tuning constants (documented here for auditability) ---
        # Confidence threshold: mean agent confidence at or above this value
        # indicates the current agent set reached a sufficiently coherent answer.
        _ADAPTIVE_CONF_THRESHOLD: float = 0.65
        # Number of new domains to add per expansion round.
        _ADAPTIVE_STEP: int = 2
        # Minimum number of domains in the initial batch.
        _ADAPTIVE_INITIAL: int = 3

        # Build the full ranked candidate list from keyword relevance.
        # IntuitionCapture.infer_domains() is a pure static keyword-scoring
        # utility that happens to live on IntuitionCapture for historical
        # reasons; it has no dependency on interactive user I/O.
        combined = f"{question} {intuition.intuitive_answer} {intuition.reasoning}"
        all_candidates: list[Domain] = IntuitionCapture.infer_domains(combined)

        # Guarantee at least _ADAPTIVE_INITIAL candidates
        if len(all_candidates) < _ADAPTIVE_INITIAL:
            for d in Domain:
                if d not in all_candidates:
                    all_candidates.append(d)
                if len(all_candidates) >= _ADAPTIVE_INITIAL:
                    break

        # Apply max_domains ceiling to prevent runaway expansion
        if self.max_domains:
            all_candidates = all_candidates[: self.max_domains]

        # Partition into active initial set and remaining candidates
        active_domains: list[Domain] = all_candidates[:_ADAPTIVE_INITIAL]
        remaining_candidates: list[Domain] = list(all_candidates[_ADAPTIVE_INITIAL:])

        all_responses: list["AgentResponse"] = []
        queried_domains: set[Domain] = set()
        start_time = time.monotonic()
        round_num = 0

        while True:
            round_num += 1
            new_domains = [d for d in active_domains if d not in queried_domains]

            if new_domains:
                logger.info(
                    "[adaptive] Round %d: querying %d new agent(s): %s",
                    round_num,
                    len(new_domains),
                    [d.value for d in new_domains],
                )
                new_agents = self._build_agents(new_domains)
                new_responses = self._query_agents(new_agents, question)
                all_responses.extend(new_responses)
                queried_domains.update(new_domains)

            # Compute mean confidence across all collected responses
            mean_conf = (
                sum(r.confidence for r in all_responses) / len(all_responses)
                if all_responses
                else 0.0
            )

            elapsed_ms = (time.monotonic() - start_time) * 1000
            time_budget_exceeded = (
                self.target_latency_ms is not None
                and elapsed_ms >= self.target_latency_ms
            )

            logger.info(
                "[adaptive] Round %d complete: %d domains queried, "
                "mean_conf=%.2f, elapsed=%.0f ms",
                round_num,
                len(queried_domains),
                mean_conf,
                elapsed_ms,
            )

            # --- Stopping criteria ---
            if mean_conf >= _ADAPTIVE_CONF_THRESHOLD:
                logger.info(
                    "[adaptive] Stopping: mean confidence %.2f >= threshold %.2f",
                    mean_conf,
                    _ADAPTIVE_CONF_THRESHOLD,
                )
                break

            if not remaining_candidates:
                logger.info("[adaptive] Stopping: no remaining candidate domains.")
                break

            if time_budget_exceeded:
                logger.info(
                    "[adaptive] Stopping: time budget %d ms exceeded (elapsed %.0f ms).",
                    self.target_latency_ms,
                    elapsed_ms,
                )
                break

            # --- Expand: add next batch of domains ---
            next_batch = remaining_candidates[:_ADAPTIVE_STEP]
            remaining_candidates = remaining_candidates[_ADAPTIVE_STEP:]
            active_domains = active_domains + next_batch
            logger.info(
                "[adaptive] Expanding: adding domains %s "
                "(mean_conf=%.2f < threshold=%.2f)",
                [d.value for d in next_batch],
                mean_conf,
                _ADAPTIVE_CONF_THRESHOLD,
            )

        return all_responses

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
                max_tokens=self._agent_max_tokens,
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
        """Release resources (MCP HTTP client, backend connections, etc.)."""
        if self._mcp_client is not None:
            self._mcp_client.close()
        if callable(getattr(self._backend, "close", None)):
            self._backend.close()  # type: ignore[union-attr]

    def __enter__(self) -> "AgentOrchestrator":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


