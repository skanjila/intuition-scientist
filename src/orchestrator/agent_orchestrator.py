"""Agent Orchestrator — the brain of the Human Intuition Scientist.

The orchestrator:
1. Receives a question and a human intuition.
2. Determines which domain agents are most relevant to the question.
3. Spins up those agents (or all of them if the question is broad).
4. Collects their responses in parallel (using threads).
5. Passes everything to the ``WeighingSystem`` for deep analysis.
6. Returns a ``WeighingResult`` to the caller.
"""

from __future__ import annotations

import os
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
from src.analysis.weighing_system import WeighingSystem
from src.intuition.human_intuition import IntuitionCapture
from src.mcp.mcp_client import MCPClient
from src.models import AgentResponse, Domain, HumanIntuition, WeighingResult


# ---------------------------------------------------------------------------
# Registry of all domain agents
# ---------------------------------------------------------------------------

_AGENT_CLASSES: dict[Domain, type[BaseAgent]] = {
    Domain.ELECTRICAL_ENGINEERING: ElectricalEngineeringAgent,
    Domain.COMPUTER_SCIENCE: ComputerScienceAgent,
    Domain.NEURAL_NETWORKS: NeuralNetworksAgent,
    Domain.SOCIAL_SCIENCE: SocialScienceAgent,
    Domain.SPACE_SCIENCE: SpaceScienceAgent,
    Domain.PHYSICS: PhysicsAgent,
    Domain.DEEP_LEARNING: DeepLearningAgent,
}


class AgentOrchestrator:
    """Brain of the Human Intuition Scientist system.

    Parameters
    ----------
    llm_provider:
        ``"anthropic"`` or ``"openai"``.  The same provider is used for all
        domain agents and the weighing system.
    model:
        Explicit model name (e.g. ``"claude-3-haiku-20240307"``).
        Defaults to the provider's recommended fast model.
    use_mcp:
        Whether to enable the MCP internet-search client for agents.
    max_workers:
        Maximum thread-pool size for parallel agent calls.
    max_domains:
        Maximum number of domain agents to invoke per question.
        When ``None`` all relevant agents are used.
    """

    def __init__(
        self,
        llm_provider: str = "anthropic",
        model: Optional[str] = None,
        use_mcp: bool = True,
        max_workers: int = 7,
        max_domains: Optional[int] = None,
    ) -> None:
        self.llm_provider = llm_provider
        self.model = model
        self.use_mcp = use_mcp
        self.max_workers = max_workers
        self.max_domains = max_domains

        self._mcp_client = MCPClient() if use_mcp else None
        self._llm_client = self._init_llm_client()
        self._weighing_system = WeighingSystem(
            llm_client=self._llm_client,
            llm_provider=llm_provider,
            model=model,
        )
        self._intuition_capture = IntuitionCapture(interactive=True)

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def run(
        self,
        question: str,
        *,
        prefilled_intuition: Optional[HumanIntuition] = None,
        domains: Optional[list[Domain]] = None,
    ) -> WeighingResult:
        """Run the full pipeline for a single *question*.

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
        # Step 1 – capture human intuition
        intuition = self._intuition_capture.capture(
            question, prefilled=prefilled_intuition
        )

        # Step 2 – select domains
        selected_domains = domains or self._select_domains(question, intuition)

        # Step 3 – build agents
        agents = self._build_agents(selected_domains)

        # Step 4 – query agents in parallel
        responses = self._query_agents(agents, question)

        # Step 5 – weigh intuition against agent responses
        result = self._weighing_system.weigh(intuition, responses)

        return result

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
            # Add remaining domains that were not already selected
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
                llm_provider=self.llm_provider,
                model=self.model,
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
                    # Build a fallback response so the pipeline never crashes
                    responses[idx] = AgentResponse(
                        domain=agents[idx].domain,
                        answer=f"Agent error: {exc}",
                        reasoning="",
                        confidence=0.1,
                    )

        return [responses[i] for i in sorted(responses)]

    # ------------------------------------------------------------------
    # LLM client initialisation
    # ------------------------------------------------------------------

    def _init_llm_client(self) -> Optional[object]:
        if self.llm_provider == "anthropic":
            try:
                import anthropic  # type: ignore

                return anthropic.Anthropic(
                    api_key=os.environ.get("ANTHROPIC_API_KEY", "")
                )
            except ImportError:
                return None
        else:
            try:
                import openai  # type: ignore

                return openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
            except ImportError:
                return None

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
