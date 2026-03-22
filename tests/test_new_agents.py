"""Tests for the new domain agents: SignalProcessingAgent, ExperimentRunnerAgent,
and the enhanced PhysicsAgent iterative-problem protocol.

All tests run in offline/mock mode — no LLM key or network required.
"""

import pytest

from src.agents.experiment_runner_agent import ExperimentRunnerAgent
from src.agents.physics_agent import PhysicsAgent
from src.agents.signal_processing_agent import SignalProcessingAgent
from src.models import AgentResponse, Domain
from src.orchestrator.agent_orchestrator import _AGENT_CLASSES


# ---------------------------------------------------------------------------
# Domain registration
# ---------------------------------------------------------------------------


class TestNewAgentRegistration:
    """Verify both new agents are registered in the orchestrator registry."""

    def test_signal_processing_registered(self):
        assert Domain.SIGNAL_PROCESSING in _AGENT_CLASSES
        assert _AGENT_CLASSES[Domain.SIGNAL_PROCESSING] is SignalProcessingAgent

    def test_experiment_runner_registered(self):
        assert Domain.EXPERIMENT_RUNNER in _AGENT_CLASSES
        assert _AGENT_CLASSES[Domain.EXPERIMENT_RUNNER] is ExperimentRunnerAgent


# ---------------------------------------------------------------------------
# Domain attribute
# ---------------------------------------------------------------------------


class TestNewAgentDomains:
    def test_signal_processing_domain(self):
        assert SignalProcessingAgent.domain == Domain.SIGNAL_PROCESSING

    def test_experiment_runner_domain(self):
        assert ExperimentRunnerAgent.domain == Domain.EXPERIMENT_RUNNER

    def test_physics_domain_unchanged(self):
        assert PhysicsAgent.domain == Domain.PHYSICS


# ---------------------------------------------------------------------------
# System prompt content
# ---------------------------------------------------------------------------


class TestSignalProcessingAgentPrompt:
    def setup_method(self):
        self.agent = SignalProcessingAgent(mcp_client=None)

    def test_prompt_not_empty(self):
        prompt = self.agent._build_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_prompt_contains_iterative_protocol(self):
        prompt = self.agent._build_system_prompt()
        assert "Iterative Problem-Selection Protocol" in prompt

    def test_prompt_contains_checkpoints(self):
        prompt = self.agent._build_system_prompt()
        assert "Checkpoint" in prompt

    def test_prompt_contains_intuition_elicitation(self):
        prompt = self.agent._build_system_prompt()
        assert "intuition" in prompt.lower()

    def test_problem_catalog_has_ten_entries(self):
        assert len(SignalProcessingAgent.ITERATIVE_PROBLEM_TYPES) == 10

    def test_problem_types_are_nonempty_tuples(self):
        for name, desc in SignalProcessingAgent.ITERATIVE_PROBLEM_TYPES:
            assert isinstance(name, str) and len(name) > 0
            assert isinstance(desc, str) and len(desc) > 0


class TestPhysicsAgentPrompt:
    def setup_method(self):
        self.agent = PhysicsAgent(mcp_client=None)

    def test_prompt_contains_iterative_protocol(self):
        prompt = self.agent._build_system_prompt()
        assert "Iterative Problem-Selection Protocol" in prompt

    def test_prompt_contains_checkpoints(self):
        prompt = self.agent._build_system_prompt()
        assert "Checkpoint" in prompt

    def test_prompt_contains_intuition_elicitation(self):
        prompt = self.agent._build_system_prompt()
        assert "intuition" in prompt.lower()

    def test_problem_catalog_has_ten_entries(self):
        assert len(PhysicsAgent.ITERATIVE_PROBLEM_TYPES) == 10

    def test_problem_types_are_nonempty_tuples(self):
        for name, desc in PhysicsAgent.ITERATIVE_PROBLEM_TYPES:
            assert isinstance(name, str) and len(name) > 0
            assert isinstance(desc, str) and len(desc) > 0


class TestExperimentRunnerAgentPrompt:
    def setup_method(self):
        self.agent = ExperimentRunnerAgent(mcp_client=None)

    def test_prompt_not_empty(self):
        prompt = self.agent._build_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_prompt_contains_phases(self):
        prompt = self.agent._build_system_prompt()
        assert "Phase 1" in prompt
        assert "Phase 2" in prompt
        assert "Phase 3" in prompt

    def test_prompt_contains_hypothesis(self):
        prompt = self.agent._build_system_prompt()
        assert "hypothesis" in prompt.lower() or "hypothes" in prompt.lower()

    def test_prompt_contains_snippet_instruction(self):
        prompt = self.agent._build_system_prompt()
        assert "snippet" in prompt.lower() or "python" in prompt.lower()

    def test_prompt_elicits_intuition(self):
        prompt = self.agent._build_system_prompt()
        assert "intuition" in prompt.lower()

    def test_experiment_catalog_has_eight_entries(self):
        assert len(ExperimentRunnerAgent.EXPERIMENT_TYPES) == 8

    def test_experiment_types_are_nonempty_tuples(self):
        for name, desc in ExperimentRunnerAgent.EXPERIMENT_TYPES:
            assert isinstance(name, str) and len(name) > 0
            assert isinstance(desc, str) and len(desc) > 0

    def test_prompt_contains_experiment_catalog(self):
        """Verify the experiment catalog is embedded in the system prompt."""
        agent = ExperimentRunnerAgent(mcp_client=None)
        prompt = agent._build_system_prompt()
        for name, _ in ExperimentRunnerAgent.EXPERIMENT_TYPES:
            assert name in prompt, f"Experiment type '{name}' missing from prompt"


# ---------------------------------------------------------------------------
# Answer / response structure (mock mode)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [SignalProcessingAgent, ExperimentRunnerAgent, PhysicsAgent],
)
class TestNewAgentMockAnswers:
    def test_answer_returns_agent_response(self, cls):
        agent = cls(mcp_client=None)
        resp = agent.answer("What is the Fourier transform?")
        assert isinstance(resp, AgentResponse)
        assert resp.domain == cls.domain
        assert isinstance(resp.answer, str)
        assert len(resp.answer) > 0
        assert 0.0 <= resp.confidence <= 1.0

    def test_answer_sources_is_list(self, cls):
        agent = cls(mcp_client=None)
        resp = agent.answer("Explain gradient descent")
        assert isinstance(resp.sources, list)

    def test_weights_sum_to_one(self, cls):
        agent = cls(mcp_client=None)
        resp = agent.answer("Design an experiment")
        assert abs(resp.intuition_weight + resp.tool_weight - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Weight bias: new agents should be intuition-heavy (no MCP)
# ---------------------------------------------------------------------------


class TestNewAgentWeightBias:
    """Without MCP results, signal processing and experiment runner agents
    should favour the intuition pipeline (weight > 0.5)."""

    def test_signal_processing_intuition_heavy(self):
        agent = SignalProcessingAgent(mcp_client=None)
        resp = agent.answer("Why does the Fourier transform decompose signals into sinusoids?")
        assert resp.intuition_weight > 0.5, (
            f"Expected intuition_weight > 0.5, got {resp.intuition_weight}"
        )

    def test_experiment_runner_intuition_heavy(self):
        agent = ExperimentRunnerAgent(mcp_client=None)
        resp = agent.answer("Why does adding momentum to gradient descent speed up convergence?")
        assert resp.intuition_weight > 0.5, (
            f"Expected intuition_weight > 0.5, got {resp.intuition_weight}"
        )

    def test_physics_intuition_heavy(self):
        agent = PhysicsAgent(mcp_client=None)
        resp = agent.answer("Why does entropy always increase in isolated systems?")
        assert resp.intuition_weight > 0.5, (
            f"Expected intuition_weight > 0.5, got {resp.intuition_weight}"
        )


# ---------------------------------------------------------------------------
# Domain routing: IntuitionCapture should pick up new domains by keyword
# ---------------------------------------------------------------------------


class TestDomainRoutingKeywords:
    def test_signal_keywords_route_to_signal_processing(self):
        from src.intuition.human_intuition import IntuitionCapture

        domains = IntuitionCapture.infer_domains("design an FIR filter with Fourier transform")
        assert Domain.SIGNAL_PROCESSING in domains

    def test_experiment_keywords_route_to_experiment_runner(self):
        from src.intuition.human_intuition import IntuitionCapture

        domains = IntuitionCapture.infer_domains(
            "I want to run a simulation to test my hypothesis"
        )
        assert Domain.EXPERIMENT_RUNNER in domains
