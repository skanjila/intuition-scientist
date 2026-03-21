"""Tests for the AgentOrchestrator."""

import pytest
from unittest.mock import MagicMock, patch
from src.models import AgentResponse, Domain, HumanIntuition, WeighingResult
from src.orchestrator.agent_orchestrator import AgentOrchestrator, _AGENT_CLASSES


class TestAgentOrchestratorInit:
    def test_all_domains_registered(self):
        """Every Domain enum value must have a registered agent class."""
        for domain in Domain:
            assert domain in _AGENT_CLASSES, f"{domain} missing from registry"

    def test_init_no_mcp(self):
        orch = AgentOrchestrator(use_mcp=False)
        assert orch._mcp_client is None
        orch.close()


class TestDomainSelection:
    def test_physics_question_selects_physics(self):
        orch = AgentOrchestrator(use_mcp=False)
        hi = HumanIntuition(
            question="How does quantum tunnelling work?",
            intuitive_answer="Particles pass through barriers using wave functions.",
            confidence=0.6,
        )
        domains = orch._select_domains("How does quantum tunnelling work?", hi)
        assert Domain.PHYSICS in domains
        orch.close()

    def test_minimum_three_domains(self):
        orch = AgentOrchestrator(use_mcp=False)
        hi = HumanIntuition(
            question="zyxwvutsrqponm",  # gibberish → no keyword hits
            intuitive_answer="I have no idea",
            confidence=0.1,
        )
        domains = orch._select_domains("zyxwvutsrqponm", hi)
        assert len(domains) >= 3
        orch.close()

    def test_max_domains_respected(self):
        orch = AgentOrchestrator(use_mcp=False, max_domains=2)
        hi = HumanIntuition(
            question="How does a neural network learn?",
            intuitive_answer="Gradient descent.",
            confidence=0.8,
        )
        domains = orch._select_domains("How does a neural network learn?", hi)
        assert len(domains) <= 2
        orch.close()


class TestOrchestatorRunOffline:
    """Test the full orchestrator pipeline without an LLM (mock mode)."""

    def test_run_returns_weighing_result(self):
        intuition = HumanIntuition(
            question="What is energy?",
            intuitive_answer="The capacity to do work.",
            confidence=0.7,
        )
        orch = AgentOrchestrator(use_mcp=False, max_domains=2)
        # Force all agent LLM clients to None (offline mode)
        result = orch.run(
            "What is energy?",
            prefilled_intuition=intuition,
            domains=[Domain.PHYSICS, Domain.ELECTRICAL_ENGINEERING],
        )
        assert isinstance(result, WeighingResult)
        assert result.question == "What is energy?"
        assert len(result.agent_responses) == 2
        assert 0.0 <= result.intuition_accuracy_pct <= 100.0
        orch.close()

    def test_run_with_all_domains(self):
        intuition = HumanIntuition(
            question="What is a neural network?",
            intuitive_answer="A system of nodes inspired by the brain.",
            confidence=0.8,
        )
        orch = AgentOrchestrator(use_mcp=False)
        result = orch.run(
            "What is a neural network?",
            prefilled_intuition=intuition,
        )
        assert isinstance(result, WeighingResult)
        assert len(result.agent_responses) >= 3
        orch.close()
