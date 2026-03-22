"""Tests for the AgentOrchestrator."""

import time
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

    def test_init_stores_agent_timeout_seconds(self):
        """agent_timeout_seconds is stored on the orchestrator."""
        orch = AgentOrchestrator(use_mcp=False, agent_timeout_seconds=42.0)
        assert orch.agent_timeout_seconds == 42.0
        orch.close()

    def test_init_default_agent_timeout_seconds(self):
        """Default agent_timeout_seconds is 30.0."""
        orch = AgentOrchestrator(use_mcp=False)
        assert orch.agent_timeout_seconds == 30.0
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


class TestAgentTimeout:
    """Ensure _query_agents returns promptly when an agent exceeds the timeout."""

    def _make_slow_agent(self, domain: Domain, sleep_seconds: float) -> MagicMock:
        """Return a mock agent whose answer() sleeps for *sleep_seconds*."""
        agent = MagicMock()
        agent.domain = domain

        def slow_answer(_question: str) -> AgentResponse:
            time.sleep(sleep_seconds)
            return AgentResponse(
                domain=domain,
                answer="slow answer",
                reasoning="",
                confidence=0.9,
            )

        agent.answer.side_effect = slow_answer
        return agent

    def test_timeout_returns_low_confidence_response(self):
        """An agent that sleeps longer than the timeout must NOT block the run.

        The orchestrator should return a low-confidence AgentResponse with a
        timeout message rather than waiting indefinitely.
        """
        orch = AgentOrchestrator(use_mcp=False, agent_timeout_seconds=0.5)
        slow_agent = self._make_slow_agent(Domain.PHYSICS, sleep_seconds=5.0)

        t0 = time.monotonic()
        responses = orch._query_agents([slow_agent], "ping")
        elapsed = time.monotonic() - t0

        # Must complete well before the agent's natural sleep time
        assert elapsed < 4.0, f"_query_agents took {elapsed:.2f}s — likely hanging"

        assert len(responses) == 1
        resp = responses[0]
        assert resp.domain == Domain.PHYSICS
        assert resp.confidence == 0.1
        assert "timed out" in resp.answer.lower()

        orch.close()

    def test_timeout_does_not_affect_fast_agents(self):
        """Agents that finish before the timeout return their real response."""
        orch = AgentOrchestrator(use_mcp=False, agent_timeout_seconds=5.0)

        fast_agent = MagicMock()
        fast_agent.domain = Domain.COMPUTER_SCIENCE
        fast_agent.answer.return_value = AgentResponse(
            domain=Domain.COMPUTER_SCIENCE,
            answer="fast answer",
            reasoning="no delay",
            confidence=0.85,
        )

        responses = orch._query_agents([fast_agent], "ping")

        assert len(responses) == 1
        assert responses[0].confidence == 0.85
        assert responses[0].answer == "fast answer"

        orch.close()

    def test_mixed_agents_timeout_only_slow_ones(self):
        """Only timed-out agents get the placeholder; fast agents return normally."""
        orch = AgentOrchestrator(
            use_mcp=False,
            agent_timeout_seconds=0.5,
            max_workers=2,
        )

        fast_agent = MagicMock()
        fast_agent.domain = Domain.COMPUTER_SCIENCE
        fast_agent.answer.return_value = AgentResponse(
            domain=Domain.COMPUTER_SCIENCE,
            answer="fast answer",
            reasoning="",
            confidence=0.8,
        )

        slow_agent = self._make_slow_agent(Domain.PHYSICS, sleep_seconds=5.0)

        t0 = time.monotonic()
        responses = orch._query_agents([fast_agent, slow_agent], "ping")
        elapsed = time.monotonic() - t0

        assert elapsed < 4.0, f"_query_agents took {elapsed:.2f}s — likely hanging"
        assert len(responses) == 2

        # Responses are returned in submission order
        cs_resp = next(r for r in responses if r.domain == Domain.COMPUTER_SCIENCE)
        phys_resp = next(r for r in responses if r.domain == Domain.PHYSICS)

        assert cs_resp.confidence == 0.8
        assert phys_resp.confidence == 0.1
        assert "timed out" in phys_resp.answer.lower()

        orch.close()


class TestCLISmoke:
    """Lightweight smoke tests for the CLI entry point (main.py)."""

    def test_help_flag_exits_cleanly(self):
        """--help must exit with code 0 and not raise."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "main.py", "--help"],
            capture_output=True,
            text=True,
            cwd=".",
        )
        assert result.returncode == 0
        assert "--agent-timeout-seconds" in result.stdout
        assert "--auto-intuition" in result.stdout
        assert "--no-mcp" in result.stdout

    def test_auto_intuition_no_mcp_runs_to_completion(self):
        """Fully non-interactive run with mock backend must complete without hanging."""
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "--question", "ping",
                "--auto-intuition",
                "--no-mcp",
                "--max-domains", "2",
                "--agent-timeout-seconds", "10",
            ],
            capture_output=True,
            text=True,
            timeout=60,  # hard wall-clock guard for the test itself
            cwd=".",
        )
        assert result.returncode == 0, (
            f"CLI exited with {result.returncode}.\n"
            f"stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )

