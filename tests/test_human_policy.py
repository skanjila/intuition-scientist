"""Tests for the human involvement policy module and new CLI flags.

Covers:
- domain selection returns >0 domains
- orchestrator queries the expected number of agents (mock backend)
- human policy escalation triggers under high-stakes domain and low confidence
- non-interactive mode never prompts (ensures no stdin read)
- new CLI flags: --interactive, --non-interactive, --human-policy, --verbose, --quiet
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from src.intuition.human_policy import (
    HIGH_DISAGREEMENT_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    HumanPolicy,
    decide_interactive,
    has_high_disagreement,
    has_high_stakes_domain,
    has_low_confidence,
    has_missing_mcp_for_tool_domains,
    should_escalate,
)
from src.models import AgentResponse, Domain, HumanIntuition, WeighingResult
from src.orchestrator.agent_orchestrator import AgentOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(domain: Domain, confidence: float, mcp_context: str = "") -> AgentResponse:
    return AgentResponse(
        domain=domain,
        answer="test answer",
        reasoning="test reasoning",
        confidence=confidence,
        mcp_context=mcp_context,
    )


def _make_intuition(question: str = "test?") -> HumanIntuition:
    return HumanIntuition(
        question=question,
        intuitive_answer="test",
        confidence=0.5,
    )


# ---------------------------------------------------------------------------
# Domain selection tests
# ---------------------------------------------------------------------------


class TestDomainSelection:
    """Domain selection must always return at least one domain."""

    def test_physics_question_returns_physics(self):
        orch = AgentOrchestrator(use_mcp=False)
        hi = _make_intuition("How does quantum tunnelling work?")
        domains = orch._select_domains("How does quantum tunnelling work?", hi)
        assert len(domains) > 0
        assert Domain.PHYSICS in domains
        orch.close()

    def test_gibberish_question_returns_at_least_three_domains(self):
        """Even a question with no keyword hits returns ≥3 domains."""
        orch = AgentOrchestrator(use_mcp=False)
        hi = _make_intuition("xyzzyplugh")
        domains = orch._select_domains("xyzzyplugh", hi)
        assert len(domains) >= 3
        orch.close()

    def test_high_stakes_domain_detected(self):
        """A medical question should include HEALTHCARE in the domain list."""
        orch = AgentOrchestrator(use_mcp=False)
        hi = _make_intuition("What is the best treatment for type-2 diabetes?")
        domains = orch._select_domains(
            "What is the best treatment for type-2 diabetes?", hi
        )
        assert len(domains) > 0
        assert Domain.HEALTHCARE in domains
        orch.close()


# ---------------------------------------------------------------------------
# Orchestrator agent count tests
# ---------------------------------------------------------------------------


class TestOrchestratorAgentCount:
    """Orchestrator must query the expected number of agents."""

    def test_explicit_domains_queries_exactly_those_agents(self):
        """When domains are specified, only those agents are queried."""
        intuition = _make_intuition("What is energy?")
        orch = AgentOrchestrator(use_mcp=False)
        result = orch.run(
            "What is energy?",
            prefilled_intuition=intuition,
            domains=[Domain.PHYSICS, Domain.COMPUTER_SCIENCE],
        )
        assert isinstance(result, WeighingResult)
        assert len(result.agent_responses) == 2
        orch.close()

    def test_max_domains_caps_agent_count(self):
        """--max-domains must limit the number of agents queried."""
        intuition = _make_intuition("neural network learning")
        orch = AgentOrchestrator(use_mcp=False, max_domains=2)
        result = orch.run(
            "How does a neural network learn?",
            prefilled_intuition=intuition,
        )
        assert len(result.agent_responses) <= 2
        orch.close()

    def test_mock_backend_agents_run_and_return_responses(self):
        """With mock backend, all queried agents return valid AgentResponse objects."""
        intuition = _make_intuition("What is 2+2?")
        orch = AgentOrchestrator(use_mcp=False, max_domains=3)
        result = orch.run(
            "What is 2+2?",
            prefilled_intuition=intuition,
        )
        assert len(result.agent_responses) >= 1
        for resp in result.agent_responses:
            assert isinstance(resp, AgentResponse)
            assert 0.0 <= resp.confidence <= 1.0
            assert resp.answer
        orch.close()


# ---------------------------------------------------------------------------
# Human policy — escalation trigger tests
# ---------------------------------------------------------------------------


class TestHumanPolicyEscalation:
    """Escalation logic must fire correctly for each trigger condition."""

    # -- High-stakes domain --

    def test_healthcare_domain_triggers_escalation(self):
        assert has_high_stakes_domain([Domain.HEALTHCARE]) is True

    def test_legal_domain_triggers_escalation(self):
        assert has_high_stakes_domain([Domain.LEGAL_COMPLIANCE]) is True

    def test_finance_domain_triggers_escalation(self):
        assert has_high_stakes_domain([Domain.FINANCE_ECONOMICS]) is True

    def test_physics_domain_does_not_trigger_escalation(self):
        assert has_high_stakes_domain([Domain.PHYSICS]) is False

    def test_mixed_domains_with_high_stakes_triggers(self):
        assert has_high_stakes_domain([Domain.PHYSICS, Domain.HEALTHCARE]) is True

    # -- Low confidence --

    def test_low_mean_confidence_triggers_escalation(self):
        responses = [
            _make_response(Domain.PHYSICS, 0.2),
            _make_response(Domain.COMPUTER_SCIENCE, 0.3),
        ]
        # mean = 0.25 < LOW_CONFIDENCE_THRESHOLD
        assert has_low_confidence(responses) is True

    def test_high_mean_confidence_does_not_trigger(self):
        responses = [
            _make_response(Domain.PHYSICS, 0.8),
            _make_response(Domain.COMPUTER_SCIENCE, 0.9),
        ]
        assert has_low_confidence(responses) is False

    def test_empty_responses_does_not_trigger_low_confidence(self):
        assert has_low_confidence([]) is False

    def test_confidence_exactly_at_threshold_does_not_trigger(self):
        """Mean confidence exactly at threshold is NOT low (strictly below)."""
        responses = [_make_response(Domain.PHYSICS, LOW_CONFIDENCE_THRESHOLD)]
        assert has_low_confidence(responses) is False

    # -- High disagreement --

    def test_wide_confidence_spread_triggers_disagreement(self):
        responses = [
            _make_response(Domain.PHYSICS, 0.1),
            _make_response(Domain.COMPUTER_SCIENCE, 0.95),
        ]
        # spread = 0.85 > HIGH_DISAGREEMENT_THRESHOLD
        assert has_high_disagreement(responses) is True

    def test_narrow_confidence_spread_does_not_trigger(self):
        responses = [
            _make_response(Domain.PHYSICS, 0.7),
            _make_response(Domain.COMPUTER_SCIENCE, 0.75),
        ]
        assert has_high_disagreement(responses) is False

    def test_single_response_does_not_trigger_disagreement(self):
        assert has_high_disagreement([_make_response(Domain.PHYSICS, 0.5)]) is False

    # -- Missing MCP for tool-heavy domains --

    def test_mcp_enabled_but_no_results_triggers_escalation(self):
        responses = [_make_response(Domain.HEALTHCARE, 0.5, mcp_context="")]
        assert has_missing_mcp_for_tool_domains(responses, use_mcp=True) is True

    def test_mcp_disabled_does_not_trigger_missing_mcp(self):
        responses = [_make_response(Domain.HEALTHCARE, 0.5, mcp_context="")]
        assert has_missing_mcp_for_tool_domains(responses, use_mcp=False) is False

    def test_mcp_with_results_does_not_trigger(self):
        responses = [
            _make_response(Domain.HEALTHCARE, 0.5, mcp_context="some search result")
        ]
        assert has_missing_mcp_for_tool_domains(responses, use_mcp=True) is False

    def test_non_tool_domain_does_not_trigger_missing_mcp(self):
        responses = [_make_response(Domain.PHYSICS, 0.5, mcp_context="")]
        assert has_missing_mcp_for_tool_domains(responses, use_mcp=True) is False

    # -- should_escalate composite --

    def test_should_escalate_high_stakes_domain(self):
        assert should_escalate([Domain.HEALTHCARE]) is True

    def test_should_not_escalate_normal_question(self):
        responses = [
            _make_response(Domain.PHYSICS, 0.75),
            _make_response(Domain.COMPUTER_SCIENCE, 0.80),
        ]
        assert should_escalate([Domain.PHYSICS, Domain.COMPUTER_SCIENCE], responses) is False

    def test_should_escalate_low_confidence(self):
        responses = [
            _make_response(Domain.PHYSICS, 0.2),
            _make_response(Domain.COMPUTER_SCIENCE, 0.25),
        ]
        assert should_escalate([Domain.PHYSICS], responses) is True


# ---------------------------------------------------------------------------
# Human policy — decide_interactive tests
# ---------------------------------------------------------------------------


class TestDecideInteractive:
    """decide_interactive must respect policy overrides and escalation."""

    def test_always_policy_returns_true(self):
        assert decide_interactive(HumanPolicy.ALWAYS, [Domain.PHYSICS]) is True

    def test_never_policy_returns_false(self):
        assert decide_interactive(HumanPolicy.NEVER, [Domain.HEALTHCARE]) is False

    def test_auto_policy_escalates_for_high_stakes(self):
        assert decide_interactive(HumanPolicy.AUTO, [Domain.HEALTHCARE]) is True

    def test_auto_policy_does_not_escalate_for_normal(self):
        responses = [
            _make_response(Domain.PHYSICS, 0.75),
            _make_response(Domain.COMPUTER_SCIENCE, 0.80),
        ]
        result = decide_interactive(
            HumanPolicy.AUTO,
            [Domain.PHYSICS, Domain.COMPUTER_SCIENCE],
            responses=responses,
        )
        assert result is False


# ---------------------------------------------------------------------------
# Non-interactive mode never prompts (no stdin read)
# ---------------------------------------------------------------------------


class TestNonInteractiveModeNoStdin:
    """--non-interactive and --auto-intuition must never read from stdin."""

    def _run_cli(self, extra_args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                "main.py",
                "--question", "What is 2+2?",
                "--no-mcp",
                "--max-domains", "2",
                "--agent-timeout-seconds", "10",
                "--workflow-map", "off",
            ] + extra_args,
            capture_output=True,
            text=True,
            timeout=60,
            input="",  # empty stdin — would block if any input() is called
            cwd=".",
        )

    def test_non_interactive_flag_completes_without_hanging(self):
        result = self._run_cli(["--non-interactive"])
        assert result.returncode == 0, (
            f"CLI exited {result.returncode}\n"
            f"stdout: {result.stdout[:400]}\n"
            f"stderr: {result.stderr[:400]}"
        )

    def test_auto_intuition_flag_completes_without_hanging(self):
        result = self._run_cli(["--auto-intuition"])
        assert result.returncode == 0, (
            f"CLI exited {result.returncode}\n"
            f"stdout: {result.stdout[:400]}\n"
            f"stderr: {result.stderr[:400]}"
        )

    def test_human_policy_never_completes_without_hanging(self):
        result = self._run_cli(["--human-policy", "never"])
        assert result.returncode == 0, (
            f"CLI exited {result.returncode}\n"
            f"stdout: {result.stdout[:400]}\n"
            f"stderr: {result.stderr[:400]}"
        )

    def test_default_mode_completes_without_hanging(self):
        """Default mode (no flags) should be non-interactive for normal questions."""
        result = self._run_cli([])
        assert result.returncode == 0, (
            f"CLI exited {result.returncode}\n"
            f"stdout: {result.stdout[:400]}\n"
            f"stderr: {result.stderr[:400]}"
        )


# ---------------------------------------------------------------------------
# Verbose / quiet flags
# ---------------------------------------------------------------------------


class TestVerbosityFlags:
    """--verbose and --quiet flags must affect output without breaking the run."""

    def _run_cli(self, extra_args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                "main.py",
                "--question", "What is 2+2?",
                "--no-mcp",
                "--max-domains", "2",
                "--agent-timeout-seconds", "10",
                "--non-interactive",
                "--workflow-map", "off",
            ] + extra_args,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=".",
        )

    def test_verbose_flag_completes(self):
        result = self._run_cli(["--verbose"])
        assert result.returncode == 0

    def test_quiet_flag_completes(self):
        result = self._run_cli(["--quiet"])
        assert result.returncode == 0

    def test_quiet_suppresses_progress_output(self):
        result = self._run_cli(["--quiet"])
        # Progress indicators should not appear in quiet mode
        assert "Querying domain experts" not in result.stdout
        assert "Auto-intuition" not in result.stdout

    def test_normal_mode_shows_progress(self):
        result = self._run_cli([])
        # At least one progress indicator should appear
        combined = result.stdout + result.stderr
        assert any(
            kw in combined for kw in ("Querying", "domain experts", "Auto-intuition")
        )


# ---------------------------------------------------------------------------
# New CLI flags help text
# ---------------------------------------------------------------------------


class TestCLIHelpText:
    """New flags must appear in --help output."""

    def test_help_includes_new_flags(self):
        result = subprocess.run(
            [sys.executable, "main.py", "--help"],
            capture_output=True,
            text=True,
            cwd=".",
        )
        assert result.returncode == 0
        assert "--interactive" in result.stdout
        assert "--non-interactive" in result.stdout
        assert "--human-policy" in result.stdout
        assert "--verbose" in result.stdout
        assert "--quiet" in result.stdout
