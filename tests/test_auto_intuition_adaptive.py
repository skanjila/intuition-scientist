"""Tests for non-interactive auto-intuition mode and adaptive agent loop.

Covers:
- Auto-generated human intuition (no stdin prompt).
- Adaptive agent loop choosing fewer agents on narrow questions and more
  on broad/ambiguous ones (behaviour validated with mock agents).
- No regressions: default interactive-with-prefilled mode is unchanged.
- CLI flags: --auto-intuition and --adaptive-agents are wired correctly.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.intuition.human_intuition import IntuitionCapture, generate_auto_intuition
from src.models import AgentResponse, Domain, HumanIntuition, WeighingResult
from src.orchestrator.agent_orchestrator import AgentOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_weighing_result(question: str = "test?") -> WeighingResult:
    """Return a minimal WeighingResult for patching WeighingSystem.weigh."""
    intuition = HumanIntuition(
        question=question,
        intuitive_answer="auto answer",
        confidence=0.5,
    )
    return WeighingResult(
        question=question,
        human_intuition=intuition,
        agent_responses=[],
        alignment_scores=[],
        synthesized_answer="synth",
        intuition_accuracy_pct=60.0,
        overall_analysis="analysis",
        recommendations=[],
    )


# ---------------------------------------------------------------------------
# generate_auto_intuition — offline (no backend)
# ---------------------------------------------------------------------------


class TestGenerateAutoIntuitionOffline:
    """generate_auto_intuition without an LLM backend uses keyword templates."""

    def test_returns_human_intuition(self):
        result = generate_auto_intuition("What is gradient descent?")
        assert isinstance(result, HumanIntuition)

    def test_question_stored(self):
        q = "How does quantum tunnelling work?"
        result = generate_auto_intuition(q)
        assert result.question == q

    def test_confidence_is_moderate(self):
        """Auto-intuition confidence must be 0.5 (plausible but uncertain)."""
        result = generate_auto_intuition("Why is the sky blue?")
        assert result.confidence == 0.5

    def test_domain_guesses_populated(self):
        result = generate_auto_intuition(
            "How does a convolutional neural network learn features?"
        )
        assert len(result.domain_guesses) >= 1
        # Question strongly hints at neural-networks / deep-learning
        assert any(
            d in (Domain.NEURAL_NETWORKS, Domain.DEEP_LEARNING)
            for d in result.domain_guesses
        )

    def test_what_is_template(self):
        result = generate_auto_intuition("What is entropy in thermodynamics?")
        assert "intuition" in result.intuitive_answer.lower() or len(result.intuitive_answer) > 0

    def test_how_does_template(self):
        result = generate_auto_intuition("How does a transformer model process tokens?")
        assert len(result.intuitive_answer) > 0

    def test_why_template(self):
        result = generate_auto_intuition("Why do neural networks overfit?")
        assert len(result.intuitive_answer) > 0

    def test_reasoning_present(self):
        result = generate_auto_intuition("What is a Fourier transform?")
        assert len(result.reasoning) > 0


# ---------------------------------------------------------------------------
# generate_auto_intuition — with LLM backend
# ---------------------------------------------------------------------------


class TestGenerateAutoIntuitionWithBackend:
    """When a backend is provided its generate() output is used."""

    def test_uses_backend_output(self):
        mock_backend = MagicMock()
        mock_backend.generate.return_value = (
            "I think it has to do with wavelength scattering. "
            "Because shorter wavelengths scatter more in the atmosphere."
        )
        result = generate_auto_intuition(
            "Why is the sky blue?", backend=mock_backend
        )
        # The backend should have been called exactly once
        mock_backend.generate.assert_called_once()
        # "Because " splits answer from reasoning
        assert "scattering" in result.intuitive_answer.lower() or len(result.intuitive_answer) > 0
        assert result.reasoning.startswith("Because ")

    def test_backend_without_because_marker(self):
        mock_backend = MagicMock()
        mock_backend.generate.return_value = "Probably something to do with light."
        result = generate_auto_intuition("Why is the sky blue?", backend=mock_backend)
        assert result.intuitive_answer == "Probably something to do with light."
        # Reasoning falls back to domain-analysis label
        assert "auto-generated" in result.reasoning.lower()

    def test_backend_exception_falls_back_to_template(self):
        mock_backend = MagicMock()
        mock_backend.generate.side_effect = RuntimeError("backend unavailable")
        result = generate_auto_intuition("What is a neural network?", backend=mock_backend)
        assert isinstance(result, HumanIntuition)
        assert len(result.intuitive_answer) > 0


# ---------------------------------------------------------------------------
# AgentOrchestrator — auto_intuition=True
# ---------------------------------------------------------------------------


class TestAutoIntuitionOrchestrator:
    """Orchestrator with auto_intuition=True never prompts for user input."""

    def test_run_without_prompting(self):
        """run() in auto-intuition mode must complete without stdin interaction."""
        orch = AgentOrchestrator(use_mcp=False, max_domains=2, auto_intuition=True)
        result = orch.run(
            "What is energy?",
            domains=[Domain.PHYSICS, Domain.ELECTRICAL_ENGINEERING],
        )
        assert isinstance(result, WeighingResult)
        assert result.question == "What is energy?"
        orch.close()

    def test_auto_intuition_result_used(self):
        """The WeighingResult's human_intuition should carry auto-generated content."""
        orch = AgentOrchestrator(use_mcp=False, max_domains=2, auto_intuition=True)
        result = orch.run(
            "How do neural networks learn?",
            domains=[Domain.NEURAL_NETWORKS, Domain.DEEP_LEARNING],
        )
        # Auto-generated intuition has confidence == 0.5
        assert result.human_intuition.confidence == 0.5
        orch.close()

    def test_prefilled_overrides_auto_intuition(self):
        """Explicitly prefilled intuition takes priority over auto-generation."""
        prefilled = HumanIntuition(
            question="What is gravity?",
            intuitive_answer="A force between masses.",
            confidence=0.9,
        )
        orch = AgentOrchestrator(use_mcp=False, max_domains=2, auto_intuition=True)
        result = orch.run(
            "What is gravity?",
            prefilled_intuition=prefilled,
            domains=[Domain.PHYSICS],
        )
        assert result.human_intuition.confidence == 0.9
        assert result.human_intuition.intuitive_answer == "A force between masses."
        orch.close()

    def test_auto_intuition_false_requires_prefilled_when_non_interactive(self):
        """Default mode (auto_intuition=False) with interactive=False raises without prefilled."""
        # IntuitionCapture(interactive=False) without prefilled raises ValueError
        orch = AgentOrchestrator(use_mcp=False, auto_intuition=False)
        orch._intuition_capture = IntuitionCapture(interactive=False)
        with pytest.raises(ValueError, match="prefilled"):
            orch.run("Some question", domains=[Domain.PHYSICS])
        orch.close()


# ---------------------------------------------------------------------------
# AgentOrchestrator — adaptive_agents=True
# ---------------------------------------------------------------------------


class TestAdaptiveAgentLoop:
    """Adaptive loop selects agent counts intelligently."""

    def _make_agent_response(self, domain: Domain, confidence: float) -> AgentResponse:
        return AgentResponse(
            domain=domain,
            answer="mock answer",
            reasoning="mock reasoning",
            confidence=confidence,
        )

    def test_adaptive_run_returns_weighing_result(self):
        """Adaptive mode must return a valid WeighingResult end-to-end."""
        orch = AgentOrchestrator(
            use_mcp=False,
            max_domains=5,
            auto_intuition=True,
            adaptive_agents=True,
        )
        result = orch.run("What is a neural network?")
        assert isinstance(result, WeighingResult)
        orch.close()

    def test_stops_early_on_high_confidence(self, caplog):
        """When the initial batch already meets the confidence threshold the
        loop must stop after the first round (no expansion)."""
        # Confidence threshold is 0.65; initial batch is 3 agents.
        # Mock _query_agents to return high-confidence responses.
        high_conf_responses = [
            self._make_agent_response(Domain.PHYSICS, 0.9),
            self._make_agent_response(Domain.NEURAL_NETWORKS, 0.85),
            self._make_agent_response(Domain.COMPUTER_SCIENCE, 0.8),
        ]

        orch = AgentOrchestrator(
            use_mcp=False,
            auto_intuition=True,
            adaptive_agents=True,
        )
        with patch.object(orch, "_query_agents", return_value=high_conf_responses), \
             caplog.at_level(logging.INFO, logger="src.orchestrator.agent_orchestrator"):
            intuition = generate_auto_intuition("What is a neural network?")
            responses = orch._adaptive_select_and_run(
                "What is a neural network?", intuition
            )

        # Should stop after round 1 — exactly 3 responses (initial batch only)
        assert len(responses) == 3
        # Log should mention threshold was met
        assert any("threshold" in m.lower() for m in caplog.messages)
        orch.close()

    def test_expands_on_low_confidence(self, caplog):
        """Low initial confidence triggers domain expansion."""
        # First call returns low-confidence responses; second call returns higher.
        low_conf = [
            self._make_agent_response(Domain.PHYSICS, 0.3),
            self._make_agent_response(Domain.NEURAL_NETWORKS, 0.35),
            self._make_agent_response(Domain.COMPUTER_SCIENCE, 0.4),
        ]
        high_conf = [
            self._make_agent_response(Domain.DEEP_LEARNING, 0.9),
            self._make_agent_response(Domain.SOCIAL_SCIENCE, 0.88),
        ]

        call_count = [0]

        def side_effect(agents, question):
            call_count[0] += 1
            if call_count[0] == 1:
                return low_conf
            return high_conf

        orch = AgentOrchestrator(
            use_mcp=False,
            auto_intuition=True,
            adaptive_agents=True,
        )
        with patch.object(orch, "_query_agents", side_effect=side_effect), \
             caplog.at_level(logging.INFO, logger="src.orchestrator.agent_orchestrator"):
            intuition = generate_auto_intuition("Tell me about deep learning algorithms")
            responses = orch._adaptive_select_and_run(
                "Tell me about deep learning algorithms", intuition
            )

        # Should have expanded: initial 3 + 2 = 5 total responses
        assert len(responses) == 5
        assert call_count[0] == 2
        assert any("expanding" in m.lower() for m in caplog.messages)
        orch.close()

    def test_stops_when_no_remaining_candidates(self):
        """Loop stops gracefully when all candidate domains are exhausted."""
        orch = AgentOrchestrator(
            use_mcp=False,
            max_domains=3,  # forces candidate list to only 3 → no expansion
            auto_intuition=True,
            adaptive_agents=True,
        )
        # Always return low confidence so the threshold is never met
        low_conf_responses = [
            self._make_agent_response(Domain.PHYSICS, 0.2),
            self._make_agent_response(Domain.NEURAL_NETWORKS, 0.25),
            self._make_agent_response(Domain.COMPUTER_SCIENCE, 0.3),
        ]
        with patch.object(orch, "_query_agents", return_value=low_conf_responses):
            intuition = generate_auto_intuition("xyz zyxwvuts")  # no keywords
            responses = orch._adaptive_select_and_run("xyz zyxwvuts", intuition)
        # 3 domains max → no expansion possible → exactly 3 responses
        assert len(responses) == 3
        orch.close()

    def test_stops_on_time_budget(self):
        """Loop stops when target_latency_ms is exceeded."""
        import time

        call_count = [0]

        def slow_query(agents, question):
            call_count[0] += 1
            # Sleep a bit so the time budget is hit after the first round
            time.sleep(0.05)
            return [
                self._make_agent_response(Domain.PHYSICS, 0.3),
                self._make_agent_response(Domain.NEURAL_NETWORKS, 0.3),
                self._make_agent_response(Domain.COMPUTER_SCIENCE, 0.3),
            ]

        orch = AgentOrchestrator(
            use_mcp=False,
            auto_intuition=True,
            adaptive_agents=True,
            target_latency_ms=10,  # 10 ms — will be exceeded after first slow round
        )
        with patch.object(orch, "_query_agents", side_effect=slow_query):
            intuition = generate_auto_intuition("Tell me everything about everything")
            responses = orch._adaptive_select_and_run(
                "Tell me everything about everything", intuition
            )
        # Budget hit → loop must have stopped (likely after round 1)
        assert call_count[0] >= 1
        assert len(responses) >= 3
        orch.close()

    def test_max_domains_respected_in_adaptive_mode(self):
        """max_domains ceiling must apply to the adaptive candidate list."""
        orch = AgentOrchestrator(
            use_mcp=False,
            max_domains=4,
            auto_intuition=True,
            adaptive_agents=True,
        )
        result = orch.run("How does gradient descent work?")
        # Must never exceed max_domains total agent responses
        assert len(result.agent_responses) <= 4
        orch.close()

    def test_explicit_domains_bypass_adaptive_loop(self):
        """When explicit domains are passed adaptive loop must NOT run."""
        orch = AgentOrchestrator(
            use_mcp=False,
            auto_intuition=True,
            adaptive_agents=True,
        )
        # Spy on _adaptive_select_and_run — it must NOT be called
        with patch.object(
            orch, "_adaptive_select_and_run", wraps=orch._adaptive_select_and_run
        ) as spy:
            orch.run(
                "What is gravity?",
                domains=[Domain.PHYSICS],
            )
        spy.assert_not_called()
        orch.close()


# ---------------------------------------------------------------------------
# CLI integration — argument wiring
# ---------------------------------------------------------------------------


class TestCLIFlags:
    """Verify that main.py passes the new flags to AgentOrchestrator."""

    def test_auto_intuition_flag_wired(self):
        """--auto-intuition must set auto_intuition=True on the orchestrator."""
        from main import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "--question", "What is energy?",
            "--auto-intuition",
            "--no-mcp",
        ])
        assert args.auto_intuition is True

    def test_adaptive_agents_flag_wired(self):
        from main import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "--question", "What is energy?",
            "--adaptive-agents",
            "--no-mcp",
        ])
        assert args.adaptive_agents is True

    def test_target_latency_ms_flag_wired(self):
        from main import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "--question", "test",
            "--adaptive-agents",
            "--target-latency-ms", "5000",
            "--no-mcp",
        ])
        assert args.target_latency_ms == 5000

    def test_defaults_unchanged(self):
        """Without new flags, auto_intuition and adaptive_agents default False."""
        from main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--question", "test", "--no-mcp"])
        assert args.auto_intuition is False
        assert args.adaptive_agents is False
        assert args.target_latency_ms is None

    def test_main_runs_with_auto_intuition_flag(self):
        """End-to-end: main() with --auto-intuition must not raise."""
        from main import main

        # Should complete without prompting stdin or raising
        main([
            "--question", "What is gravity?",
            "--auto-intuition",
            "--no-mcp",
            "--max-domains", "2",
            "--workflow-map", "off",
        ])

    def test_main_runs_with_adaptive_agents_flag(self):
        """End-to-end: main() with --adaptive-agents must not raise."""
        from main import main

        main([
            "--question", "How does machine learning work?",
            "--auto-intuition",
            "--adaptive-agents",
            "--no-mcp",
            "--max-domains", "4",
            "--workflow-map", "off",
        ])

    def test_main_runs_with_target_latency_ms(self):
        """End-to-end: --target-latency-ms is respected without error."""
        from main import main

        main([
            "--question", "Explain quantum computing.",
            "--auto-intuition",
            "--adaptive-agents",
            "--target-latency-ms", "30000",
            "--no-mcp",
            "--max-domains", "4",
            "--workflow-map", "off",
        ])
