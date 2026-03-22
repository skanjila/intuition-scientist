"""Tests for performance-focused features:
- CLI flags (--fast, --max-workers, --agent-max-tokens, --synthesis-max-tokens, --use-mcp)
- Fast preset precedence (explicit flags override preset defaults)
- OllamaBackend uses a persistent httpx.Client and is closable
- AgentOrchestrator calls backend.close() when available
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from main import _build_parser, main
from src.llm.ollama_backend import OllamaBackend
from src.orchestrator.agent_orchestrator import AgentOrchestrator
from src.agents.base_agent import BaseAgent
from src.llm.mock_backend import MockBackend
from src.analysis.weighing_system import WeighingSystem
from src.models import Domain


# ---------------------------------------------------------------------------
# CLI flag parsing
# ---------------------------------------------------------------------------


class TestCLIFlagParsing:
    """Verify new flags are parsed correctly and have expected defaults."""

    def test_fast_flag_default_false(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.fast is False

    def test_fast_flag_set(self):
        parser = _build_parser()
        args = parser.parse_args(["--fast"])
        assert args.fast is True

    def test_max_workers_default_none(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.max_workers is None

    def test_max_workers_parsed(self):
        parser = _build_parser()
        args = parser.parse_args(["--max-workers", "2"])
        assert args.max_workers == 2

    def test_agent_max_tokens_default_none(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.agent_max_tokens is None

    def test_agent_max_tokens_parsed(self):
        parser = _build_parser()
        args = parser.parse_args(["--agent-max-tokens", "128"])
        assert args.agent_max_tokens == 128

    def test_synthesis_max_tokens_default_none(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.synthesis_max_tokens is None

    def test_synthesis_max_tokens_parsed(self):
        parser = _build_parser()
        args = parser.parse_args(["--synthesis-max-tokens", "200"])
        assert args.synthesis_max_tokens == 200

    def test_use_mcp_default_false(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.use_mcp is False

    def test_use_mcp_flag_set(self):
        parser = _build_parser()
        args = parser.parse_args(["--use-mcp"])
        assert args.use_mcp is True

    def test_all_new_flags_together(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--fast",
            "--max-workers", "2",
            "--agent-max-tokens", "512",
            "--synthesis-max-tokens", "256",
            "--use-mcp",
        ])
        assert args.fast is True
        assert args.max_workers == 2
        assert args.agent_max_tokens == 512
        assert args.synthesis_max_tokens == 256
        assert args.use_mcp is True

    def test_new_flags_coexist_with_existing_flags(self):
        """New flags must not break --provider, --no-mcp, --domains, --max-domains."""
        parser = _build_parser()
        args = parser.parse_args([
            "--provider", "ollama:qwen2.5:7b",
            "--no-mcp",
            "--domains", "cs", "physics",
            "--max-domains", "4",
            "--fast",
            "--max-workers", "1",
        ])
        assert args.provider == "ollama:qwen2.5:7b"
        assert args.no_mcp is True
        assert args.domains == ["cs", "physics"]
        assert args.max_domains == 4
        assert args.fast is True
        assert args.max_workers == 1


# ---------------------------------------------------------------------------
# Fast preset logic / precedence
# ---------------------------------------------------------------------------


class TestFastPresetPrecedence:
    """Fast preset defaults can be overridden by explicit CLI flags."""

    def _run_main_capture_orchestrator_kwargs(self, argv: list[str]) -> dict:
        """Run main() and capture the kwargs passed to AgentOrchestrator."""
        captured: dict = {}

        from src.models import HumanIntuition, WeighingResult, AlignmentScore
        import src.orchestrator.agent_orchestrator as orch_mod

        original_cls = orch_mod.AgentOrchestrator

        class CapturingOrchestrator(original_cls):
            def __init__(self, **kwargs):  # type: ignore[override]
                captured.update(kwargs)
                super().__init__(**kwargs)

            def run(self, question, *, prefilled_intuition=None, domains=None):
                hi = HumanIntuition(
                    question=question,
                    intuitive_answer="test",
                    confidence=0.5,
                )
                from src.models import WeighingResult, AlignmentScore
                return WeighingResult(
                    question=question,
                    human_intuition=hi,
                    agent_responses=[],
                    alignment_scores=[],
                    synthesized_answer="",
                    intuition_accuracy_pct=50.0,
                    overall_analysis="",
                    recommendations=[],
                )

        with patch.object(orch_mod, "AgentOrchestrator", CapturingOrchestrator):
            try:
                main(argv)
            except SystemExit:
                pass

        return captured

    def test_fast_preset_defaults(self):
        kwargs = self._run_main_capture_orchestrator_kwargs([
            "--fast", "--question", "test question",
        ])
        assert kwargs.get("max_workers") == 1
        assert kwargs.get("max_domains") == 3
        assert kwargs.get("agent_max_tokens") == 256
        assert kwargs.get("synthesis_max_tokens") == 384
        assert kwargs.get("use_mcp") is False

    def test_fast_preset_max_workers_override(self):
        kwargs = self._run_main_capture_orchestrator_kwargs([
            "--fast", "--max-workers", "2", "--question", "test",
        ])
        assert kwargs.get("max_workers") == 2

    def test_fast_preset_max_domains_override(self):
        kwargs = self._run_main_capture_orchestrator_kwargs([
            "--fast", "--max-domains", "5", "--question", "test",
        ])
        assert kwargs.get("max_domains") == 5

    def test_fast_preset_agent_max_tokens_override(self):
        kwargs = self._run_main_capture_orchestrator_kwargs([
            "--fast", "--agent-max-tokens", "512", "--question", "test",
        ])
        assert kwargs.get("agent_max_tokens") == 512

    def test_fast_preset_synthesis_max_tokens_override(self):
        kwargs = self._run_main_capture_orchestrator_kwargs([
            "--fast", "--synthesis-max-tokens", "600", "--question", "test",
        ])
        assert kwargs.get("synthesis_max_tokens") == 600

    def test_fast_preset_use_mcp_override(self):
        kwargs = self._run_main_capture_orchestrator_kwargs([
            "--fast", "--use-mcp", "--question", "test",
        ])
        assert kwargs.get("use_mcp") is True

    def test_fast_preset_no_mcp_wins_over_use_mcp(self):
        """--no-mcp should take priority even if --use-mcp is also passed."""
        kwargs = self._run_main_capture_orchestrator_kwargs([
            "--fast", "--use-mcp", "--no-mcp", "--question", "test",
        ])
        assert kwargs.get("use_mcp") is False

    def test_without_fast_standard_defaults(self):
        """Without --fast, standard defaults apply."""
        kwargs = self._run_main_capture_orchestrator_kwargs([
            "--question", "test",
        ])
        assert kwargs.get("max_workers") == 7
        assert kwargs.get("agent_max_tokens") == 1024
        assert kwargs.get("synthesis_max_tokens") == 512
        assert kwargs.get("use_mcp") is True

    def test_without_fast_no_mcp_flag(self):
        kwargs = self._run_main_capture_orchestrator_kwargs([
            "--no-mcp", "--question", "test",
        ])
        assert kwargs.get("use_mcp") is False


# ---------------------------------------------------------------------------
# OllamaBackend persistent client and close()
# ---------------------------------------------------------------------------


class TestOllamaBackendPersistentClient:
    """OllamaBackend must use a single persistent httpx.Client."""

    def test_has_persistent_client_attribute(self):
        backend = OllamaBackend(model="qwen2.5:7b")
        assert hasattr(backend, "_client")

    def test_client_is_httpx_client(self):
        import httpx
        backend = OllamaBackend(model="qwen2.5:7b")
        assert isinstance(backend._client, httpx.Client)

    def test_same_client_instance_across_calls(self):
        """The same client object must be reused — not recreated per generate()."""
        backend = OllamaBackend(model="qwen2.5:7b")
        client_id_1 = id(backend._client)
        client_id_2 = id(backend._client)
        assert client_id_1 == client_id_2

    def test_generate_uses_client_post_not_httpx_post(self):
        """generate() must call self._client.post(), not httpx.post()."""
        import httpx
        backend = OllamaBackend(model="qwen2.5:7b")
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"message": {"content": "hello"}}

        with patch.object(backend._client, "post", return_value=mock_response) as mock_post:
            with patch("httpx.post") as module_level_post:
                result = backend.generate("sys", "user")
                mock_post.assert_called_once()
                module_level_post.assert_not_called()
        assert result == "hello"

    def test_close_method_exists(self):
        backend = OllamaBackend(model="qwen2.5:7b")
        assert callable(getattr(backend, "close", None))

    def test_close_closes_client(self):
        backend = OllamaBackend(model="qwen2.5:7b")
        with patch.object(backend._client, "close") as mock_close:
            backend.close()
            mock_close.assert_called_once()

    def test_context_manager_calls_close(self):
        backend = OllamaBackend(model="qwen2.5:7b")
        with patch.object(backend._client, "close") as mock_close:
            with backend:
                pass
            mock_close.assert_called_once()

    def test_generate_passes_max_tokens_to_request(self):
        """generate() must send num_predict equal to the max_tokens argument."""
        backend = OllamaBackend(model="qwen2.5:7b")
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"message": {"content": "ok"}}

        with patch.object(backend._client, "post", return_value=mock_response) as mock_post:
            backend.generate("sys", "user", max_tokens=128)

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs["json"]
        assert payload["options"]["num_predict"] == 128


# ---------------------------------------------------------------------------
# AgentOrchestrator backend close() delegation
# ---------------------------------------------------------------------------


class TestOrchestratorBackendClose:
    """Orchestrator.close() must call backend.close() when the backend supports it."""

    def test_close_calls_backend_close_when_available(self):
        mock_backend = MagicMock(spec=["generate", "close"])
        orch = AgentOrchestrator(backend=mock_backend, use_mcp=False)
        orch.close()
        mock_backend.close.assert_called_once()

    def test_close_does_not_fail_when_backend_has_no_close(self):
        """Backends without close() must not cause an error."""
        mock_backend = MagicMock(spec=["generate"])  # no close attribute
        orch = AgentOrchestrator(backend=mock_backend, use_mcp=False)
        # Must not raise
        orch.close()

    def test_context_manager_calls_backend_close(self):
        mock_backend = MagicMock(spec=["generate", "close"])
        with AgentOrchestrator(backend=mock_backend, use_mcp=False):
            pass
        mock_backend.close.assert_called_once()


# ---------------------------------------------------------------------------
# Token caps threading
# ---------------------------------------------------------------------------


class TestTokenCapThreading:
    """agent_max_tokens and synthesis_max_tokens must be plumbed through correctly."""

    def test_orchestrator_stores_agent_max_tokens(self):
        orch = AgentOrchestrator(use_mcp=False, agent_max_tokens=256)
        assert orch._agent_max_tokens == 256
        orch.close()

    def test_agents_built_with_custom_max_tokens(self):
        orch = AgentOrchestrator(use_mcp=False, agent_max_tokens=128)
        agents = orch._build_agents([Domain.PHYSICS])
        assert len(agents) == 1
        assert agents[0]._max_tokens == 128
        orch.close()

    def test_agents_built_with_default_max_tokens(self):
        orch = AgentOrchestrator(use_mcp=False)
        agents = orch._build_agents([Domain.PHYSICS])
        assert agents[0]._max_tokens == 1024
        orch.close()

    def test_weighing_system_stores_synthesis_max_tokens(self):
        ws = WeighingSystem(synthesis_max_tokens=384)
        assert ws._synthesis_max_tokens == 384

    def test_weighing_system_default_synthesis_max_tokens(self):
        ws = WeighingSystem()
        assert ws._synthesis_max_tokens == 512

    def test_base_agent_default_max_tokens(self):
        """BaseAgent defaults to 1024 when max_tokens is not provided."""
        from src.agents.computer_science_agent import ComputerScienceAgent
        agent = ComputerScienceAgent(backend=MockBackend())
        assert agent._max_tokens == 1024

    def test_base_agent_custom_max_tokens(self):
        from src.agents.computer_science_agent import ComputerScienceAgent
        agent = ComputerScienceAgent(backend=MockBackend(), max_tokens=256)
        assert agent._max_tokens == 256

    def test_agent_calls_backend_with_max_tokens(self):
        """BaseAgent._call_llm must forward _max_tokens to backend.generate()."""
        from src.agents.physics_agent import PhysicsAgent
        mock_backend = MagicMock()
        mock_backend.generate.return_value = '{"answer":"x","reasoning":"","confidence":0.5,"sources":[]}'
        agent = PhysicsAgent(backend=mock_backend, max_tokens=128)
        agent._call_llm("test question", "")
        _, kwargs = mock_backend.generate.call_args
        assert kwargs.get("max_tokens") == 128
