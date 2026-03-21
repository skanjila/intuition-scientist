"""Tests for the human intuition capture module."""

import pytest
from src.intuition.human_intuition import IntuitionCapture
from src.models import Domain, HumanIntuition


class TestIntuitionCapture:
    def test_prefilled_passthrough(self):
        """When a prefilled intuition is provided it should be returned as-is."""
        hi = HumanIntuition(
            question="Why does the sky appear blue?",
            intuitive_answer="Light scatters at different wavelengths.",
            confidence=0.7,
        )
        capture = IntuitionCapture(interactive=False)
        result = capture.capture("Why does the sky appear blue?", prefilled=hi)
        assert result is hi

    def test_non_interactive_without_prefilled_raises(self):
        capture = IntuitionCapture(interactive=False)
        with pytest.raises(ValueError, match="prefilled"):
            capture.capture("Some question")


class TestDomainInference:
    def test_physics_keywords(self):
        domains = IntuitionCapture.infer_domains(
            "How does quantum tunnelling work in semiconductors?"
        )
        assert Domain.PHYSICS in domains

    def test_deep_learning_keywords(self):
        domains = IntuitionCapture.infer_domains(
            "How do transformer models learn attention patterns?"
        )
        assert Domain.DEEP_LEARNING in domains or Domain.NEURAL_NETWORKS in domains

    def test_space_keywords(self):
        domains = IntuitionCapture.infer_domains("How do black holes emit radiation?")
        assert Domain.SPACE_SCIENCE in domains or Domain.PHYSICS in domains

    def test_unknown_text_returns_all_domains(self):
        """Ambiguous text with no keyword matches should return all domains."""
        domains = IntuitionCapture.infer_domains("xyz abc")
        assert set(domains) == set(Domain)

    def test_multiple_domains_detected(self):
        """A cross-domain question should trigger multiple domains."""
        domains = IntuitionCapture.infer_domains(
            "How do neural network algorithms use quantum computing circuits?"
        )
        assert len(domains) >= 2
