"""Tests for data models."""

import pytest
from src.models import (
    AgentResponse,
    AlignmentScore,
    Domain,
    HumanIntuition,
    SearchResult,
    WeighingResult,
)


class TestDomain:
    def test_all_domains_present(self):
        expected = {
            "electrical_engineering",
            "computer_science",
            "neural_networks",
            "social_science",
            "space_science",
            "physics",
            "deep_learning",
            # High-economic-value industry domains
            "healthcare",
            "climate_energy",
            "finance_economics",
            "cybersecurity",
            "biotech_genomics",
            "supply_chain",
        }
        assert {d.value for d in Domain} == expected


class TestHumanIntuition:
    def test_valid_creation(self):
        hi = HumanIntuition(
            question="What is gravity?",
            intuitive_answer="A force pulling objects together",
            confidence=0.8,
        )
        assert hi.confidence == 0.8
        assert hi.domain_guesses == []

    def test_confidence_bounds(self):
        with pytest.raises(ValueError):
            HumanIntuition(
                question="q", intuitive_answer="a", confidence=1.5
            )
        with pytest.raises(ValueError):
            HumanIntuition(
                question="q", intuitive_answer="a", confidence=-0.1
            )

    def test_boundary_confidence(self):
        hi = HumanIntuition(question="q", intuitive_answer="a", confidence=0.0)
        assert hi.confidence == 0.0
        hi2 = HumanIntuition(question="q", intuitive_answer="a", confidence=1.0)
        assert hi2.confidence == 1.0


class TestAgentResponse:
    def test_valid_creation(self):
        resp = AgentResponse(
            domain=Domain.PHYSICS,
            answer="Gravity is curvature of spacetime.",
            reasoning="General Relativity.",
            confidence=0.95,
        )
        assert resp.domain == Domain.PHYSICS
        assert resp.sources == []

    def test_confidence_bounds(self):
        with pytest.raises(ValueError):
            AgentResponse(
                domain=Domain.PHYSICS, answer="a", reasoning="r", confidence=1.1
            )


class TestSearchResult:
    def test_creation(self):
        sr = SearchResult(title="Test", url="https://example.com", snippet="A snippet")
        assert sr.relevance_score is None
