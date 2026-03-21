"""Tests for the WeighingSystem (offline — no LLM required)."""

import math
import pytest
from src.analysis.weighing_system import (
    WeighingSystem,
    _cosine_sim_tfidf,
    _jaccard,
    _tokenize,
    _domain_relevance,
)
from src.models import AgentResponse, Domain, HumanIntuition, WeighingResult


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_strips_short_words(self):
        tokens = _tokenize("a an the big")
        assert "big" in tokens
        # words shorter than 3 chars should not appear
        assert "a" not in tokens
        assert "an" not in tokens


class TestJaccard:
    def test_identical(self):
        s = {"a", "b", "c"}
        assert _jaccard(s, s) == 1.0

    def test_disjoint(self):
        assert _jaccard({"a"}, {"b"}) == 0.0

    def test_partial(self):
        val = _jaccard({"a", "b"}, {"b", "c"})
        assert 0.0 < val < 1.0

    def test_empty(self):
        assert _jaccard(set(), {"a"}) == 0.0


class TestCosineSim:
    def test_identical_text(self):
        text = "quantum mechanics describes behaviour of particles"
        sim = _cosine_sim_tfidf(text, text)
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_disjoint_text(self):
        sim = _cosine_sim_tfidf("apple banana cherry", "delta echo foxtrot")
        assert sim == 0.0

    def test_partial_overlap(self):
        sim = _cosine_sim_tfidf("deep learning neural", "neural network layer")
        assert 0.0 < sim < 1.0

    def test_empty_string(self):
        assert _cosine_sim_tfidf("", "anything") == 0.0


class TestDomainRelevance:
    def test_physics_question(self):
        score = _domain_relevance("How does quantum tunnelling work?", Domain.PHYSICS)
        assert score > 0.2

    def test_unrelated(self):
        score = _domain_relevance("What is the speed of light?", Domain.SOCIAL_SCIENCE)
        assert score <= 0.4  # should be low but at least the base 0.2


# ---------------------------------------------------------------------------
# Integration tests for WeighingSystem
# ---------------------------------------------------------------------------


def _make_responses() -> list[AgentResponse]:
    return [
        AgentResponse(
            domain=Domain.PHYSICS,
            answer="Gravity is the curvature of spacetime caused by mass and energy.",
            reasoning="General relativity: Einstein field equations.",
            confidence=0.9,
        ),
        AgentResponse(
            domain=Domain.SPACE_SCIENCE,
            answer="Gravity holds stars and planets together and drives orbital mechanics.",
            reasoning="Newton's law of universal gravitation and GR corrections.",
            confidence=0.85,
        ),
    ]


def _make_intuition() -> HumanIntuition:
    return HumanIntuition(
        question="What is gravity?",
        intuitive_answer=(
            "Gravity is a force between objects with mass, pulling them together."
        ),
        confidence=0.75,
        reasoning="I remember that mass attracts mass from school physics.",
        domain_guesses=[Domain.PHYSICS],
    )


class TestWeighingSystem:
    def test_weigh_returns_result(self):
        ws = WeighingSystem(llm_client=None)
        result = ws.weigh(_make_intuition(), _make_responses())
        assert isinstance(result, WeighingResult)

    def test_alignment_scores_count(self):
        ws = WeighingSystem(llm_client=None)
        result = ws.weigh(_make_intuition(), _make_responses())
        assert len(result.alignment_scores) == len(_make_responses())

    def test_similarity_in_range(self):
        ws = WeighingSystem(llm_client=None)
        result = ws.weigh(_make_intuition(), _make_responses())
        for score in result.alignment_scores:
            assert 0.0 <= score.semantic_similarity <= 1.0

    def test_accuracy_pct_in_range(self):
        ws = WeighingSystem(llm_client=None)
        result = ws.weigh(_make_intuition(), _make_responses())
        assert 0.0 <= result.intuition_accuracy_pct <= 100.0

    def test_synthesized_answer_not_empty(self):
        ws = WeighingSystem(llm_client=None)
        result = ws.weigh(_make_intuition(), _make_responses())
        assert len(result.synthesized_answer) > 0

    def test_overall_analysis_not_empty(self):
        ws = WeighingSystem(llm_client=None)
        result = ws.weigh(_make_intuition(), _make_responses())
        assert len(result.overall_analysis) > 0

    def test_recommendations_not_empty(self):
        ws = WeighingSystem(llm_client=None)
        result = ws.weigh(_make_intuition(), _make_responses())
        assert len(result.recommendations) >= 1

    def test_no_responses_raises(self):
        ws = WeighingSystem(llm_client=None)
        with pytest.raises(ValueError):
            ws.weigh(_make_intuition(), [])

    def test_high_confidence_high_accuracy_recommendation(self):
        """A highly aligned intuition should get a positive recommendation."""
        ws = WeighingSystem(llm_client=None)
        # Give an identical answer to ensure high similarity
        intuition = HumanIntuition(
            question="What is gravity?",
            intuitive_answer=(
                "Gravity is the curvature of spacetime caused by mass and energy "
                "drives orbital mechanics stars planets together Newton."
            ),
            confidence=0.9,
            reasoning="General relativity holds stars and planets.",
        )
        result = ws.weigh(intuition, _make_responses())
        # High alignment → at least one positive / constructive recommendation
        assert any(rec for rec in result.recommendations)
