"""Unit tests for the solver policy router.

Coverage
--------
- High-stakes question: exploration disabled, deterministic routing.
- Normal question with epsilon=1: exploration varies across runs (seedable).
- Fixed policy: always returns the forced approach regardless of question.
- Auto policy default: selects an approach and returns expected metadata.
- Feature extractor: high-stakes keyword detection, experiment/debate scoring.
- Approach recommendations: verify routing heuristics.
"""

from __future__ import annotations

import random

import pytest

from src.solver.policy import SolverApproach, SolverPolicy
from src.solver.router import (
    QuestionFeatures,
    SelectionResult,
    StrategyRouter,
    extract_features,
)


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    def test_high_stakes_legal(self):
        f = extract_features("Is this contract enforceable under legal standards?")
        assert f.is_high_stakes

    def test_high_stakes_medical(self):
        f = extract_features("What is the medical dosage for aspirin?")
        assert f.is_high_stakes

    def test_high_stakes_financial(self):
        f = extract_features("How should I invest my money in stocks?")
        assert f.is_high_stakes

    def test_high_stakes_security(self):
        f = extract_features("What security vulnerabilities exist in this code?")
        assert f.is_high_stakes

    def test_not_high_stakes(self):
        f = extract_features("How does photosynthesis work?")
        assert not f.is_high_stakes

    def test_experiment_score_positive(self):
        f = extract_features("How fast does the algorithm converge when we simulate it?")
        assert f.experiment_score > 0

    def test_experiment_score_negative(self):
        f = extract_features("What is the definition of entropy?")
        assert f.experiment_score <= 0

    def test_debate_score_positive(self):
        f = extract_features("What are the pros and cons of React vs Vue?")
        assert f.debate_score >= 2

    def test_length_captured(self):
        q = "short"
        f = extract_features(q)
        assert f.length == len(q)

    def test_has_run_implement(self):
        f = extract_features("Can you run the test suite and implement the fix?")
        assert f.has_run_implement

    def test_not_has_run_implement(self):
        f = extract_features("Explain the theory of relativity.")
        assert not f.has_run_implement


# ---------------------------------------------------------------------------
# StrategyRouter — fixed policy
# ---------------------------------------------------------------------------


class TestStrategyRouterFixed:
    """With fixed policy the router always returns the forced approach."""

    @pytest.mark.parametrize("approach", list(SolverApproach))
    def test_fixed_always_returns_forced_approach(self, approach: SolverApproach):
        router = StrategyRouter(
            policy=SolverPolicy.FIXED,
            forced_approach=approach,
        )
        # Run many times to confirm it never deviates
        rng = random.Random(42)
        for _ in range(10):
            result = router.select("How does quantum tunnelling work?", rng=rng)
            assert result.approach is approach

    def test_fixed_never_explores(self):
        router = StrategyRouter(
            policy=SolverPolicy.FIXED,
            forced_approach=SolverApproach.DEBATE,
        )
        result = router.select("What is energy?")
        assert not result.explored

    def test_fixed_not_affected_by_high_stakes(self):
        router = StrategyRouter(
            policy=SolverPolicy.FIXED,
            forced_approach=SolverApproach.PORTFOLIO,
            no_explore_high_stakes=True,
        )
        result = router.select("Is this medical advice legal?")
        assert result.approach is SolverApproach.PORTFOLIO
        assert not result.high_stakes_gate


# ---------------------------------------------------------------------------
# StrategyRouter — high-stakes gate
# ---------------------------------------------------------------------------


class TestHighStakesGate:
    """With no_explore_high_stakes=True, exploration is blocked for high-stakes questions."""

    def test_high_stakes_blocks_exploration_auto(self):
        """High-stakes question with auto policy: no exploration."""
        router = StrategyRouter(
            policy=SolverPolicy.AUTO,
            auto_epsilon=1.0,  # would always explore if gate weren't active
            no_explore_high_stakes=True,
        )
        rng = random.Random(0)
        result = router.select("What is the correct medical dosage for ibuprofen?", rng=rng)
        assert not result.explored
        assert result.high_stakes_gate

    def test_high_stakes_blocks_exploration_explore_policy(self):
        """High-stakes question with explore policy: still blocked."""
        router = StrategyRouter(
            policy=SolverPolicy.EXPLORE,
            explore_epsilon=1.0,
            no_explore_high_stakes=True,
        )
        rng = random.Random(0)
        result = router.select("Advise me on legal contract enforceability and money.", rng=rng)
        assert not result.explored
        assert result.high_stakes_gate

    def test_high_stakes_gate_disabled_allows_exploration(self):
        """When no_explore_high_stakes=False, high-stakes questions can be explored."""
        router = StrategyRouter(
            policy=SolverPolicy.AUTO,
            auto_epsilon=1.0,
            no_explore_high_stakes=False,
        )
        rng = random.Random(0)
        result = router.select("Is this medical procedure safe?", rng=rng)
        # Gate is disabled, so high_stakes_gate should be False
        assert not result.high_stakes_gate
        # With epsilon=1 and no gate, exploration must happen
        assert result.explored

    def test_normal_question_not_gated(self):
        """Normal question is not gated even with no_explore_high_stakes=True."""
        router = StrategyRouter(
            policy=SolverPolicy.AUTO,
            auto_epsilon=1.0,
            no_explore_high_stakes=True,
        )
        rng = random.Random(0)
        result = router.select("How does photosynthesis work?", rng=rng)
        assert not result.high_stakes_gate
        assert result.explored


# ---------------------------------------------------------------------------
# StrategyRouter — exploration randomness
# ---------------------------------------------------------------------------


class TestExplorationRandomness:
    """With epsilon=1, exploration should vary across seeded runs."""

    def test_epsilon_one_explores(self):
        """With epsilon=1 and a normal question, exploration always fires."""
        router = StrategyRouter(
            policy=SolverPolicy.AUTO,
            auto_epsilon=1.0,
            no_explore_high_stakes=True,
        )
        rng = random.Random(42)
        result = router.select("How does photosynthesis work?", rng=rng)
        assert result.explored

    def test_epsilon_zero_never_explores(self):
        """With epsilon=0 in auto mode, exploration never fires."""
        router = StrategyRouter(
            policy=SolverPolicy.AUTO,
            auto_epsilon=0.0,
            no_explore_high_stakes=True,
        )
        rng = random.Random(42)
        for _ in range(20):
            result = router.select("How does photosynthesis work?", rng=rng)
            assert not result.explored

    def test_exploration_produces_different_approaches_across_seeds(self):
        """With epsilon=1, different RNG seeds can yield different explored approaches."""
        router = StrategyRouter(
            policy=SolverPolicy.AUTO,
            auto_epsilon=1.0,
            explore_topk=5,
            no_explore_high_stakes=False,
        )
        approaches = set()
        for seed in range(50):
            rng = random.Random(seed)
            result = router.select("How fast does gradient descent converge?", rng=rng)
            approaches.add(result.approach)
        # Over 50 different seeds, we expect more than one approach to be selected
        assert len(approaches) > 1

    def test_same_seed_produces_same_result(self):
        """Identical seeds must produce identical selections (reproducibility)."""
        router = StrategyRouter(
            policy=SolverPolicy.AUTO,
            auto_epsilon=1.0,
            no_explore_high_stakes=False,
        )
        q = "How does entropy change over time?"
        result_a = router.select(q, rng=random.Random(7))
        result_b = router.select(q, rng=random.Random(7))
        assert result_a.approach is result_b.approach


# ---------------------------------------------------------------------------
# StrategyRouter — baseline policy
# ---------------------------------------------------------------------------


class TestBaselinePolicy:
    """Baseline policy never explores; returns the deterministic recommendation."""

    def test_baseline_never_explores(self):
        router = StrategyRouter(
            policy=SolverPolicy.BASELINE,
            auto_epsilon=1.0,  # would explore in auto mode
        )
        rng = random.Random(0)
        for _ in range(10):
            result = router.select("What are the pros and cons of Python vs Rust?", rng=rng)
            assert not result.explored

    def test_baseline_returns_recommended(self):
        router = StrategyRouter(policy=SolverPolicy.BASELINE)
        result = router.select("What is the capital of France?")
        assert result.approach is result.recommended


# ---------------------------------------------------------------------------
# StrategyRouter — recommendation heuristics
# ---------------------------------------------------------------------------


class TestRecommendationHeuristics:
    def test_debate_question_recommends_debate(self):
        router = StrategyRouter(policy=SolverPolicy.BASELINE)
        result = router.select(
            "What are the pros and cons of microservices versus a monolith?"
        )
        # Debate-worthy question should recommend debate (not experimentable)
        assert result.recommended is SolverApproach.DEBATE

    def test_experiment_question_recommends_experiment(self):
        router = StrategyRouter(policy=SolverPolicy.BASELINE)
        result = router.select(
            "How fast does gradient descent converge when learning rate varies?"
        )
        assert result.recommended is SolverApproach.EXPERIMENT

    def test_plain_question_recommends_direct(self):
        router = StrategyRouter(policy=SolverPolicy.BASELINE)
        result = router.select("What is the capital of France?")
        assert result.recommended is SolverApproach.DIRECT

    def test_long_question_recommends_adaptive(self):
        router = StrategyRouter(policy=SolverPolicy.BASELINE)
        # 300+ character question with no strong signals
        long_q = "Explain the following concept: " + "something neutral " * 25
        result = router.select(long_q)
        assert result.recommended is SolverApproach.ADAPTIVE


# ---------------------------------------------------------------------------
# StrategyRouter — auto policy default behaviour
# ---------------------------------------------------------------------------


class TestAutoPolicy:
    def test_auto_is_default_policy(self):
        router = StrategyRouter()
        assert router.policy is SolverPolicy.AUTO

    def test_auto_returns_selection_result(self):
        router = StrategyRouter()
        result = router.select("What is quantum entanglement?")
        assert isinstance(result, SelectionResult)
        assert isinstance(result.approach, SolverApproach)
        assert isinstance(result.explored, bool)
        assert isinstance(result.high_stakes_gate, bool)
        assert isinstance(result.recommended, SolverApproach)
        assert isinstance(result.features, QuestionFeatures)

    def test_auto_small_epsilon_rarely_explores(self):
        """Default auto epsilon (0.05) should explore rarely over 200 runs."""
        router = StrategyRouter(policy=SolverPolicy.AUTO, auto_epsilon=0.05)
        explored_count = sum(
            1
            for seed in range(200)
            if router.select("What is energy?", rng=random.Random(seed)).explored
        )
        # With epsilon=0.05 and 200 trials, expected ~10 explorations.
        # Allow a wide tolerance for the statistical test.
        assert explored_count < 50


# ---------------------------------------------------------------------------
# CLI integration smoke test
# ---------------------------------------------------------------------------


class TestCLIFlags:
    """Minimal integration tests: ensure new CLI flags are accepted."""

    def test_solver_policy_help_contains_new_flags(self, capsys):
        """--help output must mention the new solver-policy flags."""
        import sys

        from main import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])
        captured = capsys.readouterr()
        assert "--solver-policy" in captured.out
        assert "--solver-approach" in captured.out
        assert "--explore-epsilon" in captured.out

    def test_solver_policy_auto_accepted(self):
        from main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--question", "test?", "--solver-policy", "auto"])
        assert args.solver_policy == "auto"

    def test_solver_policy_fixed_accepted(self):
        from main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            ["--question", "test?", "--solver-policy", "fixed", "--solver-approach", "debate"]
        )
        assert args.solver_policy == "fixed"
        assert args.solver_approach == "debate"

    def test_explore_epsilon_accepted(self):
        from main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--question", "test?", "--explore-epsilon", "0.25"])
        assert args.explore_epsilon == pytest.approx(0.25)

    def test_explore_topk_accepted(self):
        from main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--question", "test?", "--explore-topk", "5"])
        assert args.explore_topk == 5

    def test_explore_high_stakes_flag(self):
        from main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--question", "test?", "--explore-high-stakes"])
        assert not args.no_explore_high_stakes

    def test_default_solver_policy_is_auto(self):
        from main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--question", "test?"])
        assert args.solver_policy == "auto"

    def test_invalid_solver_policy_rejected(self):
        from main import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--solver-policy", "invalid"])

    def test_invalid_solver_approach_rejected(self):
        from main import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--solver-approach", "invalid"])
