"""StrategyRouter — epsilon-greedy solver-approach selector.

The router performs two distinct tasks:

1. **Feature extraction** — lightweight, rule-based analysis of the question
   text to produce a :class:`QuestionFeatures` snapshot (no LLM call).
2. **Approach selection** — given the features, a
   :class:`~src.solver.policy.SolverPolicy`, and an epsilon value, the router
   returns a :class:`~src.solver.policy.SolverApproach` together with metadata
   about whether exploration occurred and whether the high-stakes gate fired.

Algorithm
---------
1. Extract features from *question*.
2. Compute the deterministic *recommended* approach via :meth:`_recommend`.
3. If ``policy == FIXED``:  return the forced approach immediately.
4. If ``policy == BASELINE``: return the recommended approach (no exploration).
5. Determine ε:
   - ``policy == EXPLORE``: use *explore_epsilon* (higher, e.g. 0.30).
   - ``policy == AUTO``:    use *auto_epsilon*   (lower,  e.g. 0.05).
6. High-stakes gate: if the question is high-stakes **and**
   *no_explore_high_stakes* is ``True``, skip exploration regardless of ε.
7. Epsilon-greedy draw: with probability ε, sample from the top-*k* approaches
   (weighted, not uniform) — excluding the recommended approach so exploration
   is always novel.
8. Return the selected approach with metadata.

Seeded randomness
-----------------
Pass an :class:`random.Random` instance as *rng* to :meth:`select` for fully
deterministic / reproducible behaviour in tests.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Optional

from src.solver.policy import SolverApproach, SolverPolicy

# ---------------------------------------------------------------------------
# Default epsilon values
# ---------------------------------------------------------------------------
_DEFAULT_AUTO_EPSILON: float = 0.05
_DEFAULT_EXPLORE_EPSILON: float = 0.30
_DEFAULT_TOPK: int = 3

# ---------------------------------------------------------------------------
# High-stakes keyword patterns
# ---------------------------------------------------------------------------
_HIGH_STAKES_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\blegal\b",
        r"\bmedical\b",
        r"\bsecurity\b",
        r"\bmoney\b",
        r"\bfinance\b",
        r"\bfinancial\b",
        r"\blaw\b",
        r"\bregulat\w+\b",
        r"\bsafety\b",
        r"\bcritical\b",
        r"\bemergency\b",
        r"\blife.{0,10}death\b",
        r"\bcompliance\b",
        r"\bhipaa\b",
        r"\bgdpr\b",
        r"\bphi\b",
        r"\bpii\b",
    ]
]

# ---------------------------------------------------------------------------
# Experiment-signal patterns (mirrors ExperimentRunnerAgent heuristics)
# ---------------------------------------------------------------------------
_EXPERIMENT_POSITIVE: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bhow\s+(much|many|fast|slow|large|small)\b",
        r"\bwhat\s+is\s+the\s+effect\b",
        r"\bdoes\s+\w+\s+cause\b",
        r"\bsimulat\w+\b",
        r"\btest\b",
        r"\bverif\w+\b",
        r"\brun\b",
        r"\bimplement\b",
        r"\boptimi[sz]\w+\b",
        r"\bminimis\w+\b",
        r"\bmaximis\w+\b",
        r"\bminimize\b",
        r"\bmaximize\b",
        r"\bprob(ability|abilistic)\b",
        r"\bexpected\s+value\b",
        r"\bconverg\w+\b",
        r"\bscal\w+\b",
    ]
]

_EXPERIMENT_NEGATIVE: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bdefin(e|ition)\b",
        r"\bwhat\s+does\b",
        r"\bwho\s+invented\b",
        r"\bwhen\s+was\b",
        r"\bshould\s+i\b",
        r"\bwhich\s+is\s+better\b",
    ]
]

# ---------------------------------------------------------------------------
# Debate-signal patterns
# ---------------------------------------------------------------------------
_DEBATE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bcontrover\w+\b",
        r"\bpros?\s+and\s+cons?\b",
        r"\bcompare\b",
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\btradeoff\b",
        r"\btrade.off\b",
        r"\bargue\b",
        r"\badvantages?\b",
        r"\bdisadvantages?\b",
        r"\bshould\s+we\b",
        r"\bshould\s+i\b",
        r"\bbetter\b",
        r"\bworse\b",
        r"\banalys[ie]\w+\b",
    ]
]


# ---------------------------------------------------------------------------
# QuestionFeatures
# ---------------------------------------------------------------------------


@dataclass
class QuestionFeatures:
    """Lightweight feature vector extracted from a question string.

    All attributes are cheap to compute (regex + length checks only).
    """

    question: str
    length: int
    is_high_stakes: bool
    experiment_score: float   # positive → experimentable
    debate_score: float        # positive → debate-worthy
    has_run_implement: bool    # contains "run" or "implement"

    # Derived convenience properties
    @property
    def is_experimentable(self) -> bool:
        """True when the experiment score exceeds a threshold of 0.15."""
        return self.experiment_score >= 0.15

    @property
    def is_debate_worthy(self) -> bool:
        """True when the debate score exceeds 1 match."""
        return self.debate_score >= 1


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------


def extract_features(question: str) -> QuestionFeatures:
    """Return a :class:`QuestionFeatures` snapshot for *question*.

    This is a cheap, purely rule-based function: no LLM calls.
    """
    length = len(question)

    is_high_stakes = any(p.search(question) for p in _HIGH_STAKES_PATTERNS)

    exp_pos = sum(1 for p in _EXPERIMENT_POSITIVE if p.search(question))
    exp_neg = sum(1 for p in _EXPERIMENT_NEGATIVE if p.search(question))
    # Normalise to a [-1, +1]-ish score similar to ExperimentRunnerAgent
    total = exp_pos + exp_neg or 1
    experiment_score = (exp_pos - exp_neg) / total

    debate_score = float(sum(1 for p in _DEBATE_PATTERNS if p.search(question)))

    has_run_implement = bool(
        re.search(r"\b(run|implement)\b", question, re.IGNORECASE)
    )

    return QuestionFeatures(
        question=question,
        length=length,
        is_high_stakes=is_high_stakes,
        experiment_score=experiment_score,
        debate_score=debate_score,
        has_run_implement=has_run_implement,
    )


# ---------------------------------------------------------------------------
# SelectionResult
# ---------------------------------------------------------------------------


@dataclass
class SelectionResult:
    """The output of :meth:`StrategyRouter.select`."""

    approach: SolverApproach
    #: Whether epsilon-greedy exploration was used for this selection
    explored: bool = False
    #: Whether the high-stakes gate fired (and blocked exploration)
    high_stakes_gate: bool = False
    #: Deterministic recommendation before any exploration
    recommended: SolverApproach = SolverApproach.DIRECT
    features: Optional[QuestionFeatures] = None


# ---------------------------------------------------------------------------
# StrategyRouter
# ---------------------------------------------------------------------------


class StrategyRouter:
    """Route questions to a :class:`~src.solver.policy.SolverApproach`.

    Parameters
    ----------
    policy:
        The active :class:`~src.solver.policy.SolverPolicy`.
    forced_approach:
        When *policy* is ``FIXED``, always return this approach.
    auto_epsilon:
        Exploration probability used in ``AUTO`` mode.  Default: 0.05.
    explore_epsilon:
        Exploration probability used in ``EXPLORE`` mode.  Default: 0.30.
    explore_topk:
        Number of top-scored approaches to sample from during exploration.
        Default: 3.
    no_explore_high_stakes:
        When ``True`` (default) the high-stakes gate disables exploration for
        questions that contain legal, medical, financial, or security signals.
    """

    def __init__(
        self,
        policy: SolverPolicy = SolverPolicy.AUTO,
        forced_approach: SolverApproach = SolverApproach.DIRECT,
        auto_epsilon: float = _DEFAULT_AUTO_EPSILON,
        explore_epsilon: float = _DEFAULT_EXPLORE_EPSILON,
        explore_topk: int = _DEFAULT_TOPK,
        no_explore_high_stakes: bool = True,
    ) -> None:
        self.policy = policy
        self.forced_approach = forced_approach
        self.auto_epsilon = auto_epsilon
        self.explore_epsilon = explore_epsilon
        self.explore_topk = explore_topk
        self.no_explore_high_stakes = no_explore_high_stakes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        question: str,
        rng: Optional[random.Random] = None,
    ) -> SelectionResult:
        """Select an approach for *question*.

        Parameters
        ----------
        question:
            The question to route.
        rng:
            Optional seeded :class:`random.Random` instance.  When ``None``,
            the module-level ``random`` functions are used (non-deterministic).
        """
        _rng = rng or random

        features = extract_features(question)
        recommended = self._recommend(features)

        # ------------------------------------------------------------------
        # Fixed policy: ignore everything else
        # ------------------------------------------------------------------
        if self.policy is SolverPolicy.FIXED:
            return SelectionResult(
                approach=self.forced_approach,
                explored=False,
                high_stakes_gate=False,
                recommended=recommended,
                features=features,
            )

        # ------------------------------------------------------------------
        # Baseline policy: deterministic, no exploration
        # ------------------------------------------------------------------
        if self.policy is SolverPolicy.BASELINE:
            return SelectionResult(
                approach=recommended,
                explored=False,
                high_stakes_gate=False,
                recommended=recommended,
                features=features,
            )

        # ------------------------------------------------------------------
        # Auto / Explore policies: possibly explore
        # ------------------------------------------------------------------
        epsilon = (
            self.explore_epsilon
            if self.policy is SolverPolicy.EXPLORE
            else self.auto_epsilon
        )

        # High-stakes gate: if triggered, skip exploration
        if self.no_explore_high_stakes and features.is_high_stakes:
            return SelectionResult(
                approach=recommended,
                explored=False,
                high_stakes_gate=True,
                recommended=recommended,
                features=features,
            )

        # Epsilon-greedy draw
        if _rng.random() < epsilon:
            # Sample from top-k approaches excluding the recommended one
            candidates = self._top_k_approaches(features, recommended)
            if candidates:
                weights = self._approach_weights(candidates, features)
                chosen = _rng.choices(candidates, weights=weights, k=1)[0]
                return SelectionResult(
                    approach=chosen,
                    explored=True,
                    high_stakes_gate=False,
                    recommended=recommended,
                    features=features,
                )

        # Deterministic: use the recommended approach
        return SelectionResult(
            approach=recommended,
            explored=False,
            high_stakes_gate=False,
            recommended=recommended,
            features=features,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recommend(self, features: QuestionFeatures) -> SolverApproach:
        """Return the deterministic best approach for *features*."""
        # Experimentable, moderate length → experiment approach
        if features.is_experimentable and not features.is_debate_worthy:
            return SolverApproach.EXPERIMENT

        # Debate-worthy (comparative / trade-off question) → debate
        if features.is_debate_worthy and not features.is_experimentable:
            return SolverApproach.DEBATE

        # Both signals → portfolio (run multiple approaches)
        if features.is_experimentable and features.is_debate_worthy:
            return SolverApproach.PORTFOLIO

        # Long questions with no clear signal → adaptive (expand as needed)
        if features.length > 300:
            return SolverApproach.ADAPTIVE

        # Default: direct
        return SolverApproach.DIRECT

    def _top_k_approaches(
        self,
        features: QuestionFeatures,
        exclude: SolverApproach,
    ) -> list[SolverApproach]:
        """Return up to *explore_topk* approaches excluding *exclude*."""
        all_approaches = list(SolverApproach)
        scored: list[tuple[float, SolverApproach]] = []
        for approach in all_approaches:
            if approach is exclude:
                continue
            score = self._score_approach(approach, features)
            scored.append((score, approach))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [a for _, a in scored[: self.explore_topk]]

    def _score_approach(
        self, approach: SolverApproach, features: QuestionFeatures
    ) -> float:
        """Heuristic score for *approach* given *features*.

        Higher is better.  Used for weighted sampling during exploration.
        """
        score = 0.0
        if approach is SolverApproach.EXPERIMENT:
            score += max(0.0, features.experiment_score)
        if approach is SolverApproach.DEBATE:
            score += features.debate_score * 0.3
        if approach is SolverApproach.ADAPTIVE:
            score += 0.2  # mild baseline desirability
        if approach is SolverApproach.PORTFOLIO:
            score += 0.1  # always somewhat viable but costly
        if approach is SolverApproach.DIRECT:
            score += 0.15  # always somewhat viable
        return score

    def _approach_weights(
        self,
        approaches: list[SolverApproach],
        features: QuestionFeatures,
    ) -> list[float]:
        """Return positive sampling weights for *approaches* (must sum > 0)."""
        raw = [max(0.01, self._score_approach(a, features)) for a in approaches]
        total = sum(raw)
        return [w / total for w in raw]
