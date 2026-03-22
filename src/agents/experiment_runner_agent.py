"""ExperimentRunnerAgent — proposes and coordinates lightweight experiments
to solve or illuminate problems.

Design principles
-----------------
* **Question classification first**: before designing any experiments the
  agent classifies whether a question *warrants* experimental investigation.
  Questions that are purely definitional, historical, or conceptual are
  routed to a direct expert analysis instead.  Only questions that are
  quantitative, causal-mechanistic, probabilistic, hypothesis-bearing, or
  optimisation-oriented receive a full experiment plan.

* Converts qualifying questions into a structured ``ExperimentPlan`` with
  explicit hypotheses, variables, runnable Python snippets, and a synthesis
  strategy that explains how to combine results across experiments.

* Defines "experiments" as **safe, local, deterministic actions**: small
  numeric simulations, toy examples, dimensional checks, or pseudo-code
  that the user can run locally — no external credentials required.

* Keeps the **human intuition loop** explicit: the ``plan_experiments``
  method accepts an optional ``HumanIntuition`` object and threads it
  through the experiment design so outcomes can be compared to the user's
  prior beliefs.

* If live code execution is not available, produces a complete
  *experiment protocol* with runnable Python/NumPy snippets the user can
  execute in their own environment.

Classification heuristics
--------------------------
The rule-based question classifier scores questions on a continuous
``[-1.0, +1.0]`` axis:

Positive signals (question warrants experiments)
    * Quantitative language: "how much/many/fast", "what is the effect of",
      "how does X scale/vary/change with Y"
    * Causal-mechanistic: "what effect does X have on Y", "does X cause Y"
    * Hypothesis-bearing: "is it true that", "prove that", "will X converge"
    * Experiment verbs: "simulate", "model", "test", "verify"
    * Optimisation: "minimise", "maximise", "find the best value of X"
    * Probabilistic: "probability of", "expected value", "Monte Carlo"
    * Comparative with metric: "which is faster", "compare A and B in terms
      of Z"

Negative signals (question is better served by direct analysis)
    * Pure definition: "what is the definition of", "define", "what does
      X mean"
    * Historical fact: "when was", "who invented", "what year"
    * Subjective recommendation: "should I use", "which framework is best"

Questions with a net score ≥ 0.15 are classified as experimentable.
"""

from __future__ import annotations

import json
import re
import textwrap
from typing import TYPE_CHECKING, Optional

from src.agents.base_agent import BaseAgent
from src.models import (
    Domain,
    ExperimentCategory,
    ExperimentPlan,
    ExperimentSpec,
    HumanIntuition,
    QuestionExperimentability,
)

if TYPE_CHECKING:
    pass  # kept for future type-only imports


# ---------------------------------------------------------------------------
# Question classification patterns
# ---------------------------------------------------------------------------
# Each pattern contributes to the experimentability score.  The patterns are
# compiled once at module load and shared across all agent instances.

# ── Positive signals ──────────────────────────────────────────────────────

_QUANTITATIVE_RE = re.compile(
    r"\b("
    r"how (?:much|many|long|fast|often|far|quickly|efficiently|slowly)|"
    r"what (?:is the (?:value|rate|probability|percentage|amount|size|speed|"
    r"magnitude|number|count|difference|ratio|relationship|correlation|effect|"
    r"complexity|growth rate|convergence rate|error rate))|"
    r"how does .{1,60}(?:scale|vary|change|depend|grow|shrink|decrease|increase)"
    r" (?:with|as|when|for)|"
    r"what (?:happens|occurs|changes|is the effect) (?:when|if|as)|"
    r"what (?:percentage|proportion|fraction)|"
    r"(?:compare|benchmark|quantify|estimate|calculate|compute|measure)"
    r")",
    re.IGNORECASE,
)

_CAUSAL_EXPERIMENT_RE = re.compile(
    r"\b("
    r"what (?:is the )?effect of .{1,60}on|"
    r"does .{1,60}(?:cause|affect|influence|impact|determine)|"
    r"how does .{1,60}(?:affect|influence|impact|change|alter)|"
    r"(?:relationship|correlation|dependence|coupling) between|"
    r"effect of .{1,60}on"
    r")",
    re.IGNORECASE,
)

_HYPOTHESIS_RE = re.compile(
    r"\b("
    r"is it true that|prove that|show that|verify that|"
    r"does .{1,60}(?:hold|work|apply|converge|diverge)|"
    r"will .{1,60}(?:converge|diverge|increase|decrease|stabilise|stabilize)|"
    r"can we (?:show|prove|demonstrate|verify)"
    r")",
    re.IGNORECASE,
)

_EXPERIMENT_VERB_RE = re.compile(
    r"\b("
    r"simulat(?:e|ion)|model(?:ling)?|run (?:an? )?(?:experiment|simulation|test)|"
    r"test(?:ing)?|experiment(?:ally)?|empirically|numerically|computationally|"
    r"implement|write (?:a )?(?:program|code|function|script|simulation)|"
    r"code (?:up|a)|benchmark"
    r")",
    re.IGNORECASE,
)

_OPTIMISATION_RE = re.compile(
    r"\b("
    r"optim(?:ize|ise|al|um|ization|isation)|"
    r"best (?:value|parameter|setting|choice|hyperparameter)|"
    r"minimis(?:e|ing)|minimiz(?:e|ing)|"
    r"maximis(?:e|ing)|maximiz(?:e|ing)|"
    r"find the (?:best|optimal|minimum|maximum|lowest|highest)"
    r")",
    re.IGNORECASE,
)

_PROBABILISTIC_RE = re.compile(
    r"\b("
    r"probability|likelihood|expected value|distribution|random variable|"
    r"monte carlo|stochastic|variance|standard deviation|"
    r"confidence interval|p-value|significance|bayesian|prior|posterior"
    r")",
    re.IGNORECASE,
)

_COMPARATIVE_METRIC_RE = re.compile(
    r"\b("
    r"which is (?:faster|slower|better|worse|more|less) (?:than|in terms of)|"
    r"compare .{1,60}(?:in terms of|with respect to|vs|versus)|"
    r"(?:faster|slower|more efficient|less efficient) than"
    r")",
    re.IGNORECASE,
)

# ── Negative signals ──────────────────────────────────────────────────────

# Strong definitional: requires an explicit definitional marker such as
# "definition of", "meaning of", "define", "what does X mean", etc.
# Deliberately does NOT match bare "What is X?" so that questions like
# "What is the expected value of X?" or "What is the probability of X?"
# are NOT penalised here (they are caught by positive probabilistic patterns).
_DEFINITIONAL_RE = re.compile(
    r"^(?:"
    r"(?:what is|what are)(?: the)? (?:definition|meaning|concept)\b|"
    r"define\b|"
    r"what does .{1,40}\bmean\b|"
    r"explain (?:what|the concept of|the meaning)\b|"
    r"describe (?:what|the concept of|the meaning)\b"
    r")",
    re.IGNORECASE,
)

# Weak definitional: bare "What is X?" or "What are X?" with a short noun phrase.
# Applied with a smaller penalty (-0.20) than _DEFINITIONAL_RE (-0.40) so a single
# positive signal (e.g. probabilistic +0.25) can still outweigh it.
_SIMPLE_WHAT_IS_RE = re.compile(
    r"^what (?:is|are) (?:a |an |the )?[a-zA-Z][a-zA-Z\- ]{1,40}\??\s*$",
    re.IGNORECASE,
)

_HISTORICAL_RE = re.compile(
    r"\b("
    r"when (?:was|did|were|is)|who (?:invented|discovered|created|first)|"
    r"what year|history of|originally|founded|origin of"
    r")",
    re.IGNORECASE,
)

_RECOMMENDATION_ONLY_RE = re.compile(
    r"\b("
    r"should i (?:use|choose|prefer|adopt|pick)|"
    r"what (?:is the best|should i (?:use|choose))|"
    r"recommend|"
    r"which (?:framework|library|approach|tool|language|method|technique)"
    r" (?:is|should i|do you) "
    r")",
    re.IGNORECASE,
)

# ── Threshold ─────────────────────────────────────────────────────────────
# Public so tests and downstream code can reference it without importing
# a private symbol.
EXPERIMENTABLE_THRESHOLD: float = 0.15
# Keep the private alias for backward compatibility within this module.
_EXPERIMENTABLE_THRESHOLD = EXPERIMENTABLE_THRESHOLD


class ExperimentRunnerAgent(BaseAgent):
    """Agent that classifies questions and converts qualifying ones into
    structured experiment plans.

    The agent operates in two modes depending on question classification:

    **Experiment mode** (``is_experimentable=True``)
        The agent generates a :class:`~src.models.ExperimentPlan` containing
        2–4 :class:`~src.models.ExperimentSpec` objects.  Each spec covers
        one falsifiable hypothesis with a runnable Python/NumPy snippet.

    **Direct-analysis mode** (``is_experimentable=False``)
        The agent produces a standard expert analysis (same interface as all
        other domain agents) and explains why experiments would not add value.

    The classification is deterministic and rule-based (no LLM required),
    making behaviour predictable and fully unit-testable.
    """

    domain = Domain.EXPERIMENT_RUNNER

    # Taxonomy of lightweight experiment types supported by the agent
    EXPERIMENT_TYPES: list[tuple[str, str]] = [
        (
            "Numeric sweep",
            "Vary one parameter across a range and observe the output metric; "
            "encapsulated in a few lines of NumPy/SciPy.",
        ),
        (
            "Monte Carlo simulation",
            "Sample from probability distributions N times; compute mean, variance, "
            "and tail probabilities to test a probabilistic claim.",
        ),
        (
            "Toy analytical example",
            "Simplify to a 2-3 variable system, solve exactly, then check whether "
            "the exact solution confirms or refutes the hypothesis.",
        ),
        (
            "Dimensional / scaling analysis",
            "Use Buckingham-π to derive the expected scaling law; validate against "
            "limiting cases (very small / very large parameter values).",
        ),
        (
            "Finite-difference / Euler integration",
            "Discretise a differential equation, integrate forward in time, and "
            "compare the trajectory to the analytical or intuited result.",
        ),
        (
            "Search / combinatorial enumeration",
            "Enumerate all small cases (N ≤ 20) and collect statistics to test "
            "a conjecture before attempting a formal proof.",
        ),
        (
            "Perturbation / sensitivity analysis",
            "Apply a small perturbation δ to each input and measure ∂output/∂input; "
            "rank inputs by sensitivity to understand system behaviour.",
        ),
        (
            "Order-of-magnitude / Fermi estimate",
            "Build a Fermi estimate from first principles; bound the answer within "
            "one order of magnitude before committing to a detailed calculation.",
        ),
    ]

    # ------------------------------------------------------------------
    # Question classification — deterministic, no LLM required
    # ------------------------------------------------------------------

    @classmethod
    def classify_question(cls, question: str) -> QuestionExperimentability:
        """Classify whether *question* warrants experimental investigation.

        Uses a deterministic scoring algorithm based on regex pattern matching.
        The score is in ``[-1.0, +1.0]``; a question is considered
        experimentable when the score is ≥ ``_EXPERIMENTABLE_THRESHOLD``
        (0.15).

        This method requires **no LLM** — it runs entirely from keywords and
        syntax patterns so results are stable, fast, and unit-testable.

        Parameters
        ----------
        question:
            The raw question string to classify.

        Returns
        -------
        QuestionExperimentability
            Structured classification including score, type label, suggested
            experiment categories, and a plain-English reason.
        """
        score, question_type, categories, reason = cls._score_question(question)
        is_experimentable = score >= _EXPERIMENTABLE_THRESHOLD
        return QuestionExperimentability(
            question=question,
            is_experimentable=is_experimentable,
            score=round(score, 3),
            question_type=question_type,
            suggested_categories=categories,
            reason=reason,
        )

    @classmethod
    def _score_question(
        cls, question: str
    ) -> tuple[float, str, list[ExperimentCategory], str]:
        """Internal scoring engine.  Returns ``(score, type_label, categories, reason)``."""
        score = 0.0
        active_signals: list[str] = []

        # ── Positive signals ──────────────────────────────────────────
        if _QUANTITATIVE_RE.search(question):
            score += 0.30
            active_signals.append("quantitative")
        if _CAUSAL_EXPERIMENT_RE.search(question):
            score += 0.25
            active_signals.append("causal")
        if _HYPOTHESIS_RE.search(question):
            score += 0.25
            active_signals.append("hypothesis")
        if _EXPERIMENT_VERB_RE.search(question):
            score += 0.20
            active_signals.append("experiment-verb")
        if _OPTIMISATION_RE.search(question):
            score += 0.20
            active_signals.append("optimisation")
        if _PROBABILISTIC_RE.search(question):
            score += 0.25
            active_signals.append("probabilistic")
        if _COMPARATIVE_METRIC_RE.search(question):
            score += 0.20
            active_signals.append("comparative")

        # ── Negative signals ──────────────────────────────────────────
        # Strong definitional (requires explicit marker like "definition of")
        if _DEFINITIONAL_RE.search(question):
            score -= 0.40
            active_signals.append("definitional")
        elif _SIMPLE_WHAT_IS_RE.search(question):
            # Weak penalty: bare "What is X?" without a quantitative/probabilistic frame.
            # A single positive signal (e.g. probabilistic +0.25) overrides this.
            score -= 0.20
            active_signals.append("definitional")
        if _HISTORICAL_RE.search(question):
            score -= 0.30
            active_signals.append("historical")
        if _RECOMMENDATION_ONLY_RE.search(question):
            score -= 0.15
            active_signals.append("recommendation")

        score = round(max(-1.0, min(1.0, score)), 3)

        # ── Derive question type label ─────────────────────────────────
        positive = [s for s in active_signals if s not in ("definitional", "historical", "recommendation")]
        negative = [s for s in active_signals if s in ("definitional", "historical", "recommendation")]

        if not active_signals:
            question_type = "conceptual"
        elif "definitional" in negative and not positive:
            question_type = "definitional"
        elif "historical" in negative and not positive:
            question_type = "historical"
        elif "probabilistic" in positive:
            question_type = "probabilistic"
        elif "optimisation" in positive:
            question_type = "optimisation"
        elif "causal" in positive and "quantitative" in positive:
            question_type = "quantitative-causal"
        elif "causal" in positive:
            question_type = "causal-mechanistic"
        elif "quantitative" in positive:
            question_type = "quantitative"
        elif "hypothesis" in positive:
            question_type = "hypothesis-testing"
        elif "comparative" in positive:
            question_type = "comparative"
        elif "experiment-verb" in positive:
            question_type = "simulation-request"
        else:
            question_type = "conceptual"

        # ── Suggest experiment categories ──────────────────────────────
        categories: list[ExperimentCategory] = []
        if "quantitative" in positive or "causal" in positive:
            categories += [ExperimentCategory.NUMERIC_SWEEP, ExperimentCategory.PERTURBATION]
        if "probabilistic" in positive:
            categories += [ExperimentCategory.MONTE_CARLO]
        if "hypothesis" in positive:
            categories += [ExperimentCategory.TOY_ANALYTICAL, ExperimentCategory.COMBINATORIAL]
        if "optimisation" in positive:
            categories += [ExperimentCategory.NUMERIC_SWEEP, ExperimentCategory.PERTURBATION]
        if "comparative" in positive:
            categories += [ExperimentCategory.NUMERIC_SWEEP, ExperimentCategory.FERMI_ESTIMATE]
        if "experiment-verb" in positive:
            categories += [ExperimentCategory.FINITE_DIFFERENCE, ExperimentCategory.NUMERIC_SWEEP]
        # Deduplicate while preserving order
        seen: set[ExperimentCategory] = set()
        unique_categories: list[ExperimentCategory] = []
        for c in categories:
            if c not in seen:
                unique_categories.append(c)
                seen.add(c)
        if not unique_categories:
            unique_categories = [ExperimentCategory.FERMI_ESTIMATE, ExperimentCategory.TOY_ANALYTICAL]

        # ── Compose reason string ──────────────────────────────────────
        if score >= _EXPERIMENTABLE_THRESHOLD:
            reason_parts: list[str] = []
            if "quantitative" in positive:
                reason_parts.append("asks about measurable quantities")
            if "causal" in positive:
                reason_parts.append("explores a causal relationship")
            if "probabilistic" in positive:
                reason_parts.append("involves probability or randomness")
            if "optimisation" in positive:
                reason_parts.append("seeks an optimal value")
            if "hypothesis" in positive:
                reason_parts.append("poses a falsifiable hypothesis")
            if "comparative" in positive:
                reason_parts.append("requests a quantitative comparison")
            if "experiment-verb" in positive:
                reason_parts.append("explicitly requests simulation or testing")
            reason = "Experimentable: " + "; ".join(reason_parts) + f" (score={score:.2f})."
        else:
            reason_parts = []
            if "definitional" in negative:
                reason_parts.append("is a definition/conceptual question")
            if "historical" in negative:
                reason_parts.append("asks for a historical fact")
            if "recommendation" in negative:
                reason_parts.append("asks for a subjective recommendation")
            if not reason_parts:
                reason_parts.append("is conceptual with no measurable claim")
            reason = (
                "Not experimentable: question "
                + "; ".join(reason_parts)
                + f" (score={score:.2f})."
            )

        return score, question_type, unique_categories, reason

    # ------------------------------------------------------------------
    # Experiment planning — returns a structured ExperimentPlan
    # ------------------------------------------------------------------

    def plan_experiments(
        self,
        question: str,
        *,
        human_intuition: Optional[HumanIntuition] = None,
    ) -> ExperimentPlan:
        """Generate a structured set of experiments to answer *question*.

        This is the primary public method for the experiment-planning
        capability.  It first runs :meth:`classify_question` to decide
        whether experiments are warranted.  For non-experimentable questions
        it returns an ``ExperimentPlan`` with an empty ``experiments`` list
        and a clear ``synthesis_strategy`` explaining why.

        Parameters
        ----------
        question:
            The question to investigate experimentally.
        human_intuition:
            Optional prior intuition from the human.  When provided, each
            experiment's ``expected_outcome`` is framed in relation to what
            the human expects, enabling downstream calibration.

        Returns
        -------
        ExperimentPlan
            Structured plan containing 0–4 :class:`ExperimentSpec` objects,
            the experimentability classification, and a synthesis strategy.
        """
        experimentability = self.classify_question(question)

        if not experimentability.is_experimentable:
            return ExperimentPlan(
                question=question,
                experimentability=experimentability,
                experiments=[],
                synthesis_strategy=(
                    f"This question is best answered through direct expert analysis "
                    f"rather than computational experiments. "
                    f"Reason: {experimentability.reason} "
                    f"See the standard AgentResponse for a detailed expert answer."
                ),
            )

        # For experimentable questions: ask the LLM to produce structured specs.
        # Fall back to rule-based templates when the LLM is unavailable or
        # returns unparseable output.
        raw = self._call_llm_for_plan(question, experimentability, human_intuition)
        experiments = self._parse_experiment_specs(raw, experimentability)

        if not experiments:
            # LLM parse failed — generate deterministic fallback specs
            experiments = self._fallback_experiment_specs(question, experimentability)

        return ExperimentPlan(
            question=question,
            experimentability=experimentability,
            experiments=experiments,
            synthesis_strategy=self._build_synthesis_strategy(
                question, experimentability, experiments
            ),
        )

    # ------------------------------------------------------------------
    # Experiment planning helpers (internal)
    # ------------------------------------------------------------------

    def _call_llm_for_plan(
        self,
        question: str,
        experimentability: QuestionExperimentability,
        human_intuition: Optional[HumanIntuition],
    ) -> str:
        """Issue a structured experiment-planning prompt to the LLM backend."""
        system = self._build_experiment_planning_system_prompt(experimentability)
        user = self._build_experiment_planning_user_message(
            question, experimentability, human_intuition
        )
        try:
            return self._backend.generate(system, user, max_tokens=self._max_tokens)
        except Exception:
            return ""

    def _build_experiment_planning_system_prompt(
        self, experimentability: QuestionExperimentability
    ) -> str:
        cat_names = ", ".join(c.value for c in experimentability.suggested_categories[:3])
        return (
            "You are an expert computational scientist. "
            "Design a set of 2–4 lightweight, reproducible experiments to answer "
            "the user's question. Each experiment must test one falsifiable hypothesis "
            "and include a self-contained Python/NumPy snippet (≤40 lines, no external "
            "APIs, deterministic with fixed random seeds).\n\n"
            f"The question type is: {experimentability.question_type}. "
            f"Preferred experiment categories: {cat_names}.\n\n"
            "Respond ONLY with a JSON object matching this schema:\n"
            "{\n"
            '  "experiments": [\n'
            "    {\n"
            '      "id": "exp_1",\n'
            '      "category": "<one of: numeric_sweep|monte_carlo|toy_analytical|'
            "dimensional_scaling|finite_difference|combinatorial|perturbation|fermi_estimate>\",\n"
            '      "hypothesis": "<one-sentence falsifiable claim>",\n'
            '      "variables": {"independent": "...", "dependent": "...", "controlled": "..."},\n'
            '      "procedure": ["step 1", "step 2", "step 3"],\n'
            '      "python_snippet": "<self-contained runnable code>",\n'
            '      "expected_outcome": "<quantitative prediction>",\n'
            '      "disconfirmation": "<what result would refute the hypothesis>"\n'
            "    }\n"
            "  ],\n"
            '  "synthesis_strategy": "<how to combine results across all experiments>"\n'
            "}"
        )

    def _build_experiment_planning_user_message(
        self,
        question: str,
        experimentability: QuestionExperimentability,
        human_intuition: Optional[HumanIntuition],
    ) -> str:
        parts = [f"Question: {question}"]
        if human_intuition:
            parts.append(
                f"\nHuman intuition: {human_intuition.intuitive_answer} "
                f"(confidence: {human_intuition.confidence:.0%})"
            )
        parts.append(
            f"\nClassification: {experimentability.reason}\n"
            "Design 2–4 experiments that together comprehensively answer this question."
        )
        return "\n".join(parts)

    def _parse_experiment_specs(
        self,
        raw: str,
        experimentability: QuestionExperimentability,
    ) -> list[ExperimentSpec]:
        """Parse LLM JSON output into ``ExperimentSpec`` objects."""
        if not raw:
            return []
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            data = json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            return []

        default_cat = (
            experimentability.suggested_categories[0]
            if experimentability.suggested_categories
            else ExperimentCategory.NUMERIC_SWEEP
        )
        specs: list[ExperimentSpec] = []
        for item in data.get("experiments", []):
            try:
                cat = self._parse_category(str(item.get("category", "")), default_cat)
                specs.append(ExperimentSpec(
                    id=str(item.get("id", f"exp_{len(specs)+1}")),
                    category=cat,
                    hypothesis=str(item.get("hypothesis", "")),
                    variables=dict(item.get("variables", {})),
                    procedure=list(item.get("procedure", [])),
                    python_snippet=str(item.get("python_snippet", "")),
                    expected_outcome=str(item.get("expected_outcome", "")),
                    disconfirmation=str(item.get("disconfirmation", "")),
                ))
            except Exception:
                continue
        return specs

    @staticmethod
    def _parse_category(
        raw: str, default: ExperimentCategory
    ) -> ExperimentCategory:
        """Coerce *raw* to an ``ExperimentCategory``, falling back to *default*."""
        try:
            return ExperimentCategory(raw)
        except ValueError:
            return default

    def _fallback_experiment_specs(
        self,
        question: str,
        experimentability: QuestionExperimentability,
    ) -> list[ExperimentSpec]:
        """Generate deterministic template experiments when LLM output is unavailable.

        Each template targets one of the most common question types and
        produces genuinely runnable Python code so the plan is immediately
        useful even in offline/mock mode.
        """
        question_type = experimentability.question_type
        q_short = question[:60]
        categories = experimentability.suggested_categories

        primary_cat = categories[0] if categories else ExperimentCategory.NUMERIC_SWEEP
        secondary_cat = categories[1] if len(categories) > 1 else ExperimentCategory.PERTURBATION

        # ── Template A: numeric sweep / parameter variation ────────────
        spec_a = ExperimentSpec(
            id="exp_1",
            category=primary_cat,
            hypothesis=(
                f"The outcome of '{q_short}' varies systematically as the "
                "primary parameter is swept across its plausible range."
            ),
            variables={
                "independent": "primary parameter x (swept from 0.1 to 10.0 in 50 steps)",
                "dependent": "output metric y = f(x)",
                "controlled": "all other parameters held at their nominal values",
            },
            procedure=[
                "Define the function f(x) that maps the primary parameter to the outcome.",
                "Sweep x over 50 evenly-spaced values between 0.1 and 10.",
                "Record y = f(x) for each value.",
                "Plot y vs. x to identify the trend (linear, sub-linear, super-linear).",
                "Compare the observed trend to the initial intuition.",
            ],
            python_snippet=textwrap.dedent("""\
                import numpy as np

                # Parameter sweep — adapt f(x) to the specific question
                x = np.linspace(0.1, 10.0, 50)

                # Example: quadratic relationship (replace with actual model)
                def f(x):
                    return x ** 2  # hypothesis: quadratic growth

                y = f(x)

                print("x range  :", x[0], "->", x[-1])
                print("y range  :", round(y.min(), 3), "->", round(y.max(), 3))
                print("Growth   :", "super-linear" if y[-1]/y[0] > (x[-1]/x[0]) else "sub-linear")
                print("Peak idx :", int(np.argmax(y)), "at x =", round(x[np.argmax(y)], 3))
                """),
            expected_outcome=(
                "The output metric y increases monotonically with x; "
                "the exact growth rate reveals the functional form."
            ),
            disconfirmation=(
                "If y shows no systematic trend or decreases with x, "
                "the hypothesised relationship does not hold."
            ),
        )

        # ── Template B: perturbation / sensitivity analysis ────────────
        spec_b = ExperimentSpec(
            id="exp_2",
            category=secondary_cat,
            hypothesis=(
                "The output is most sensitive to small changes in the primary "
                "parameter; secondary parameters have negligible influence."
            ),
            variables={
                "independent": "perturbation δ applied to each input parameter",
                "dependent": "absolute change in output |Δy| per unit δ",
                "controlled": "perturbation magnitude δ = 0.01 (1% of nominal)",
            },
            procedure=[
                "Set all parameters to their nominal values.",
                "Perturb each parameter by δ = 0.01 one at a time.",
                "Record |Δy| for each perturbation.",
                "Rank parameters by sensitivity |Δy| / δ.",
                "Identify which parameter dominates the output variance.",
            ],
            python_snippet=textwrap.dedent("""\
                import numpy as np

                # Sensitivity analysis — adapt f() to the specific question
                nominal = np.array([1.0, 1.0, 1.0])  # [param1, param2, param3]
                delta = 0.01  # 1% perturbation

                def f(params):
                    # Replace with actual model; example: weighted sum
                    return np.dot(params, [3.0, 1.5, 0.5])

                baseline = f(nominal)
                sensitivities = []
                for i in range(len(nominal)):
                    perturbed = nominal.copy()
                    perturbed[i] += delta
                    sensitivity = abs(f(perturbed) - baseline) / delta
                    sensitivities.append((f"param_{i+1}", round(sensitivity, 4)))

                sensitivities.sort(key=lambda x: x[1], reverse=True)
                print("Baseline output:", round(baseline, 4))
                print("Sensitivity ranking:")
                for name, s in sensitivities:
                    print(f"  {name}: {s}")
                """),
            expected_outcome=(
                "Parameter 1 dominates sensitivity; others contribute less "
                "than 30% of the leading sensitivity."
            ),
            disconfirmation=(
                "If a secondary parameter shows sensitivity within 10% of the "
                "primary, the system is multi-dimensional and cannot be reduced "
                "to single-parameter analysis."
            ),
        )

        # ── Optional Template C: Monte Carlo (for probabilistic questions) ─
        specs = [spec_a, spec_b]

        if question_type == "probabilistic" or ExperimentCategory.MONTE_CARLO in categories:
            spec_c = ExperimentSpec(
                id="exp_3",
                category=ExperimentCategory.MONTE_CARLO,
                hypothesis=(
                    "The distribution of outcomes follows the analytically "
                    "predicted distribution; empirical statistics converge to "
                    "theoretical values with N ≥ 10,000 samples."
                ),
                variables={
                    "independent": "number of Monte Carlo samples N",
                    "dependent": "empirical mean, variance, and 95th-percentile",
                    "controlled": "random seed (42) for reproducibility",
                },
                procedure=[
                    "Set random seed to 42 for reproducibility.",
                    "Draw N = 10,000 samples from the assumed distribution.",
                    "Compute empirical mean, variance, and key quantiles.",
                    "Compare empirical statistics to theoretical predictions.",
                    "Report the 95% confidence interval for the mean.",
                ],
                python_snippet=textwrap.dedent("""\
                    import numpy as np

                    rng = np.random.default_rng(42)
                    N = 10_000

                    # Replace with actual distribution; example: sum of two uniform [0,1]
                    samples = rng.uniform(0, 1, N) + rng.uniform(0, 1, N)

                    mean  = samples.mean()
                    std   = samples.std()
                    p95   = np.percentile(samples, 95)

                    print(f"N         = {N:,}")
                    print(f"Mean      = {mean:.4f}  (theoretical: 1.0)")
                    print(f"Std dev   = {std:.4f}  (theoretical: {(2/12)**0.5:.4f})")
                    print(f"95th pct  = {p95:.4f}  (theoretical ≈ 1.73)")
                    print(f"95% CI    = [{mean - 1.96*std/N**0.5:.4f}, "
                          f"{mean + 1.96*std/N**0.5:.4f}]")
                    """),
                expected_outcome=(
                    "Empirical mean converges to the theoretical value within "
                    "±0.02 for N = 10,000."
                ),
                disconfirmation=(
                    "If empirical statistics deviate from theoretical predictions "
                    "by more than 3 standard errors the assumed distribution is wrong."
                ),
            )
            specs.append(spec_c)

        return specs

    def _build_synthesis_strategy(
        self,
        question: str,
        experimentability: QuestionExperimentability,
        experiments: list[ExperimentSpec],
    ) -> str:
        """Produce a plain-English strategy for combining experiment results."""
        n = len(experiments)
        cat_labels = ", ".join(e.category.value.replace("_", " ") for e in experiments)
        return (
            f"Run all {n} experiment(s) ({cat_labels}) and collect their outputs. "
            f"Compare the observed trends and values across experiments to check for "
            f"consistency: if all experiments point to the same conclusion the answer "
            f"is well-supported; diverging results indicate the question has multiple "
            f"regimes or the model needs refinement. "
            f"For a '{experimentability.question_type}' question, pay particular "
            f"attention to: "
            + {
                "quantitative": "the functional form of the relationship (linear, quadratic, logarithmic).",
                "quantitative-causal": "the direction, magnitude, and threshold of the causal effect.",
                "causal-mechanistic": "which perturbation direction most strongly affects the outcome.",
                "probabilistic": "the shape of the output distribution and convergence of statistics.",
                "optimisation": "the location and sharpness of the optimum.",
                "hypothesis-testing": "whether the hypothesis is supported or refuted across all small cases.",
                "comparative": "the effect size (not just direction) of the difference.",
                "simulation-request": "the stability and convergence of the simulation over time.",
            }.get(experimentability.question_type, "the trend and variability of results.")
        )

    # ------------------------------------------------------------------
    # Standard answer() override — embeds classification + plan summary
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        exp_catalog = "\n".join(
            f"  {i + 1}. [{et}] {desc}"
            for i, (et, desc) in enumerate(self.EXPERIMENT_TYPES)
        )
        return (
            "You are an expert experimental scientist and computational researcher "
            "specialised in designing **lightweight, reproducible experiments** that "
            "can be run locally without any external credentials or paid services.\n\n"
            "Your core competencies span:\n"
            "- Scientific method: hypothesis formation, variable isolation, "
            "falsifiability, and reproducibility\n"
            "- Numerical methods: Monte Carlo, finite-difference integration, "
            "parameter sweeps, sensitivity analysis\n"
            "- Analytical methods: dimensional analysis, perturbation theory, "
            "order-of-magnitude estimation\n"
            "- Experiment communication: protocol writing, result tables, "
            "run-me-now Python/NumPy snippets\n\n"
            "## Question Classification\n\n"
            "Before designing experiments ALWAYS classify the question:\n"
            "- If the question is quantitative, causal, probabilistic, "
            "hypothesis-bearing, or optimisation-oriented → design experiments.\n"
            "- If the question is purely definitional, historical, or asks for a "
            "subjective recommendation → explain directly without experiments.\n\n"
            "## Experiment Design Protocol (for experimentable questions)\n\n"
            "For every experimentable question follow this four-phase protocol:\n\n"
            "### Phase 1 — Elicit Intuition\n"
            "Note the user's intuition and confidence if provided.\n\n"
            "### Phase 2 — Hypothesis Formation\n"
            "Enumerate 2-4 **falsifiable hypotheses** ordered from most to least "
            "likely given domain knowledge. For each hypothesis state:\n"
            "- H_n: [one-sentence claim]\n"
            "- Prediction: [measurable expected outcome if H_n is true]\n"
            "- Disconfirmation: [what result would refute H_n]\n\n"
            "### Phase 3 — Experiment Design\n"
            "For **each hypothesis** select the most appropriate experiment type "
            "from the catalog below, then produce:\n"
            "a) **Experiment protocol** — step-by-step procedure in plain language\n"
            "b) **Variables** — independent, dependent, controlled\n"
            "c) **Expected outcome** — quantitative prediction\n"
            "d) **Runnable snippet** — self-contained Python/NumPy code the user "
            "can copy-paste and execute locally (no external APIs, no file I/O "
            "beyond stdout)\n\n"
            "### Phase 4 — Outcome Analysis\n"
            "After the experiment (or based on predicted outcomes):\n"
            "- State which hypothesis is supported / refuted\n"
            "- Compare the outcome to the user's initial intuition\n"
            "- Highlight where intuition was accurate, partially correct, or "
            "misleading, and explain *why*\n"
            "- Propose the next experiment iteration if uncertainty remains\n\n"
            "## Experiment-Type Catalog\n\n"
            f"{exp_catalog}\n\n"
            "## Constraints\n\n"
            "- All snippets MUST be self-contained and runnable with only the Python "
            "standard library + NumPy/SciPy (already in requirements.txt).\n"
            "- Never require external API keys, network access, or file downloads.\n"
            "- Keep each snippet under 40 lines for readability.\n"
            "- Prefer deterministic experiments (fix random seeds) so results are "
            "reproducible.\n\n"
            "Respond only with the requested JSON structure."
        )

    def _build_user_message(self, question: str, mcp_context: str) -> str:
        """Extend the base user message with experiment classification context."""
        experimentability = self.classify_question(question)
        parts = [f"Question: {question}"]

        # Embed the classification so the LLM tailors its response
        parts.append(
            f"\n[Experiment classification: {experimentability.question_type}, "
            f"score={experimentability.score:.2f}, "
            f"experimentable={experimentability.is_experimentable}]"
            f"\nReason: {experimentability.reason}"
        )

        if mcp_context:
            parts.append(f"\nAdditional context from the internet:\n{mcp_context}")
        parts.append(
            "\nPlease answer in the following JSON format:\n"
            "{\n"
            '  "answer": "<concise expert answer>",\n'
            '  "reasoning": "<step-by-step reasoning>",\n'
            '  "confidence": <0.0-1.0>,\n'
            '  "sources": ["<source or reference>", ...]\n'
            "}"
        )
        return "\n".join(parts)

