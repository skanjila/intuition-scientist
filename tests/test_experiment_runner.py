"""Tests for the enhanced ExperimentRunnerAgent.

Covers:
- Question classification (deterministic, no LLM needed):
    * Quantitative, causal, probabilistic, hypothesis-bearing, optimisation
      questions are classified as experimentable.
    * Definitional, historical, and recommendation-only questions are NOT.
- ExperimentPlan generation:
    * Non-experimentable questions → empty experiment list.
    * Experimentable questions → 2+ ExperimentSpec objects.
    * Each spec has required fields (id, category, hypothesis, snippet, etc.).
    * Python snippets in fallback specs are syntactically valid.
- plan_experiments() respects human intuition when provided.
- Standard answer() interface unchanged (backward compatibility).
- New data models: ExperimentCategory, QuestionExperimentability,
  ExperimentSpec, ExperimentPlan are importable from src.models.
"""

from __future__ import annotations

import ast
import json

import pytest

from src.agents.experiment_runner_agent import ExperimentRunnerAgent, EXPERIMENTABLE_THRESHOLD
from src.models import (
    AgentResponse,
    Domain,
    ExperimentCategory,
    ExperimentPlan,
    ExperimentSpec,
    HumanIntuition,
    QuestionExperimentability,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def agent() -> ExperimentRunnerAgent:
    return ExperimentRunnerAgent(mcp_client=None)


# ---------------------------------------------------------------------------
# Data model imports and structure
# ---------------------------------------------------------------------------


class TestDataModels:
    def test_experiment_category_values(self):
        expected = {
            "numeric_sweep", "monte_carlo", "toy_analytical",
            "dimensional_scaling", "finite_difference", "combinatorial",
            "perturbation", "fermi_estimate", "not_applicable",
        }
        actual = {c.value for c in ExperimentCategory}
        assert expected == actual

    def test_question_experimentability_fields(self):
        qe = QuestionExperimentability(
            question="test?",
            is_experimentable=True,
            score=0.5,
            question_type="quantitative",
            suggested_categories=[ExperimentCategory.NUMERIC_SWEEP],
            reason="test reason",
        )
        assert qe.question == "test?"
        assert qe.is_experimentable is True
        assert qe.score == 0.5
        assert qe.question_type == "quantitative"
        assert qe.suggested_categories == [ExperimentCategory.NUMERIC_SWEEP]
        assert qe.reason == "test reason"

    def test_experiment_spec_fields(self):
        spec = ExperimentSpec(
            id="exp_1",
            category=ExperimentCategory.NUMERIC_SWEEP,
            hypothesis="x grows quadratically",
            variables={"independent": "x", "dependent": "y", "controlled": "none"},
            procedure=["step 1", "step 2"],
            python_snippet="print('hello')",
            expected_outcome="y ~ x^2",
            disconfirmation="y is constant",
        )
        assert spec.id == "exp_1"
        assert spec.category == ExperimentCategory.NUMERIC_SWEEP

    def test_experiment_plan_fields(self):
        qe = QuestionExperimentability(
            question="test?", is_experimentable=False, score=-0.3,
            question_type="definitional", suggested_categories=[],
            reason="definitional"
        )
        plan = ExperimentPlan(
            question="test?",
            experimentability=qe,
            experiments=[],
            synthesis_strategy="N/A",
        )
        assert plan.question == "test?"
        assert plan.experiments == []


# ---------------------------------------------------------------------------
# Question classification — experimentable cases
# ---------------------------------------------------------------------------


class TestClassifyQuestionExperimentable:
    """Questions that SHOULD be classified as experimentable."""

    @pytest.mark.parametrize("question", [
        "How does the learning rate affect convergence speed of gradient descent?",
        "What is the effect of batch size on training loss?",
        "How does the convergence rate change as the step size decreases?",
        "Does momentum cause gradient descent to converge faster on quadratic loss?",
        "Is it true that doubling the learning rate halves the convergence time?",
        "What is the probability of getting heads at least 7 times in 10 fair coin flips?",
        "Simulate the spread of an epidemic using an SIR model",
        "What is the expected value of the sum of two fair dice?",
        "How does sorting algorithm runtime scale with input size N?",
        "Compare the convergence of SGD versus Adam in terms of loss reduction rate",
        "Find the optimal regularisation parameter that minimises validation loss",
        "What percentage of variance does the first principal component capture?",
        "How does the number of layers affect accuracy on MNIST?",
        "Verify that the central limit theorem holds for exponential distributions",
    ])
    def test_is_experimentable(self, question):
        result = ExperimentRunnerAgent.classify_question(question)
        assert result.is_experimentable, (
            f"Expected '{question}' to be experimentable, "
            f"got score={result.score:.3f}, reason='{result.reason}'"
        )
        assert result.score >= EXPERIMENTABLE_THRESHOLD

    def test_quantitative_question_type(self):
        q = "How does training loss change as learning rate increases from 0.001 to 0.1?"
        result = ExperimentRunnerAgent.classify_question(q)
        assert "quantitative" in result.question_type

    def test_probabilistic_question_type(self):
        q = "What is the probability that a random walk returns to origin within 100 steps?"
        result = ExperimentRunnerAgent.classify_question(q)
        assert result.question_type == "probabilistic"
        assert ExperimentCategory.MONTE_CARLO in result.suggested_categories

    def test_causal_question_type(self):
        q = "What is the effect of dropout on generalisation error?"
        result = ExperimentRunnerAgent.classify_question(q)
        assert "causal" in result.question_type

    def test_optimisation_question_type(self):
        q = "Find the optimal learning rate that minimises the final training loss"
        result = ExperimentRunnerAgent.classify_question(q)
        assert result.question_type == "optimisation"

    def test_hypothesis_testing_type(self):
        q = "Is it true that L2 regularisation always reduces overfitting?"
        result = ExperimentRunnerAgent.classify_question(q)
        assert result.question_type == "hypothesis-testing"
        assert ExperimentCategory.TOY_ANALYTICAL in result.suggested_categories

    def test_simulation_request_type(self):
        q = "Simulate the trajectory of a ball thrown at 45 degrees"
        result = ExperimentRunnerAgent.classify_question(q)
        assert result.is_experimentable

    def test_suggested_categories_nonempty(self):
        q = "How does the error decrease as we increase the number of training samples?"
        result = ExperimentRunnerAgent.classify_question(q)
        assert len(result.suggested_categories) >= 1
        assert ExperimentCategory.NOT_APPLICABLE not in result.suggested_categories

    def test_reason_mentions_experimentable(self):
        q = "How does momentum affect convergence speed?"
        result = ExperimentRunnerAgent.classify_question(q)
        assert "experimentable" in result.reason.lower() or result.score >= EXPERIMENTABLE_THRESHOLD


# ---------------------------------------------------------------------------
# Question classification — non-experimentable cases
# ---------------------------------------------------------------------------


class TestClassifyQuestionNonExperimentable:
    """Questions that should NOT be classified as experimentable."""

    @pytest.mark.parametrize("question", [
        "What is gradient descent?",
        "What is the definition of overfitting?",
        "Define backpropagation",
        "What does 'learning rate' mean in machine learning?",
        "When was the transformer architecture invented?",
        "Who invented the backpropagation algorithm?",
        "What year was GPT-2 released?",
        "What is the history of neural networks?",
        "Should I use PyTorch or TensorFlow for my project?",
        "Which deep learning framework is best for beginners?",
    ])
    def test_not_experimentable(self, question):
        result = ExperimentRunnerAgent.classify_question(question)
        assert not result.is_experimentable, (
            f"Expected '{question}' NOT to be experimentable, "
            f"got score={result.score:.3f}, reason='{result.reason}'"
        )

    def test_definitional_question_type(self):
        q = "What is the definition of entropy in information theory?"
        result = ExperimentRunnerAgent.classify_question(q)
        assert result.question_type == "definitional"

    def test_historical_question_type(self):
        q = "When was the transformer architecture first published?"
        result = ExperimentRunnerAgent.classify_question(q)
        assert result.question_type == "historical"

    def test_reason_mentions_not_experimentable(self):
        q = "What is the definition of backpropagation?"
        result = ExperimentRunnerAgent.classify_question(q)
        assert "not experimentable" in result.reason.lower()


# ---------------------------------------------------------------------------
# Score properties
# ---------------------------------------------------------------------------


class TestClassificationScoreProperties:
    def test_score_in_valid_range(self):
        for q in [
            "What is X?",
            "How does X vary with Y?",
            "Simulate the system with N=100 particles",
        ]:
            result = ExperimentRunnerAgent.classify_question(q)
            assert -1.0 <= result.score <= 1.0, f"Score out of range for: {q}"

    def test_more_signals_higher_score(self):
        """A question with more positive signals should score higher."""
        simple_q = "How does X change with Y?"
        richer_q = "How does the convergence rate change as step size varies? Simulate and compute the probability distribution of outcomes."
        score_simple = ExperimentRunnerAgent.classify_question(simple_q).score
        score_rich = ExperimentRunnerAgent.classify_question(richer_q).score
        assert score_rich >= score_simple, (
            f"Richer question should score higher: {score_rich:.3f} vs {score_simple:.3f}"
        )

    def test_definitional_question_has_negative_score(self):
        result = ExperimentRunnerAgent.classify_question("What is the definition of entropy?")
        assert result.score < 0


# ---------------------------------------------------------------------------
# ExperimentPlan generation — non-experimentable path
# ---------------------------------------------------------------------------


class TestPlanExperimentsNonExperimentable:
    def test_empty_experiments_for_definition(self, agent):
        plan = agent.plan_experiments("What is the definition of backpropagation?")
        assert isinstance(plan, ExperimentPlan)
        assert plan.experiments == []
        assert not plan.experimentability.is_experimentable

    def test_synthesis_strategy_explains_why(self, agent):
        plan = agent.plan_experiments("When was GPT-2 released?")
        assert len(plan.synthesis_strategy) > 0
        assert "direct" in plan.synthesis_strategy.lower() or "analysis" in plan.synthesis_strategy.lower()

    def test_experimentability_stored_on_plan(self, agent):
        q = "What is the definition of dropout?"
        plan = agent.plan_experiments(q)
        assert plan.experimentability.question == q
        assert plan.experimentability.is_experimentable is False


# ---------------------------------------------------------------------------
# ExperimentPlan generation — experimentable path (fallback specs)
# ---------------------------------------------------------------------------


class TestPlanExperimentsExperimentable:
    def test_returns_experiment_plan(self, agent):
        plan = agent.plan_experiments(
            "How does learning rate affect the convergence speed of gradient descent?"
        )
        assert isinstance(plan, ExperimentPlan)

    def test_has_experiments(self, agent):
        plan = agent.plan_experiments(
            "How does learning rate affect the convergence speed of gradient descent?"
        )
        assert len(plan.experiments) >= 2

    def test_experiments_have_required_fields(self, agent):
        plan = agent.plan_experiments(
            "How does batch size affect training loss on a simple regression?"
        )
        for spec in plan.experiments:
            assert isinstance(spec.id, str) and spec.id
            assert isinstance(spec.category, ExperimentCategory)
            assert isinstance(spec.hypothesis, str) and spec.hypothesis
            assert isinstance(spec.variables, dict)
            assert isinstance(spec.procedure, list) and len(spec.procedure) >= 1
            assert isinstance(spec.python_snippet, str) and spec.python_snippet
            assert isinstance(spec.expected_outcome, str) and spec.expected_outcome
            assert isinstance(spec.disconfirmation, str) and spec.disconfirmation

    def test_experiment_ids_are_unique(self, agent):
        plan = agent.plan_experiments(
            "How does the regularisation coefficient affect model generalisation?"
        )
        ids = [spec.id for spec in plan.experiments]
        assert len(ids) == len(set(ids)), "Experiment IDs must be unique"

    def test_python_snippets_are_syntactically_valid(self, agent):
        plan = agent.plan_experiments(
            "How does step size affect the numerical integration error in Euler's method?"
        )
        for spec in plan.experiments:
            if spec.python_snippet:
                try:
                    ast.parse(spec.python_snippet)
                except SyntaxError as e:
                    pytest.fail(
                        f"Snippet for {spec.id} has a syntax error: {e}\n"
                        f"Snippet:\n{spec.python_snippet}"
                    )

    def test_synthesis_strategy_nonempty(self, agent):
        plan = agent.plan_experiments(
            "How does the number of hidden units affect validation accuracy?"
        )
        assert len(plan.synthesis_strategy) > 30

    def test_experimentability_attached(self, agent):
        plan = agent.plan_experiments(
            "What is the effect of dropout rate on generalisation error?"
        )
        assert plan.experimentability.is_experimentable is True
        assert plan.experimentability.score >= EXPERIMENTABLE_THRESHOLD


# ---------------------------------------------------------------------------
# plan_experiments with human intuition
# ---------------------------------------------------------------------------


class TestPlanExperimentsWithHumanIntuition:
    def test_intuition_does_not_break_plan(self, agent):
        hi = HumanIntuition(
            question="How does learning rate affect convergence?",
            intuitive_answer="Higher learning rate converges faster but may overshoot.",
            confidence=0.7,
        )
        plan = agent.plan_experiments(
            "How does learning rate affect convergence?",
            human_intuition=hi,
        )
        assert isinstance(plan, ExperimentPlan)
        assert len(plan.experiments) >= 2

    def test_none_intuition_is_accepted(self, agent):
        plan = agent.plan_experiments(
            "How does step size affect integration accuracy?",
            human_intuition=None,
        )
        assert isinstance(plan, ExperimentPlan)


# ---------------------------------------------------------------------------
# Monte Carlo plan — probabilistic questions
# ---------------------------------------------------------------------------


class TestProbabilisticQuestionPlan:
    def test_monte_carlo_in_categories(self):
        q = "What is the probability that a random walk returns to origin within 50 steps?"
        result = ExperimentRunnerAgent.classify_question(q)
        assert ExperimentCategory.MONTE_CARLO in result.suggested_categories

    def test_plan_includes_monte_carlo_spec(self, agent):
        plan = agent.plan_experiments(
            "What is the expected value of the maximum of 10 uniform random variables?"
        )
        categories = [spec.category for spec in plan.experiments]
        assert ExperimentCategory.MONTE_CARLO in categories, (
            f"Expected MONTE_CARLO in plan categories, got: {categories}"
        )


# ---------------------------------------------------------------------------
# Backward-compatibility: standard answer() still works
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_answer_returns_agent_response(self, agent):
        resp = agent.answer("What is gradient descent?")
        assert isinstance(resp, AgentResponse)
        assert resp.domain == Domain.EXPERIMENT_RUNNER
        assert isinstance(resp.answer, str) and len(resp.answer) > 0
        assert 0.0 <= resp.confidence <= 1.0

    def test_answer_weights_sum_to_one(self, agent):
        resp = agent.answer("How does learning rate affect convergence?")
        assert abs(resp.intuition_weight + resp.tool_weight - 1.0) < 1e-9

    def test_answer_sources_is_list(self, agent):
        resp = agent.answer("Simulate gradient descent on a quadratic loss")
        assert isinstance(resp.sources, list)

    def test_answer_still_intuition_heavy(self, agent):
        """ExperimentRunnerAgent is in INTUITION_HEAVY set; weight > 0.5 without MCP."""
        resp = agent.answer(
            "Why does adding momentum to gradient descent speed up convergence?"
        )
        assert resp.intuition_weight > 0.5, (
            f"Expected intuition_weight > 0.5, got {resp.intuition_weight}"
        )

    def test_domain_is_experiment_runner(self, agent):
        assert agent.domain == Domain.EXPERIMENT_RUNNER


# ---------------------------------------------------------------------------
# System prompt content
# ---------------------------------------------------------------------------


class TestSystemPromptContent:
    def test_prompt_has_classification_section(self, agent):
        prompt = agent._build_system_prompt()
        assert "Question Classification" in prompt

    def test_prompt_has_all_phases(self, agent):
        prompt = agent._build_system_prompt()
        for phase in ("Phase 1", "Phase 2", "Phase 3", "Phase 4"):
            assert phase in prompt, f"Phase '{phase}' missing from system prompt"

    def test_prompt_has_experiment_catalog(self, agent):
        prompt = agent._build_system_prompt()
        assert "Experiment-Type Catalog" in prompt

    def test_prompt_describes_experimentable_vs_not(self, agent):
        prompt = agent._build_system_prompt()
        assert "definitional" in prompt.lower() or "definition" in prompt.lower()
        assert "experimentable" in prompt.lower() or "quantitative" in prompt.lower()


# ---------------------------------------------------------------------------
# _build_user_message embeds classification
# ---------------------------------------------------------------------------


class TestUserMessageEmbedding:
    def test_classification_embedded_in_user_message(self, agent):
        msg = agent._build_user_message(
            "How does learning rate affect convergence?", ""
        )
        assert "Experiment classification" in msg
        assert "score=" in msg

    def test_experimentable_flag_in_message(self, agent):
        msg = agent._build_user_message(
            "What is the probability of heads 7 times in 10 tosses?", ""
        )
        assert "experimentable=True" in msg

    def test_not_experimentable_flag_in_message(self, agent):
        msg = agent._build_user_message(
            "What is the definition of entropy?", ""
        )
        assert "experimentable=False" in msg
