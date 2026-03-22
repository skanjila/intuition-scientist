"""Shared data models for the Human Intuition Scientist."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Domain(str, Enum):
    """Scientific / engineering domains covered by the system."""

    ELECTRICAL_ENGINEERING = "electrical_engineering"
    COMPUTER_SCIENCE = "computer_science"
    NEURAL_NETWORKS = "neural_networks"
    SOCIAL_SCIENCE = "social_science"
    SPACE_SCIENCE = "space_science"
    PHYSICS = "physics"
    DEEP_LEARNING = "deep_learning"
    # High-economic-value industry domains
    HEALTHCARE = "healthcare"
    CLIMATE_ENERGY = "climate_energy"
    FINANCE_ECONOMICS = "finance_economics"
    CYBERSECURITY = "cybersecurity"
    BIOTECH_GENOMICS = "biotech_genomics"
    SUPPLY_CHAIN = "supply_chain"
    # Enterprise problem domains
    LEGAL_COMPLIANCE = "legal_compliance"
    ENTERPRISE_ARCHITECTURE = "enterprise_architecture"
    MARKETING_GROWTH = "marketing_growth"
    ORGANIZATIONAL_BEHAVIOR = "organizational_behavior"
    STRATEGY_INTELLIGENCE = "strategy_intelligence"
    # Mastery domains
    ALGORITHMS_PROGRAMMING = "algorithms_programming"
    # Interview preparation
    INTERVIEW_PREP = "interview_prep"
    # PhD research domains
    EE_LLM_RESEARCH = "ee_llm_research"
    # Signal processing (dedicated iterative-problem agent)
    SIGNAL_PROCESSING = "signal_processing"
    # Experiment runner (experiment-protocol and simulation agent)
    EXPERIMENT_RUNNER = "experiment_runner"


@dataclass
class HumanIntuition:
    """Structured representation of a human's intuitive answer."""

    question: str
    intuitive_answer: str
    # 0.0 (wild guess) → 1.0 (highly confident)
    confidence: float
    reasoning: str = ""
    domain_guesses: list[Domain] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")


@dataclass
class AgentResponse:
    """A domain-specific agent's answer to the question."""

    domain: Domain
    answer: str
    reasoning: str
    confidence: float
    sources: list[str] = field(default_factory=list)
    mcp_context: str = ""
    #: Weight given to pure-intuition (knowledge-only) pipeline (0.0–1.0)
    intuition_weight: float = 0.5
    #: Weight given to MCP/tool-grounded pipeline (0.0–1.0)
    tool_weight: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")


@dataclass
class AlignmentScore:
    """Semantic alignment between the human intuition and one agent response."""

    domain: Domain
    # 0.0 = completely divergent, 1.0 = perfect alignment
    semantic_similarity: float
    key_agreements: list[str] = field(default_factory=list)
    key_divergences: list[str] = field(default_factory=list)
    intuition_insight: str = ""


@dataclass
class WeighingResult:
    """Full cross-agent weighing of human intuition vs. expert answers."""

    question: str
    human_intuition: HumanIntuition
    agent_responses: list[AgentResponse]
    alignment_scores: list[AlignmentScore]
    # Weighted blend of human intuition and agent consensus
    synthesized_answer: str
    # How well the human's intuition held up overall (0-100 %)
    intuition_accuracy_pct: float
    overall_analysis: str
    recommendations: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """A single web-search result returned by the MCP client."""

    title: str
    url: str
    snippet: str
    relevance_score: Optional[float] = None


# ---------------------------------------------------------------------------
# Experiment runner models
# ---------------------------------------------------------------------------


class ExperimentCategory(str, Enum):
    """Taxonomy of lightweight experiment types the ExperimentRunnerAgent uses.

    Each value corresponds to one of the canonical experiment types defined in
    ``ExperimentRunnerAgent.EXPERIMENT_TYPES``.  ``NOT_APPLICABLE`` is used
    when a question is classified as non-experimentable.
    """

    NUMERIC_SWEEP = "numeric_sweep"
    MONTE_CARLO = "monte_carlo"
    TOY_ANALYTICAL = "toy_analytical"
    DIMENSIONAL_SCALING = "dimensional_scaling"
    FINITE_DIFFERENCE = "finite_difference"
    COMBINATORIAL = "combinatorial"
    PERTURBATION = "perturbation"
    FERMI_ESTIMATE = "fermi_estimate"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class QuestionExperimentability:
    """Classification of whether a question warrants experimental investigation.

    Produced by :meth:`ExperimentRunnerAgent.classify_question` using
    deterministic rule-based scoring (no LLM required).

    Attributes
    ----------
    question:
        The original question that was classified.
    is_experimentable:
        ``True`` when the question can be meaningfully answered (at least in
        part) through lightweight computational experiments.
    score:
        Raw classification score in ``[-1.0, +1.0]``.  Positive values
        indicate evidence for experimentability; negative values indicate the
        question is better served by direct analysis.
    question_type:
        Human-readable label for the dominant question category, e.g.
        ``"quantitative-causal"``, ``"probabilistic"``, ``"definitional"``.
    suggested_categories:
        Ordered list of the most appropriate ``ExperimentCategory`` values for
        this question (most relevant first).
    reason:
        Short plain-English explanation of why the question was classified
        as experimentable or not.
    """

    question: str
    is_experimentable: bool
    score: float
    question_type: str
    suggested_categories: list[ExperimentCategory]
    reason: str


@dataclass
class ExperimentSpec:
    """Specification for one targeted experiment within an experiment plan.

    Each ``ExperimentSpec`` is self-contained: a reader can reproduce the
    experiment using only the fields here without any external dependencies.

    Attributes
    ----------
    id:
        Short identifier, e.g. ``"exp_1"`` or ``"monte_carlo_baseline"``.
    category:
        The experiment type drawn from ``ExperimentCategory``.
    hypothesis:
        One falsifiable claim this experiment tests (plain English).
    variables:
        Mapping with keys ``"independent"``, ``"dependent"``, and
        ``"controlled"`` describing the experimental variables.
    procedure:
        Ordered list of plain-English steps to run the experiment.
    python_snippet:
        Self-contained, runnable Python/NumPy code implementing the
        experiment.  Must use only the standard library + NumPy/SciPy.
        Must be deterministic (random seeds fixed).  Max ~40 lines.
    expected_outcome:
        Quantitative prediction of what the snippet will show.
    disconfirmation:
        What result would *refute* the hypothesis.
    """

    id: str
    category: ExperimentCategory
    hypothesis: str
    variables: dict[str, str]
    procedure: list[str]
    python_snippet: str
    expected_outcome: str
    disconfirmation: str


@dataclass
class ExperimentPlan:
    """A structured set of experiments designed to answer a question.

    Produced by :meth:`ExperimentRunnerAgent.plan_experiments`.

    Attributes
    ----------
    question:
        The original question the plan addresses.
    experimentability:
        Classification metadata explaining *why* experiments were or were not
        proposed.
    experiments:
        Ordered list of :class:`ExperimentSpec` objects.  Empty when
        ``experimentability.is_experimentable`` is ``False``.
    synthesis_strategy:
        Plain-English description of how to interpret and combine results
        across all experiments to reach an overall conclusion.
    """

    question: str
    experimentability: QuestionExperimentability
    experiments: list[ExperimentSpec]
    synthesis_strategy: str


# ---------------------------------------------------------------------------
# Debate harness models
# ---------------------------------------------------------------------------


@dataclass
class DebatePosition:
    """One participant's position in the multi-party debate.

    A position can come from the human, from a domain agent, or from
    MCP/tool-based evidence gathered at runtime.
    """

    #: Identifies the source: ``"human"``, ``"tool"``, or ``"agent:<domain>"``
    source: str
    #: The substantive position or argument
    position: str
    #: 0.0–1.0 confidence in this position
    confidence: float
    #: Supporting evidence, citations, or references
    evidence: list[str] = field(default_factory=list)


@dataclass
class DebateRound:
    """One structured round of the debate, focusing on a specific aspect.

    Rounds surface where all participants agree vs. diverge, making the
    human–machine–tool disagreement visible and auditable.
    """

    #: The specific question or dimension being examined this round
    aspect: str
    #: All positions put forward in this round
    positions: list[DebatePosition]
    #: Concepts/claims all participants converge on
    agreements: list[str] = field(default_factory=list)
    #: Claims where human and agents/tools diverge (``"<source>: <claim>"``)
    disagreements: list[str] = field(default_factory=list)
    #: Brief synthesis of what this round established
    round_synthesis: str = ""


@dataclass
class DebateResult:
    """Full structured debate outcome.

    Captures the multi-party exchange between human intuition, MCP/tool
    evidence, and domain-agent reasoning, then produces a moderated verdict.
    """

    question: str
    human_position: DebatePosition
    #: Evidence gathered by MCP tools/web search (empty when MCP disabled)
    tool_evidence: DebatePosition
    #: One position per domain agent that was invoked
    agent_positions: list[DebatePosition]
    #: Structured rounds (one per analytical dimension of the question)
    rounds: list[DebateRound]
    #: Final synthesised answer from all perspectives
    synthesized_verdict: str
    #: Confidence in the synthesised verdict (0.0–1.0)
    verdict_confidence: float
    #: How well human intuition held up (0–100 %)
    intuition_accuracy_pct: float
    #: Key takeaways from the debate
    key_insights: list[str] = field(default_factory=list)
    #: Recommendations for the human based on debate outcome
    recommendations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Interview coaching models
# ---------------------------------------------------------------------------


@dataclass
class InterviewResult:
    """Full FAANG interview coaching result.

    Produced by :meth:`AgentOrchestrator.interview_prep` which combines
    three agents: InterviewPrepAgent (technical), AlgorithmsProgrammingAgent
    (algo/language depth), and SocialScienceAgent (mental readiness).
    """

    question: str
    #: The candidate's own answer
    candidate_answer: str
    #: Candidate's self-reported confidence (0.0–1.0)
    candidate_confidence: float
    #: How well candidate answer aligned with expert consensus (0.0–1.0)
    technical_score: float
    #: InterviewPrepAgent's technical evaluation and optimal solution
    technical_feedback: str
    #: InterviewPrepAgent's step-by-step reasoning
    technical_reasoning: str
    #: AlgorithmsProgrammingAgent's deep algorithmic / language insight
    algorithmic_insight: str
    #: SocialScienceAgent's mental prep coaching (stress, communication, STAR)
    mental_preparation: str
    #: Synthesised overall analysis from all three agents
    overall_analysis: str
    #: Best synthesised answer blending all perspectives
    synthesized_answer: str
    #: Actionable improvement recommendations
    recommendations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Model evaluation / cycling models
# ---------------------------------------------------------------------------


@dataclass
class ModelRunResult:
    """Result from running the full pipeline with a single model backend."""

    #: Provider spec string, e.g. ``"ollama:llama3.1:8b"`` or ``"mock"``
    model_spec: str
    #: Whether the backend was reachable
    backend_available: bool
    #: The full weighing result (``None`` when backend was unavailable)
    weighing_result: Optional[WeighingResult]
    #: Error message if the run failed
    error: Optional[str] = None
    #: Wall-clock time for this run in seconds
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Workflow trace / agentic-visibility models
# ---------------------------------------------------------------------------


class WorkflowMapMode(str, Enum):
    """Controls how much workflow detail is appended to each answer."""

    OFF = "off"
    COMPACT = "compact"
    STANDARD = "standard"
    DEEP = "deep"


@dataclass
class WorkflowStep:
    """One step in the agentic workflow trace."""

    #: Short label shown in the Mermaid diagram
    label: str
    #: Human-readable description of what happened / why
    description: str = ""
    #: Optional tool name used in this step (e.g. ``"mcp_search"``)
    tool: str = ""
    #: Summary of the tool result (no raw secrets)
    tool_result_summary: str = ""


@dataclass
class WorkflowTrace:
    """Structured trace of the agentic reasoning workflow for one request.

    This is produced *after* the orchestrator completes a run so that no
    chain-of-thought is exposed—only an explainability summary.
    """

    question: str
    #: Ordered steps the system took (used to build the Mermaid diagram)
    steps: list[WorkflowStep] = field(default_factory=list)
    #: Key inputs fed into the pipeline (question, domains, settings)
    inputs_context: list[str] = field(default_factory=list)
    #: Explicit assumptions the system made
    assumptions: list[str] = field(default_factory=list)
    #: High-level numbered plan
    plan: list[str] = field(default_factory=list)
    #: Tool-call entries: ``(tool_name, reason, result_summary)``
    tool_calls: list[tuple[str, str, str]] = field(default_factory=list)
    #: Intermediate artifacts: checklists, tables, acceptance criteria
    intermediate_artifacts: list[str] = field(default_factory=list)
    #: Suggested next actions for the user
    next_actions: list[str] = field(default_factory=list)


@dataclass
class ModelEvaluationResult:
    """Cross-model evaluation summary produced by
    :meth:`AgentOrchestrator.evaluate_models`.

    Cycles through a list of model backends, runs the full dual-pipeline
    (human intuition + MCP tools), and compares results to surface:
    - which models agree with the human intuition best,
    - where models diverge from each other,
    - what the multi-model consensus answer is.
    """

    question: str
    #: Individual result per model
    model_results: list[ModelRunResult]
    #: Answer text that the majority of models converged on
    consensus_answer: str
    #: Summary of where models disagreed
    divergence_summary: str
    #: Spec of the model that achieved the highest intuition accuracy
    best_model_spec: str
    #: Number of model specs evaluated
    models_evaluated: int
    #: Number of models that were actually reachable
    models_available: int
    #: Mean intuition accuracy across available models (0–100 %)
    mean_intuition_accuracy_pct: float
