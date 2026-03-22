"""Large test suite — dual-pipeline agents across all domains.

Tests the complete system:
- Dual-pipeline (intuition + MCP) weight computation per domain
- All domain agents produce structured AgentResponse with weight fields
- Debate engine produces DebateResult with rounds, positions, verdict
- Interview prep mode returns InterviewResult with all three agents
- Model cycling returns ModelEvaluationResult (gated: RUN_MODEL_SWEEP=1)
- Complex open-ended questions across every domain

Environment variables
---------------------
RUN_MODEL_SWEEP=1
    Enable model cycling tests (disabled by default in CI).
INTUITION_SCIENTIST_MODELS
    Comma-separated model spec list for cycling tests.
    Defaults to ``mock`` (always offline-safe).
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from src.agents.base_agent import _FACTUAL_RE, _ANALYTICAL_RE, _INTUITION_HEAVY, _TOOL_HEAVY
from src.llm.mock_backend import MockBackend
from src.models import (
    AgentResponse,
    Domain,
    HumanIntuition,
    ModelEvaluationResult,
)
from src.orchestrator.agent_orchestrator import AgentOrchestrator, _AGENT_CLASSES

# ---------------------------------------------------------------------------
# Fixtures — one complex open-ended question per domain
# ---------------------------------------------------------------------------

DOMAIN_QUESTIONS: dict[Domain, tuple[str, str, str]] = {
    # (question, intuitive_answer, reasoning)
    Domain.ELECTRICAL_ENGINEERING: (
        "How do switched-mode power supplies achieve high efficiency?",
        "By rapidly switching transistors to minimise resistive losses, using inductors and capacitors to store and transfer energy rather than dissipating it as heat.",
        "Linear regulators drop excess voltage as heat. SMPS convert voltage through high-frequency switching, achieving 85-95% efficiency.",
    ),
    Domain.COMPUTER_SCIENCE: (
        "Why is the P vs NP problem so central to computer science?",
        "It asks whether every problem whose solution can be quickly verified can also be quickly solved. If P=NP, cryptography and optimisation as we know them would collapse.",
        "The distinction between polynomial and exponential time underpins our understanding of tractability. Most cryptography relies on NP problems being hard to solve.",
    ),
    Domain.NEURAL_NETWORKS: (
        "What is the vanishing gradient problem and how do modern architectures address it?",
        "During backpropagation through deep networks, gradients shrink exponentially, making early layers learn very slowly. Solutions include residual connections, layer normalisation, and gated architectures.",
        "Sigmoid/tanh activations compress gradients. ReLU, batch norm, and skip connections in ResNets keep gradients healthy across 100+ layers.",
    ),
    Domain.SOCIAL_SCIENCE: (
        "How do social norms emerge and self-enforce in the absence of formal institutions?",
        "Through repeated interactions, reputation systems, and coordinated punishment of norm violators. Game-theoretic mechanisms like tit-for-tat enable cooperation without central authority.",
        "Experimental economics (Fehr & Gächter) shows humans punish cheaters even at personal cost. Repeated games create folk-theorem equilibria.",
    ),
    Domain.SPACE_SCIENCE: (
        "What are the most critical engineering challenges for a crewed Mars mission?",
        "Radiation exposure during transit, reliable life support for 30+ months, psychological isolation, landing on thin Martian atmosphere, and in-situ resource utilisation for return propellant.",
        "Mars is 6-9 months away. The Martian atmosphere is 1% of Earth's density. Galactic cosmic rays increase cancer risk. Communication latency reaches 24 minutes.",
    ),
    Domain.PHYSICS: (
        "Why does quantum entanglement not allow faster-than-light communication?",
        "While measuring one entangled particle instantly determines the state of the other, the outcome is random and cannot be controlled by the sender, so no information is transmitted.",
        "No-communication theorem: you cannot choose what outcome you get. The correlation is only apparent after classical comparison of results, which travels at light speed.",
    ),
    Domain.DEEP_LEARNING: (
        "What are the key architectural differences between GPT and BERT, and when should you use each?",
        "GPT is autoregressive (left-to-right) and excellent for generation. BERT uses bidirectional attention and is better for classification and extraction. GPT for text generation, BERT for understanding tasks.",
        "GPT pre-trains on next-token prediction; BERT on masked language modelling. The attention mask differs fundamentally. Modern preference has shifted to decoder-only models for both tasks.",
    ),
    Domain.HEALTHCARE: (
        "What role does the microbiome play in immune system development and disease?",
        "The gut microbiome trains the immune system by presenting antigens, producing metabolites like short-chain fatty acids that regulate T-cell differentiation, and competing with pathogens.",
        "Germ-free mice have underdeveloped immune systems. Dysbiosis is linked to autoimmune disease, IBD, and metabolic syndrome. C-section birth and antibiotic overuse alter colonisation.",
    ),
    Domain.CLIMATE_ENERGY: (
        "What is the most cost-effective near-term path to decarbonising electricity grids?",
        "Solar PV and onshore wind are now the cheapest sources of new electricity in most markets. Combined with grid-scale batteries, demand response, and transmission expansion, they can provide reliable low-carbon power.",
        "LCOE of utility solar and wind has fallen 90% in a decade. IRENA data shows renewables are cheaper than new coal in most regions. Intermittency requires storage and grid flexibility.",
    ),
    Domain.FINANCE_ECONOMICS: (
        "How does quantitative easing affect asset prices and wealth inequality?",
        "QE purchases bonds to lower long-term yields, pushing investors into riskier assets (equities, real estate), inflating their prices. Since wealthy households own more of these assets, wealth inequality typically widens.",
        "Portfolio balance channel: lower bond yields make equities relatively more attractive. Rising house prices help existing owners but hurt first-time buyers. IMF and academic studies confirm QE-inequality link.",
    ),
    Domain.CYBERSECURITY: (
        "How do adversarial machine learning attacks threaten deployed AI systems?",
        "By adding carefully crafted, imperceptible perturbations to inputs, attackers can cause models to misclassify with high confidence. This threatens autonomous vehicles, medical imaging, and content moderation.",
        "Goodfellow et al. (2014) introduced FGSM. Physical-world attacks (adversarial patches, stop sign stickers) demonstrated real-world feasibility. Certified defences via randomised smoothing offer formal guarantees.",
    ),
    Domain.BIOTECH_GENOMICS: (
        "How does AlphaFold2 predict protein structure and why is it transformative?",
        "AlphaFold2 uses an Evoformer network trained on evolutionary multiple sequence alignments and structural templates to predict 3D coordinates with near-experimental accuracy. It unlocks drug target discovery and fundamental biology.",
        "50 years of the protein folding problem solved in a single model. 200M+ structures released in open access. It accelerates drug discovery by removing the need for costly crystallography.",
    ),
    Domain.SUPPLY_CHAIN: (
        "What is the bullwhip effect and how can companies mitigate it?",
        "Small demand fluctuations at the retail end amplify into large oscillations in orders and inventory upstream. Mitigation: share real demand data across the supply chain, reduce lead times, and use VMI.",
        "Forrester (1961) described the effect. Information asymmetry and order batching amplify variability. Procter & Gamble quantified it in the 1990s. ECR, CPFR, and better forecasting reduce it.",
    ),
    Domain.LEGAL_COMPLIANCE: (
        "How should enterprises manage GDPR compliance across multi-cloud environments?",
        "Implement data residency controls to ensure EU personal data stays within the EU, maintain a ROPA (Record of Processing Activities), apply data minimisation by design, and establish DPA agreements with all cloud providers.",
        "GDPR Articles 28/46 require data processor agreements and standard contractual clauses for transfers. Cross-border data flows post-Schrems II require transfer impact assessments.",
    ),
    Domain.ENTERPRISE_ARCHITECTURE: (
        "What are the key trade-offs between microservices and monolithic architectures for a fast-growing startup?",
        "Monoliths are simpler to develop and debug initially. Microservices enable team autonomy and independent scaling but add distributed systems complexity (network latency, distributed transactions, observability). Most successful startups start as a monolith and extract services strategically.",
        "Conway's Law: organisations build systems that mirror their communication structure. Martin Fowler's MonolithFirst pattern recommends starting monolithic until service boundaries are clear.",
    ),
    Domain.MARKETING_GROWTH: (
        "What is the most reliable way to measure the true incrementality of marketing spend?",
        "Randomised controlled experiments — geo holdouts or ghost ads — provide the cleanest incrementality measurement. Media mix modelling with Bayesian priors can extrapolate outside test regions but requires careful calibration against experiments.",
        "Platform-reported ROAS is heavily biased toward last-touch. Google/Meta attribution takes credit for organic conversions. Only properly designed holdout tests isolate incremental lift.",
    ),
    Domain.ORGANIZATIONAL_BEHAVIOR: (
        "What does the research say about the effectiveness of remote vs in-office work on team collaboration and innovation?",
        "Research shows complex collaboration (whiteboard sessions, serendipitous interactions) suffers remotely, while focused individual work improves. Hybrid models preserving 2-3 anchor days show the best balance of innovation and flexibility.",
        "Microsoft research (2021, Nature Human Behaviour) found remote work siloed communication networks. Longitudinal studies show junior employees suffer most from lost mentorship. However, commute elimination boosts satisfaction and retention.",
    ),
    Domain.STRATEGY_INTELLIGENCE: (
        "How do platform businesses build and defend competitive moats differently from traditional companies?",
        "Platforms create network effects (value grows with users), data moats (usage improves the product), and multi-homing switching costs. The key is identifying which side of the market to subsidise to achieve critical mass.",
        "Rochet & Tirole (2003) formalised multi-sided markets. Platforms must solve the chicken-and-egg problem. Successful moats combine direct network effects with proprietary data and developer ecosystem lock-in.",
    ),
    Domain.ALGORITHMS_PROGRAMMING: (
        "How does Rust's ownership model prevent data races at compile time without a garbage collector?",
        "Rust enforces that each value has exactly one owner, borrows are either many immutable refs or one mutable ref (never both), and all refs must be valid (lifetimes). The Send/Sync traits statically determine what can cross thread boundaries.",
        "The borrow checker encodes aliasing XOR mutation at compile time. std::sync::Mutex<T> wraps T so it can only be accessed behind a lock. Arc<T> enables shared ownership across threads. No GC needed because ownership is tracked statically.",
    ),
    Domain.INTERVIEW_PREP: (
        "How would you approach solving the 'Trapping Rain Water' problem in an interview, and what patterns does it test?",
        "First clarify: array of wall heights, compute trapped water. Brute force: for each element compute min(max_left, max_right) - height. Optimise: two-pointer approach O(n) time O(1) space. Tests: two-pointer pattern, prefix/suffix thinking.",
        "Explain approach before coding. Handle edge cases (empty array, single element). Communicate complexity. The two-pointer insight is that the side with the smaller maximum determines water height.",
    ),
    Domain.EE_LLM_RESEARCH: (
        "What is the connection between attention mechanisms in transformers and adaptive filtering in signal processing?",
        "Self-attention computes a weighted sum of values where weights come from query-key dot products — structurally identical to a time-varying FIR filter where the filter coefficients adapt per input. Both perform learned, input-dependent linear combinations.",
        "Wiener filtering minimises MSE given signal statistics; attention minimises cross-entropy given learned representations. The Evoformer uses attention over sequence and pair dimensions analogously to 2D filtering. SSMs (Mamba/S4) make this connection explicit via HiPPO theory and linear recurrences.",
    ),
    Domain.SIGNAL_PROCESSING: (
        "How do you design an optimal Wiener filter for removing additive white noise from a signal?",
        "The Wiener filter minimises the mean-square error between the filtered output and the desired signal. In the frequency domain it equals S_xy(f) / S_xx(f), the cross-PSD divided by the input PSD.",
        "Derived from the orthogonality principle: the error must be orthogonal to all observations. For white noise, S_xx = S_signal + S_noise, so the filter reduces to the SNR-weighted frequency response.",
    ),
    Domain.EXPERIMENT_RUNNER: (
        "Does adding momentum to gradient descent speed up convergence on a quadratic loss surface?",
        "Yes — momentum accumulates a velocity term that damps oscillations in ravines and accelerates descent along directions of consistent gradient. The convergence rate improves from O(κ) to O(√κ) for quadratic objectives.",
        "Polyak (1964) showed heavy-ball momentum achieves optimal convergence rate on strongly convex quadratics. Nesterov's accelerated gradient generalises this with a look-ahead correction, provably achieving O(1/k²) vs O(1/k) for gradient descent.",
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_intuition(domain: Domain) -> HumanIntuition:
    question, answer, reasoning = DOMAIN_QUESTIONS[domain]
    return HumanIntuition(
        question=question,
        intuitive_answer=answer,
        confidence=0.70,
        reasoning=reasoning,
    )


def _make_orch(use_mcp: bool = False) -> AgentOrchestrator:
    return AgentOrchestrator(
        backend=MockBackend(),
        use_mcp=use_mcp,
    )


# ---------------------------------------------------------------------------
# 1. Weight computation — one test per domain
# ---------------------------------------------------------------------------

class TestDualPipelineWeights:
    """Verify weight computation is in range and respects domain type."""

    @pytest.mark.parametrize("domain", list(Domain))
    def test_weights_sum_to_one(self, domain: Domain) -> None:
        cls = _AGENT_CLASSES.get(domain)
        if cls is None:
            pytest.skip(f"No agent registered for {domain}")
        agent = cls(backend=MockBackend())
        question, _, _ = DOMAIN_QUESTIONS[domain]
        iw, tw = agent._compute_weights(question, [])
        assert abs(iw + tw - 1.0) < 1e-9, f"Weights don't sum to 1: {iw}+{tw}"
        assert 0.0 <= iw <= 1.0
        assert 0.0 <= tw <= 1.0

    @pytest.mark.parametrize("domain", list(_INTUITION_HEAVY))
    def test_intuition_heavy_domains_favour_intuition(self, domain: Domain) -> None:
        cls = _AGENT_CLASSES.get(domain)
        if cls is None:
            pytest.skip(f"No agent registered for {domain}")
        agent = cls(backend=MockBackend())
        # No MCP results → tool_quality = 0 → intuition should dominate
        iw, tw = agent._compute_weights("Why does this happen?", [])
        assert iw > tw, f"{domain}: expected intuition_weight > tool_weight, got {iw} vs {tw}"

    @pytest.mark.parametrize("domain", list(_TOOL_HEAVY))
    def test_tool_heavy_domains_favour_tools_with_mcp_results(self, domain: Domain) -> None:
        from src.models import SearchResult
        cls = _AGENT_CLASSES.get(domain)
        if cls is None:
            pytest.skip(f"No agent registered for {domain}")
        agent = cls(backend=MockBackend())
        # Simulate 4 rich MCP results
        results = [
            SearchResult(title=f"Result {i}", url=f"https://example.com/{i}", snippet="evidence")
            for i in range(4)
        ]
        iw, tw = agent._compute_weights("What is the current regulation?", results)
        assert tw >= iw, f"{domain}: expected tool_weight >= intuition_weight with full MCP, got {tw} vs {iw}"

    def test_factual_question_boosts_tool_weight(self) -> None:
        from src.agents.physics_agent import PhysicsAgent
        agent = PhysicsAgent(backend=MockBackend())
        iw_factual, tw_factual = agent._compute_weights("What is the speed of light?", [])
        iw_why, tw_why = agent._compute_weights("Why does quantum tunnelling occur?", [])
        assert tw_factual > tw_why, "Factual question should have higher tool weight"

    def test_analytical_question_boosts_intuition_weight(self) -> None:
        from src.agents.physics_agent import PhysicsAgent
        agent = PhysicsAgent(backend=MockBackend())
        iw_why, tw_why = agent._compute_weights("Why does gravity bend light?", [])
        iw_list, tw_list = agent._compute_weights("List the fundamental forces.", [])
        assert iw_why > iw_list, "Analytical question should have higher intuition weight"


# ---------------------------------------------------------------------------
# 2. All 22 agents produce AgentResponse with weight fields
# ---------------------------------------------------------------------------

class TestAllAgentsDualPipeline:
    """Every registered agent returns a valid AgentResponse with weights."""

    @pytest.mark.parametrize("domain", list(Domain))
    def test_agent_answer_has_weight_fields(self, domain: Domain) -> None:
        cls = _AGENT_CLASSES.get(domain)
        if cls is None:
            pytest.skip(f"No agent registered for {domain}")
        agent = cls(backend=MockBackend(), mcp_client=None)
        question, _, _ = DOMAIN_QUESTIONS[domain]
        resp = agent.answer(question)
        assert isinstance(resp, AgentResponse)
        assert resp.domain == domain
        assert resp.answer
        assert 0.0 <= resp.intuition_weight <= 1.0
        assert 0.0 <= resp.tool_weight <= 1.0
        assert abs(resp.intuition_weight + resp.tool_weight - 1.0) < 1e-9

    @pytest.mark.parametrize("domain", list(Domain))
    def test_agent_confidence_in_range(self, domain: Domain) -> None:
        cls = _AGENT_CLASSES.get(domain)
        if cls is None:
            pytest.skip(f"No agent registered for {domain}")
        agent = cls(backend=MockBackend(), mcp_client=None)
        question, _, _ = DOMAIN_QUESTIONS[domain]
        resp = agent.answer(question)
        assert 0.0 <= resp.confidence <= 1.0


# ---------------------------------------------------------------------------
# 3. Full pipeline (run) — all 22 domains via orchestrator
# ---------------------------------------------------------------------------

class TestFullPipelineAllDomains:
    """Run the full pipeline for every domain and assert WeighingResult structure."""

    @pytest.mark.parametrize("domain", list(Domain))
    def test_run_produces_weighing_result(self, domain: Domain) -> None:
        from src.models import WeighingResult
        if domain not in _AGENT_CLASSES:
            pytest.skip(f"No agent registered for {domain}")
        question, answer, reasoning = DOMAIN_QUESTIONS[domain]
        intuition = HumanIntuition(
            question=question, intuitive_answer=answer,
            confidence=0.70, reasoning=reasoning,
        )
        with _make_orch() as orch:
            result = orch.run(
                question,
                prefilled_intuition=intuition,
                domains=[domain],
            )
        assert isinstance(result, WeighingResult)
        assert result.agent_responses
        assert result.synthesized_answer
        assert 0.0 <= result.intuition_accuracy_pct <= 100.0


# ---------------------------------------------------------------------------
# 4. Debate engine — produces DebateResult with rounds
# ---------------------------------------------------------------------------

class TestDebateEngine:
    """Debate engine produces correct structure and non-empty content."""

    def test_debate_returns_result(self) -> None:
        from src.models import DebateResult
        domain = Domain.PHYSICS
        question, answer, reasoning = DOMAIN_QUESTIONS[domain]
        intuition = HumanIntuition(
            question=question, intuitive_answer=answer,
            confidence=0.70, reasoning=reasoning,
        )
        with _make_orch() as orch:
            result = orch.debate(
                question,
                prefilled_intuition=intuition,
                domains=[domain],
            )
        assert isinstance(result, DebateResult)
        assert result.synthesized_verdict
        assert result.rounds, "Debate must have at least one round"
        assert result.human_position.source == "human"
        assert result.tool_evidence.source == "tool"
        assert result.agent_positions

    def test_debate_rounds_have_aspect_and_synthesis(self) -> None:
        domain = Domain.SOCIAL_SCIENCE
        question, answer, reasoning = DOMAIN_QUESTIONS[domain]
        intuition = HumanIntuition(
            question=question, intuitive_answer=answer,
            confidence=0.65, reasoning=reasoning,
        )
        with _make_orch() as orch:
            result = orch.debate(
                question,
                prefilled_intuition=intuition,
                domains=[domain],
            )
        for rnd in result.rounds:
            assert rnd.aspect
            assert rnd.round_synthesis
            assert rnd.positions

    def test_debate_accuracy_pct_in_range(self) -> None:
        domain = Domain.DEEP_LEARNING
        question, answer, reasoning = DOMAIN_QUESTIONS[domain]
        intuition = HumanIntuition(
            question=question, intuitive_answer=answer,
            confidence=0.75, reasoning=reasoning,
        )
        with _make_orch() as orch:
            result = orch.debate(
                question,
                prefilled_intuition=intuition,
                domains=[domain],
            )
        assert 0.0 <= result.intuition_accuracy_pct <= 100.0
        assert 0.0 <= result.verdict_confidence <= 1.0


# ---------------------------------------------------------------------------
# 5. Interview prep mode — three agents, InterviewResult structure
# ---------------------------------------------------------------------------

class TestInterviewPrepMode:
    """interview_prep() engages three agents and returns InterviewResult."""

    def test_interview_result_structure(self) -> None:
        from src.models import InterviewResult
        domain = Domain.INTERVIEW_PREP
        question, answer, reasoning = DOMAIN_QUESTIONS[domain]
        intuition = HumanIntuition(
            question=question, intuitive_answer=answer,
            confidence=0.72, reasoning=reasoning,
        )
        with _make_orch() as orch:
            result = orch.interview_prep(question, prefilled_intuition=intuition)
        assert isinstance(result, InterviewResult)
        assert result.question == question
        assert result.candidate_answer == answer
        assert 0.0 <= result.technical_score <= 1.0
        assert result.technical_feedback
        assert result.overall_analysis
        assert result.synthesized_answer

    def test_interview_prep_uses_social_science_for_mental_prep(self) -> None:
        question, answer, reasoning = DOMAIN_QUESTIONS[Domain.INTERVIEW_PREP]
        intuition = HumanIntuition(
            question=question, intuitive_answer=answer,
            confidence=0.70, reasoning=reasoning,
        )
        with _make_orch() as orch:
            result = orch.interview_prep(question, prefilled_intuition=intuition)
        # SocialScienceAgent provides the mental preparation section
        assert result.mental_preparation, "mental_preparation should come from SocialScienceAgent"


# ---------------------------------------------------------------------------
# 6. Model evaluation (offline — mock only)
# ---------------------------------------------------------------------------

class TestModelEvaluation:
    """evaluate_models() with mock backend cycles and returns ModelEvaluationResult."""

    def test_evaluate_single_mock_model(self) -> None:
        from src.models import ModelEvaluationResult
        domain = Domain.ALGORITHMS_PROGRAMMING
        question, answer, reasoning = DOMAIN_QUESTIONS[domain]
        intuition = HumanIntuition(
            question=question, intuitive_answer=answer,
            confidence=0.72, reasoning=reasoning,
        )
        with _make_orch() as orch:
            result = orch.evaluate_models(
                question,
                model_specs=["mock"],
                prefilled_intuition=intuition,
                domains=[domain],
            )
        assert isinstance(result, ModelEvaluationResult)
        assert result.models_evaluated == 1
        assert result.models_available == 1
        assert result.consensus_answer
        assert result.best_model_spec == "mock"

    def test_evaluate_handles_invalid_spec_gracefully(self) -> None:
        question, answer, reasoning = DOMAIN_QUESTIONS[Domain.PHYSICS]
        intuition = HumanIntuition(
            question=question, intuitive_answer=answer,
            confidence=0.70, reasoning=reasoning,
        )
        with _make_orch() as orch:
            result = orch.evaluate_models(
                question,
                model_specs=["mock", "badprovider:model"],
                prefilled_intuition=intuition,
                domains=[Domain.PHYSICS],
            )
        assert result.models_evaluated == 2
        assert result.models_available == 1  # only mock is valid
        assert result.model_results[1].backend_available is False
        assert result.model_results[1].error is not None

    def test_evaluate_two_mock_models(self) -> None:
        question, answer, reasoning = DOMAIN_QUESTIONS[Domain.HEALTHCARE]
        intuition = HumanIntuition(
            question=question, intuitive_answer=answer,
            confidence=0.68, reasoning=reasoning,
        )
        with _make_orch() as orch:
            result = orch.evaluate_models(
                question,
                model_specs=["mock", "mock"],
                prefilled_intuition=intuition,
                domains=[Domain.HEALTHCARE],
            )
        assert result.models_available == 2
        assert 0.0 <= result.mean_intuition_accuracy_pct <= 100.0


# ---------------------------------------------------------------------------
# 7. Model sweep + cycling (opt-in via RUN_MODEL_SWEEP=1)
# ---------------------------------------------------------------------------

_RUN_SWEEP = os.environ.get("RUN_MODEL_SWEEP", "").strip() in {"1", "true", "yes"}
_SWEEP_MODELS = [
    s.strip()
    for s in os.environ.get("INTUITION_SCIENTIST_MODELS", "mock").split(",")
    if s.strip()
]

sweep_mark = pytest.mark.skipif(
    not _RUN_SWEEP,
    reason="Model sweep tests are opt-in: set RUN_MODEL_SWEEP=1 to enable.",
)


@sweep_mark
@pytest.mark.parametrize("domain", list(Domain))
@pytest.mark.parametrize("model_spec", _SWEEP_MODELS)
def test_model_sweep_all_domains(domain: Domain, model_spec: str) -> None:
    """Run the full pipeline for every domain × every model in INTUITION_SCIENTIST_MODELS."""
    from src.llm.registry import get_backend

    if domain not in _AGENT_CLASSES:
        pytest.skip(f"No agent registered for {domain}")

    try:
        backend = get_backend(model_spec)
    except ValueError as exc:
        pytest.skip(f"Invalid model spec '{model_spec}': {exc}")

    question, answer, reasoning = DOMAIN_QUESTIONS[domain]
    intuition = HumanIntuition(
        question=question, intuitive_answer=answer,
        confidence=0.70, reasoning=reasoning,
    )

    try:
        with AgentOrchestrator(backend=backend, use_mcp=False) as orch:
            result = orch.run(
                question,
                prefilled_intuition=intuition,
                domains=[domain],
            )
    except RuntimeError as exc:
        pytest.skip(f"Backend '{model_spec}' unavailable: {exc}")

    assert result.agent_responses, "agent_responses must not be empty"
    assert result.synthesized_answer, "synthesized_answer must not be empty"
    assert result.alignment_scores, "alignment_scores must not be empty"
    for score in result.alignment_scores:
        assert 0.0 <= score.semantic_similarity <= 1.0


@sweep_mark
@pytest.mark.parametrize("model_spec", _SWEEP_MODELS)
def test_model_sweep_evaluate_models(model_spec: str) -> None:
    """evaluate_models() works with real backends when sweep is enabled."""
    domain = Domain.STRATEGY_INTELLIGENCE
    question, answer, reasoning = DOMAIN_QUESTIONS[domain]
    intuition = HumanIntuition(
        question=question, intuitive_answer=answer,
        confidence=0.70, reasoning=reasoning,
    )

    try:
        with AgentOrchestrator(backend_spec=model_spec, use_mcp=False) as orch:
            result = orch.evaluate_models(
                question,
                model_specs=[model_spec],
                prefilled_intuition=intuition,
                domains=[domain],
            )
    except (RuntimeError, ValueError) as exc:
        pytest.skip(f"Backend '{model_spec}' unavailable: {exc}")

    assert result.models_evaluated >= 1
    assert result.consensus_answer


@sweep_mark
@pytest.mark.parametrize("model_spec", _SWEEP_MODELS)
def test_model_sweep_debate(model_spec: str) -> None:
    """Debate engine runs end-to-end with real backends when sweep is enabled."""
    domain = Domain.EE_LLM_RESEARCH
    question, answer, reasoning = DOMAIN_QUESTIONS[domain]
    intuition = HumanIntuition(
        question=question, intuitive_answer=answer,
        confidence=0.72, reasoning=reasoning,
    )

    try:
        from src.llm.registry import get_backend
        backend = get_backend(model_spec)
        with AgentOrchestrator(backend=backend, use_mcp=False) as orch:
            result = orch.debate(
                question,
                prefilled_intuition=intuition,
                domains=[domain],
            )
    except (RuntimeError, ValueError) as exc:
        pytest.skip(f"Backend '{model_spec}' unavailable: {exc}")

    assert result.synthesized_verdict
    assert result.rounds
