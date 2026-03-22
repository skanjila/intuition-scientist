"""ExperimentRunnerAgent — proposes and coordinates lightweight experiments
to solve or illuminate problems.

Design principles
-----------------
* Converts any question into a structured experiment plan with explicit
  hypotheses, variables, expected outcomes, and iteration steps.
* Defines "experiments" as **safe, local, deterministic actions**: small
  numeric simulations, toy examples, dimensional checks, or pseudo-code
  that the user can run locally — no external credentials required.
* Keeps the **human intuition loop** explicit: requests the user's intuition
  and confidence level, then compares observed experimental outcomes to that
  intuition and surfaces discrepancies.
* If live code execution is not available, produces a complete
  *experiment protocol* with runnable Python/NumPy snippets the user can
  execute in their own environment.
"""

from src.agents.base_agent import BaseAgent
from src.models import Domain


class ExperimentRunnerAgent(BaseAgent):
    """Agent that converts questions into structured experiment plans.

    The agent operates in three phases:

    Phase 1 — Hypothesis Formation
        Given the question (and the user's intuition), enumerate 2-4 falsifiable
        hypotheses ordered from most to least likely.

    Phase 2 — Experiment Design
        For each hypothesis propose a lightweight, local experiment:
        - Numeric simulation (NumPy / SciPy snippet)
        - Toy analytical example with explicit parameter sweep
        - Dimensional / order-of-magnitude check
        - Thought experiment with stated assumptions

    Phase 3 — Outcome Analysis & Intuition Calibration
        Compare actual (or predicted) outcomes to the user's initial intuition.
        Identify which hypothesis is supported/refuted and explain *why* the
        intuition was accurate, partially correct, or misleading.
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
            "## Experiment Design Protocol\n\n"
            "For every question you MUST follow this three-phase protocol:\n\n"
            "### Phase 1 — Elicit Intuition\n"
            "Ask the user: 'Before we design the experiment, what is your intuition "
            "about the answer? How confident are you (0 = wild guess, 1 = very sure)?'\n"
            "Record the stated intuition and confidence.\n\n"
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
