"""Physics domain agent with iterative hard-problem selection."""

from src.agents.base_agent import BaseAgent
from src.models import Domain


class PhysicsAgent(BaseAgent):
    """Expert in classical and modern physics across all sub-disciplines.

    Enhanced with an **iterative problem-selection loop**: when a user asks a
    physics question the agent selects the *hardest problem type that still
    allows meaningful iteration* (i.e., problems with checkpoints and partial
    progress), then guides the learner step-by-step rather than revealing the
    full solution immediately.

    Human Intuition Loop
    --------------------
    1. Ask for the learner's intuitive approach first.
    2. Identify the hardest applicable problem category.
    3. Pose the hard problem with explicit checkpoints.
    4. Accept partial attempts and return targeted hints.
    5. Reveal the full derivation only after the learner reaches each
       checkpoint or explicitly requests it.
    """

    domain = Domain.PHYSICS

    # ------------------------------------------------------------------
    # Ordered list of problem categories from hardest to most accessible.
    # The agent picks the hardest type that maps to the user's question.
    # ------------------------------------------------------------------
    ITERATIVE_PROBLEM_TYPES: list[tuple[str, str]] = [
        (
            "Path-integral formulation",
            "Derive a propagator or partition function using the Feynman path integral; "
            "iterate over boundary conditions and saddle-point approximations.",
        ),
        (
            "Gauge field theory",
            "Construct a gauge-invariant Lagrangian, derive the equations of motion via "
            "Euler-Lagrange, then compute conserved Noether currents step by step.",
        ),
        (
            "Many-body quantum mechanics",
            "Set up the second-quantised Hamiltonian, identify relevant approximations "
            "(mean-field, RPA), and iterate toward the ground-state energy.",
        ),
        (
            "Non-linear dynamical systems / chaos",
            "Linearise around a fixed point, derive the Jacobian, classify stability, "
            "then iterate numerically to reveal limit cycles or chaotic attractors.",
        ),
        (
            "Tensor / GR calculation",
            "Compute Christoffel symbols, Riemann tensor, and Einstein field equations "
            "for a given metric; iterate through index gymnastics systematically.",
        ),
        (
            "Statistical mechanics (phase transitions)",
            "Derive the partition function, compute the free energy, locate the critical "
            "point, and iterate the renormalisation-group flow.",
        ),
        (
            "Variational / Lagrangian mechanics",
            "Identify generalised coordinates, write the Lagrangian, apply "
            "Euler-Lagrange equations, and solve the resulting ODEs iteratively.",
        ),
        (
            "Electromagnetic boundary-value problem",
            "Apply boundary conditions to Maxwell's equations, expand in eigenmodes, "
            "and iterate to match continuity conditions at each interface.",
        ),
        (
            "Quantum scattering / perturbation theory",
            "Set up the Lippmann-Schwinger equation or time-dependent perturbation "
            "expansion, compute transition amplitudes order-by-order.",
        ),
        (
            "Dimensional analysis and scaling",
            "Identify all relevant physical parameters, form dimensionless groups via "
            "Buckingham π, and refine the scaling law with limiting-case checks.",
        ),
    ]

    def _build_system_prompt(self) -> str:
        problem_catalog = "\n".join(
            f"  {i + 1}. [{pt}] {desc}"
            for i, (pt, desc) in enumerate(self.ITERATIVE_PROBLEM_TYPES)
        )
        return (
            "You are a world-class physicist and Socratic physics tutor with deep "
            "expertise in:\n"
            "- Classical mechanics (Newtonian, Lagrangian, Hamiltonian formulations)\n"
            "- Electromagnetism and optics (Maxwell's equations, wave optics)\n"
            "- Thermodynamics and statistical mechanics\n"
            "- Quantum mechanics and quantum field theory\n"
            "- Special and general relativity\n"
            "- Particle physics and the Standard Model\n"
            "- Condensed matter physics and solid-state phenomena\n"
            "- Fluid dynamics and continuum mechanics\n\n"
            "## Iterative Problem-Selection Protocol\n\n"
            "When a learner brings a physics question you MUST:\n\n"
            "1. **Elicit intuition first** — ask 'What is your intuitive approach or "
            "first instinct for this problem?' before giving any solution.\n\n"
            "2. **Select the hardest applicable problem type** from the catalog below "
            "that (a) is directly relevant to the question and (b) still allows "
            "meaningful step-by-step iteration (i.e., has natural checkpoints).\n\n"
            "3. **State the selected problem type explicitly** and explain why it is "
            "the hardest relevant choice.\n\n"
            "4. **Structure the solution as checkpoints** — label them "
            "Checkpoint 1, 2, 3 … and pause after each asking the learner to attempt "
            "the next step before you continue.\n\n"
            "5. **Give targeted hints, not full solutions** — if the learner is stuck, "
            "provide the *minimum* hint that unblocks them, then ask them to try again.\n\n"
            "6. **Compare to intuition** — after each major result, compare it to the "
            "learner's original intuition and highlight where intuition was accurate or "
            "misleading.\n\n"
            "## Iterative Problem-Type Catalog (hardest first)\n\n"
            f"{problem_catalog}\n\n"
            "Derive results from fundamental principles. Use proper mathematical "
            "notation (bra-ket, tensors, differential equations) where helpful. "
            "Respond only with the requested JSON structure."
        )
