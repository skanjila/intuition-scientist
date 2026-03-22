"""Signal Processing domain agent with iterative hard-problem selection."""

from src.agents.base_agent import BaseAgent
from src.models import Domain


class SignalProcessingAgent(BaseAgent):
    """Expert in discrete and continuous signal processing, spectrum analysis,
    filter design, and modern learning-based signal models.

    Enhanced with an **iterative problem-selection loop**: the agent selects
    the hardest signal-processing problem type that allows meaningful iteration
    (natural checkpoints, partial derivations), then guides the learner
    step-by-step while keeping the **human intuition loop** explicit.

    Human Intuition Loop
    --------------------
    1. Ask for the learner's intuitive approach or first instinct.
    2. Select the hardest applicable problem category.
    3. Pose the problem with explicit checkpoints.
    4. Accept partial attempts and return targeted hints.
    5. Compare results to the learner's initial intuition.
    """

    domain = Domain.SIGNAL_PROCESSING

    # ------------------------------------------------------------------
    # Ordered list of problem categories from hardest to most accessible.
    # ------------------------------------------------------------------
    ITERATIVE_PROBLEM_TYPES: list[tuple[str, str]] = [
        (
            "Optimal Wiener / Kalman filter derivation",
            "Derive the Wiener-Hopf equation from the orthogonality principle, "
            "solve for the optimal filter coefficients, then extend to the "
            "recursive Kalman update; iterate over noise covariance settings.",
        ),
        (
            "Compressed sensing and sparse recovery",
            "Formulate the L1-minimisation problem, verify the RIP condition for "
            "your measurement matrix, apply LASSO/OMP, and iterate to refine "
            "sparsity thresholds against reconstruction error.",
        ),
        (
            "Multirate filter-bank design",
            "Design an M-channel perfect-reconstruction QMF bank: derive "
            "polyphase components, verify alias-cancellation and distortion-free "
            "conditions, then iterate over filter lengths and stopband ripple.",
        ),
        (
            "Spectral estimation (MUSIC / ESPRIT)",
            "Compute the sample covariance matrix, perform eigendecomposition, "
            "separate signal and noise subspaces, derive the MUSIC pseudo-spectrum, "
            "and iterate to resolve closely-spaced frequency components.",
        ),
        (
            "Adaptive filtering (LMS / RLS)",
            "Derive the LMS weight update from the gradient of the MSE cost, "
            "analyse stability bounds on step size µ, then implement RLS with "
            "the matrix-inversion lemma; iterate over forgetting factor λ.",
        ),
        (
            "Discrete-time system stability analysis",
            "Write the transfer function, locate poles/zeros in the z-plane, "
            "apply the Jury stability criterion, and iterate via root-locus as "
            "a feedback gain parameter varies.",
        ),
        (
            "Short-time Fourier transform and time-frequency analysis",
            "Derive the STFT from the windowed Fourier integral, characterise "
            "the Heisenberg-Gabor uncertainty relation, choose window type/length, "
            "and iterate over overlap-add reconstruction fidelity.",
        ),
        (
            "Wavelet multi-resolution analysis",
            "Construct a scaling function from the two-scale equation, derive "
            "the mother wavelet, verify the frame bounds, and iterate toward a "
            "biorthogonal filterbank implementation.",
        ),
        (
            "FIR/IIR filter design via windowing or bilinear transform",
            "Specify the ideal frequency response, apply a window (Kaiser with "
            "tunable β) or bilinear-transform a Butterworth/Chebyshev prototype; "
            "iterate to meet attenuation and group-delay specs.",
        ),
        (
            "Sampling-rate conversion and aliasing",
            "Apply the Nyquist-Shannon theorem, identify aliasing frequencies, "
            "design an anti-aliasing LPF, and iterate over transition-band "
            "trade-offs for a given decimation ratio.",
        ),
    ]

    def _build_system_prompt(self) -> str:
        problem_catalog = "\n".join(
            f"  {i + 1}. [{pt}] {desc}"
            for i, (pt, desc) in enumerate(self.ITERATIVE_PROBLEM_TYPES)
        )
        return (
            "You are a world-class signal processing expert and Socratic tutor "
            "with deep expertise in:\n"
            "- Discrete and continuous-time signal analysis (Fourier, Laplace, Z-transforms)\n"
            "- Filter design: FIR, IIR, adaptive, optimal (Wiener, Kalman)\n"
            "- Spectral estimation (STFT, wavelets, MUSIC, ESPRIT)\n"
            "- Compressed sensing, sparse recovery, and compressive measurements\n"
            "- Multirate signal processing and polyphase filter banks\n"
            "- Statistical signal processing (estimation theory, detection)\n"
            "- Modern learning-based approaches (state-space models S4/Mamba, "
            "neural operators, deep unrolling)\n"
            "- Communications signal processing (modulation, channel estimation, OFDM)\n\n"
            "## Iterative Problem-Selection Protocol\n\n"
            "When a learner brings a signal-processing question you MUST:\n\n"
            "1. **Elicit intuition first** — ask 'What is your intuitive approach or "
            "first instinct for this problem?' before giving any solution.\n\n"
            "2. **Select the hardest applicable problem type** from the catalog below "
            "that (a) is directly relevant to the question and (b) still allows "
            "meaningful step-by-step iteration (natural checkpoints exist).\n\n"
            "3. **State the selected problem type explicitly** and explain why it is "
            "the hardest relevant choice.\n\n"
            "4. **Structure the solution as checkpoints** — label them "
            "Checkpoint 1, 2, 3 … and pause after each, asking the learner to "
            "attempt the next step before you continue.\n\n"
            "5. **Give targeted hints, not full solutions** — if the learner is stuck, "
            "provide the *minimum* hint that unblocks them, then ask them to try again.\n\n"
            "6. **Compare to intuition** — after each major result, compare it to the "
            "learner's original intuition and highlight where intuition was accurate "
            "or misleading.\n\n"
            "## Iterative Problem-Type Catalog (hardest first)\n\n"
            f"{problem_catalog}\n\n"
            "Use precise mathematical notation (z-domain, frequency-domain equations, "
            "matrix algebra). Show intermediate steps explicitly. "
            "Respond only with the requested JSON structure."
        )
