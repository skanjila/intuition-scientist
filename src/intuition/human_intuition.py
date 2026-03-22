"""Human Intuition capture and structuring system.

This module is responsible for:
1. Interactively prompting a human user for their intuitive answer.
2. Inferring which domains the intuition touches on.
3. Returning a structured ``HumanIntuition`` object that the weighing system
   can compare against domain-agent responses.

The ``IntuitionCapture`` class works in two modes:

* **Interactive** (default): prompts the user via stdin/stdout.
* **Programmatic**: accepts a pre-built ``HumanIntuition`` directly, which is
  useful for testing or when the caller already has the intuition data.

The module also exposes :func:`generate_auto_intuition` for the non-interactive
``--auto-intuition`` CLI mode, which synthesises a plausible human-like
intuition response without blocking on stdin.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

from src.models import Domain, HumanIntuition

if TYPE_CHECKING:
    # Avoid a circular import at runtime; only needed for type annotations.
    from src.llm.base import LLMBackend

# Keywords that map natural-language phrases to domains
_DOMAIN_KEYWORDS: dict[Domain, list[str]] = {
    Domain.ELECTRICAL_ENGINEERING: [
        "circuit", "voltage", "current", "resistor", "capacitor", "inductor",
        "transistor", "amplifier", "power", "electromagnet", "antenna", "rf",
        "semiconductor", "diode", "oscillator", "signal", "waveform",
    ],
    Domain.COMPUTER_SCIENCE: [
        "algorithm", "data structure", "complexity", "big-o", "compiler",
        "operating system", "network", "protocol", "database", "software",
        "programming", "code", "function", "recursion", "graph", "tree",
        "sorting", "search", "security", "encryption",
    ],
    Domain.NEURAL_NETWORKS: [
        "neuron", "synapse", "activation", "backprop", "gradient", "layer",
        "perceptron", "recurrent", "lstm", "attention", "transformer",
        "convolution", "pooling", "weight", "bias", "network",
    ],
    Domain.SOCIAL_SCIENCE: [
        "society", "culture", "behaviour", "behavior", "psychology", "social",
        "group", "norm", "institution", "economy", "political", "influence",
        "cognition", "bias", "survey", "qualitative", "ethnograph",
    ],
    Domain.SPACE_SCIENCE: [
        "star", "planet", "galaxy", "universe", "cosmos", "orbit", "gravity",
        "black hole", "neutron", "supernova", "nasa", "telescope", "asteroid",
        "comet", "dark matter", "dark energy", "big bang", "space", "solar",
    ],
    Domain.PHYSICS: [
        "quantum", "relativity", "photon", "electron", "proton", "wave",
        "particle", "energy", "force", "mass", "momentum", "thermodynamic",
        "entropy", "field", "mechanics", "optics", "laser", "nuclear",
    ],
    Domain.DEEP_LEARNING: [
        "deep learning", "llm", "gpt", "transformer", "embedding", "fine-tun",
        "pretrain", "foundation model", "diffusion", "generative", "llama",
        "reinforcement learning", "rlhf", "lora", "bert", "token",
    ],
    # High-economic-value industry domains
    Domain.HEALTHCARE: [
        "disease", "drug", "medicine", "patient", "clinical", "diagnosis",
        "therapy", "treatment", "hospital", "pharma", "cancer", "vaccine",
        "genomic", "biomarker", "trial", "dose", "symptom", "surgery",
        "health", "epidemic", "public health", "imaging", "radiology",
    ],
    Domain.CLIMATE_ENERGY: [
        "climate", "carbon", "emission", "greenhouse", "renewable", "solar",
        "wind", "battery", "storage", "grid", "decarbonisation", "net zero",
        "fossil fuel", "clean energy", "hydrogen", "capture", "warming",
        "temperature", "sea level", "electricity", "power", "energy transition",
    ],
    Domain.FINANCE_ECONOMICS: [
        "market", "risk", "finance", "economic", "investment", "capital",
        "portfolio", "asset", "equity", "bond", "derivative", "hedge",
        "monetary", "inflation", "interest rate", "bank", "credit", "liquidity",
        "gdp", "fiscal", "arbitrage", "quant", "trading", "valuation",
    ],
    Domain.CYBERSECURITY: [
        "security", "attack", "vulnerability", "threat", "encrypt", "cyber",
        "malware", "ransomware", "phishing", "firewall", "intrusion",
        "exploit", "zero-day", "authentication", "cryptography", "hack",
        "breach", "privacy", "compliance", "password", "certificate",
    ],
    Domain.BIOTECH_GENOMICS: [
        "gene", "genome", "protein", "crispr", "cell", "biotech",
        "dna", "rna", "sequencing", "mutation", "chromosome", "allele",
        "expression", "stem cell", "therapy", "synthetic biology", "enzyme",
        "amino acid", "folding", "omics", "microbiome", "bioinformatics",
    ],
    Domain.SUPPLY_CHAIN: [
        "supply", "logistics", "inventory", "demand", "shipping", "warehouse",
        "procurement", "sourcing", "distribution", "fulfillment", "route",
        "lead time", "stock", "vendor", "delivery", "freight", "last mile",
        "operations", "forecast", "resilience", "disruption", "manufacturing",
    ],
    # Enterprise problem domains
    Domain.LEGAL_COMPLIANCE: [
        "contract", "legal", "compliance", "regulation", "law", "liability",
        "intellectual property", "patent", "trademark", "copyright", "gdpr",
        "sox", "hipaa", "litigation", "arbitration", "indemnification",
        "governance", "fiduciary", "regulatory", "audit", "policy",
    ],
    Domain.ENTERPRISE_ARCHITECTURE: [
        "architecture", "microservice", "cloud", "kubernetes", "docker",
        "api", "platform", "technical debt", "migration", "integration",
        "data mesh", "event-driven", "ddd", "cqrs", "service mesh",
        "infrastructure", "devops", "devsecops", "observability", "sre",
    ],
    Domain.MARKETING_GROWTH: [
        "marketing", "growth", "customer", "acquisition", "retention",
        "conversion", "funnel", "attribution", "clv", "cac", "churn",
        "brand", "campaign", "segment", "pricing", "revenue", "demand gen",
        "seo", "content", "product-led", "viral", "paid media",
    ],
    Domain.ORGANIZATIONAL_BEHAVIOR: [
        "organisation", "culture", "talent", "workforce", "leadership",
        "team", "performance", "engagement", "motivation", "hiring",
        "retention", "succession", "diversity", "inclusion", "coaching",
        "change management", "communication", "collaboration", "wellbeing",
    ],
    Domain.STRATEGY_INTELLIGENCE: [
        "strategy", "competitive", "market", "positioning", "moat",
        "competitive advantage", "m&a", "acquisition", "merger", "scenario",
        "porter", "value chain", "disruption", "business model", "innovation",
        "intelligence", "intelligence gathering", "swot", "pestle",
    ],
    # Mastery / interview / PhD research domains
    Domain.ALGORITHMS_PROGRAMMING: [
        "algorithm", "data structure", "python", "rust", "golang", "go",
        "complexity", "big-o", "dynamic programming", "graph", "tree",
        "sorting", "binary search", "recursion", "pointer", "iterator",
        "ownership", "borrow", "goroutine", "channel", "async", "concurrency",
    ],
    Domain.INTERVIEW_PREP: [
        "interview", "leetcode", "system design", "faang", "google", "meta",
        "amazon", "apple", "microsoft", "coding", "behavioral", "star",
        "whiteboard", "offer", "technical screen", "oa", "online assessment",
        "two pointer", "sliding window", "backtracking", "dp pattern",
    ],
    Domain.EE_LLM_RESEARCH: [
        "llm", "signal processing", "safety", "alignment", "transformer",
        "phd", "research", "dissertation", "paper", "publication", "fft",
        "filter", "spectrum", "modulation", "jailbreak", "interpretability",
        "mechanistic", "rlhf", "dpo", "fine-tuning", "inference", "training",
        "hallucination", "red team", "watermark", "sparse autoencoder",
    ],
    # Dedicated signal processing domain
    Domain.SIGNAL_PROCESSING: [
        "signal", "filter", "fft", "dft", "fourier", "laplace", "z-transform",
        "nyquist", "sampling", "aliasing", "convolution", "deconvolution",
        "spectrum", "frequency", "bandwidth", "noise", "snr", "lms", "rls",
        "wiener", "kalman", "adaptive filter", "wavelet", "stft", "spectrogram",
        "fir", "iir", "butterworth", "chebyshev", "bode", "impulse response",
        "transfer function", "pole", "zero", "multirate", "decimation",
        "interpolation", "polyphase", "music algorithm", "esprit", "sparse",
    ],
    # Experiment runner domain
    Domain.EXPERIMENT_RUNNER: [
        "experiment", "hypothesis", "test", "simulation", "simulate",
        "model", "predict", "verify", "validate", "measure", "observation",
        "variable", "control", "outcome", "result", "evidence", "protocol",
        "reproduce", "reproducible", "monte carlo", "sweep", "sensitivity",
        "perturbation", "fermi estimate", "order of magnitude", "toy example",
        "numerical", "parametric", "falsifiable", "trial", "run",
    ],
}


class IntuitionCapture:
    """Captures and structures human intuition about a question."""

    def __init__(self, interactive: bool = True) -> None:
        self.interactive = interactive

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def capture(
        self,
        question: str,
        *,
        prefilled: Optional[HumanIntuition] = None,
    ) -> HumanIntuition:
        """Return a ``HumanIntuition`` for *question*.

        If *prefilled* is provided it is returned as-is (programmatic mode).
        Otherwise the user is prompted interactively.
        """
        if prefilled is not None:
            return prefilled

        if self.interactive:
            return self._interactive_capture(question)

        raise ValueError(
            "interactive=False but no prefilled HumanIntuition was provided."
        )

    # ------------------------------------------------------------------
    # Interactive prompting
    # ------------------------------------------------------------------

    def _interactive_capture(self, question: str) -> HumanIntuition:
        print("\n" + "=" * 70)
        print("🧠  HUMAN INTUITION CAPTURE")
        print("=" * 70)
        print(f"\nQuestion:\n  {question}\n")

        intuitive_answer = self._prompt_multiline(
            "Your intuitive answer (press Enter twice when done):\n> "
        )
        reasoning = self._prompt_multiline(
            "What reasoning or gut feeling drove that answer? "
            "(press Enter twice when done):\n> "
        )
        confidence = self._prompt_float(
            "How confident are you in your intuition? (0 = wild guess, 1 = very sure): ",
            min_val=0.0,
            max_val=1.0,
        )

        inferred = self._infer_domains(question + " " + intuitive_answer)

        return HumanIntuition(
            question=question,
            intuitive_answer=intuitive_answer,
            confidence=confidence,
            reasoning=reasoning,
            domain_guesses=inferred,
        )

    # ------------------------------------------------------------------
    # Domain inference
    # ------------------------------------------------------------------

    @staticmethod
    def infer_domains(text: str) -> list[Domain]:
        """Return the domains most likely relevant to *text* (public helper)."""
        return IntuitionCapture._infer_domains(text)

    @staticmethod
    def _infer_domains(text: str) -> list[Domain]:
        text_lower = text.lower()
        scores: dict[Domain, int] = {d: 0 for d in Domain}
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[domain] += 1
        # Return domains that have at least one keyword match, sorted by score
        relevant = [(d, s) for d, s in scores.items() if s > 0]
        relevant.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in relevant] or list(Domain)

    # ------------------------------------------------------------------
    # Input helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prompt_multiline(prompt: str) -> str:
        print(prompt, end="", flush=True)
        lines: list[str] = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        return "\n".join(lines).strip()

    @staticmethod
    def _prompt_float(prompt: str, min_val: float, max_val: float) -> float:
        while True:
            raw = input(prompt).strip()
            try:
                val = float(raw)
                if min_val <= val <= max_val:
                    return val
                print(f"  Please enter a number between {min_val} and {max_val}.")
            except ValueError:
                print("  Please enter a numeric value (e.g. 0.7).")


# ---------------------------------------------------------------------------
# Auto-intuition generation (non-interactive mode)
# ---------------------------------------------------------------------------


def generate_auto_intuition(
    question: str,
    *,
    backend: Optional["LLMBackend"] = None,
) -> HumanIntuition:
    """Generate a lightweight "human intuition" response without user prompting.

    This is used by the ``--auto-intuition`` CLI mode to synthesise a plausible
    non-expert perspective on *question* without blocking on stdin.

    Strategy
    --------
    1. Infer the most relevant domain(s) from *question* using the same
       keyword-matching heuristic as interactive capture.
    2. If an LLM backend is available, issue a short "quick-think" prompt
       asking the model to role-play as a curious non-expert and share their
       gut feeling.  The prompt is intentionally lightweight (≤ 120 tokens)
       so the auto-intuition pass does not dominate wall-clock time.
    3. If no backend is available (offline / mock mode), fall back to a
       domain-templated placeholder answer that is clearly labelled.

    The returned :class:`~src.models.HumanIntuition` uses moderate confidence
    (0.5) to signal that it represents a plausible but uncertain human estimate
    rather than a carefully considered human judgment.  Downstream consumers
    (the weighing system, debate engine, etc.) treat it identically to an
    interactively captured intuition.

    Parameters
    ----------
    question:
        The question to generate intuition for.
    backend:
        Optional LLM backend.  When supplied a short "quick-think" prompt is
        issued; otherwise a keyword-templated fallback is used.
    """
    inferred_domains = IntuitionCapture.infer_domains(question)
    primary_domain = inferred_domains[0] if inferred_domains else None
    domain_label = (
        primary_domain.value.replace("_", " ") if primary_domain else "general"
    )

    if backend is not None:
        # Ask the model to mimic a thoughtful non-expert's gut reaction.
        # Keeping max_tokens low (120) ensures this stays a lightweight pass.
        system_prompt = (
            "You are a curious, intelligent non-expert. "
            "Give a short intuitive (gut-feeling) answer to the question below. "
            "Do NOT use technical jargon — respond as a thoughtful layperson would. "
            "Keep the answer to 2-3 sentences. "
            "Then write one sentence of reasoning starting with 'Because '."
        )
        user_prompt = f"Question: {question}\n\nAnswer:"
        try:
            raw = backend.generate(system_prompt, user_prompt, max_tokens=120)
            # Split answer from reasoning if "Because " marker is present
            if "Because " in raw:
                parts = raw.split("Because ", 1)
                intuitive_answer = parts[0].strip()
                reasoning = "Because " + parts[1].strip()
            else:
                intuitive_answer = raw.strip()
                reasoning = (
                    f"Auto-generated intuition based on {domain_label} domain analysis."
                )
        except Exception:
            # LLM call failed — fall back gracefully to template answer
            intuitive_answer = _template_answer(question, domain_label)
            reasoning = (
                f"Auto-generated intuition based on {domain_label} domain analysis."
            )
    else:
        intuitive_answer = _template_answer(question, domain_label)
        reasoning = (
            f"Auto-generated intuition based on {domain_label} domain analysis."
        )

    return HumanIntuition(
        question=question,
        intuitive_answer=intuitive_answer,
        confidence=0.5,
        reasoning=reasoning,
        domain_guesses=inferred_domains[:3],
    )


def _template_answer(question: str, domain_label: str) -> str:
    """Return a domain-appropriate placeholder answer for offline auto-intuition.

    The templates are deliberately written in first-person "gut feeling" style
    to mimic how a thoughtful non-expert would frame their initial reaction.
    """
    q_lower = question.lower()

    if any(w in q_lower for w in ("what is", "what are", "define", "explain")):
        return (
            f"My intuition is that this is a core concept in {domain_label} "
            "that involves fundamental principles I have encountered before, "
            "though I am not entirely sure of the specifics."
        )
    if any(w in q_lower for w in ("how does", "how do", "how can", "how to")):
        return (
            f"I think this works through some kind of iterative or systematic process "
            f"in {domain_label}, where components interact to produce the observed effect."
        )
    if any(w in q_lower for w in ("why does", "why do", "why is", "why are", "why")):
        return (
            f"My gut says this happens because of underlying constraints or natural laws "
            f"in {domain_label} that push the system toward a stable state."
        )
    if any(w in q_lower for w in ("when", "where", "who")):
        return (
            "My intuition says the context matters a lot here — there is not a single "
            "universal answer but rather it depends on the specific situation."
        )
    return (
        f"Based on my general intuition about {domain_label}, "
        "I believe the answer involves balancing competing factors "
        "in a way that optimises for the key outcome."
    )
