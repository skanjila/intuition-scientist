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
"""

from __future__ import annotations

import re
from typing import Optional

from src.models import Domain, HumanIntuition

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
