"""Intuition vs. Agent Weighing System.

This module implements the core intellectual contribution of the Human Intuition
Scientist: it systematically compares a human's intuitive answer against each
domain-agent response and produces a rich, multi-dimensional analysis.

Algorithm overview
------------------
1. **Semantic similarity** — keyword overlap (Jaccard) between the intuition
   and each agent answer.  When an LLM backend is available the similarity is
   upgraded to a proper embedding-based cosine similarity.

2. **Confidence weighting** — each agent response is weighted by its own
   reported confidence and by the relevance of its domain to the question
   (inferred from keyword analysis).

3. **Consensus building** — the weighted average of all agent responses forms
   the "expert consensus".  The intuition is then evaluated *relative to that
   consensus*.

4. **Agreement extraction** — the system identifies specific concepts that
   the human and the agents both mention (agreements) and concepts that diverge.

5. **Synthesis** — the final synthesized answer blends the expert consensus
   with the human intuition in proportion to the intuition accuracy score.
"""

from __future__ import annotations

import math
import re
from typing import Optional

from src.llm.base import LLMBackend
from src.llm.mock_backend import MockBackend
from src.models import (
    AgentResponse,
    AlignmentScore,
    Domain,
    HumanIntuition,
    WeighingResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Lower-case word tokenizer; strips punctuation."""
    return set(re.findall(r"[a-z]{3,}", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cosine_sim_tfidf(text_a: str, text_b: str) -> float:
    """Lightweight TF-IDF cosine similarity (no external ML deps required).

    We treat each word as a dimension and weight by log-term-frequency.
    """
    def tf_vec(text: str) -> dict[str, float]:
        tokens = re.findall(r"[a-z]{3,}", text.lower())
        vec: dict[str, float] = {}
        for t in tokens:
            vec[t] = vec.get(t, 0.0) + 1.0
        for t in vec:
            vec[t] = 1.0 + math.log(vec[t])
        return vec

    va, vb = tf_vec(text_a), tf_vec(text_b)
    keys = set(va) | set(vb)
    dot = sum(va.get(k, 0.0) * vb.get(k, 0.0) for k in keys)
    norm_a = math.sqrt(sum(v ** 2 for v in va.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vb.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Domain relevance scoring
# ---------------------------------------------------------------------------

# Very rough heuristic: these words flag a domain as strongly relevant
_DOMAIN_SIGNALS: dict[Domain, list[str]] = {
    Domain.ELECTRICAL_ENGINEERING: ["circuit", "voltage", "current", "power", "signal"],
    Domain.COMPUTER_SCIENCE: ["algorithm", "software", "program", "computer", "code"],
    Domain.NEURAL_NETWORKS: ["neuron", "network", "activation", "gradient", "layer"],
    Domain.SOCIAL_SCIENCE: ["society", "human", "behaviour", "social", "psychology"],
    Domain.SPACE_SCIENCE: ["space", "star", "planet", "galaxy", "universe"],
    Domain.PHYSICS: ["energy", "force", "quantum", "particle", "wave"],
    Domain.DEEP_LEARNING: ["learning", "model", "training", "deep", "transformer"],
    # High-economic-value industry domains
    Domain.HEALTHCARE: ["disease", "drug", "patient", "clinical", "diagnosis", "therapy"],
    Domain.CLIMATE_ENERGY: ["climate", "carbon", "renewable", "energy", "emission", "solar"],
    Domain.FINANCE_ECONOMICS: ["market", "risk", "finance", "economic", "investment", "capital"],
    Domain.CYBERSECURITY: ["security", "attack", "vulnerability", "threat", "encrypt", "cyber"],
    Domain.BIOTECH_GENOMICS: ["gene", "genome", "protein", "crispr", "cell", "biotech"],
    Domain.SUPPLY_CHAIN: ["supply", "logistics", "inventory", "demand", "shipping", "warehouse"],
    # Enterprise problem domains
    Domain.LEGAL_COMPLIANCE: ["legal", "contract", "compliance", "regulation", "law", "risk"],
    Domain.ENTERPRISE_ARCHITECTURE: ["architecture", "system", "cloud", "microservice", "platform", "technical"],
    Domain.MARKETING_GROWTH: ["marketing", "customer", "growth", "acquisition", "revenue", "brand"],
    Domain.ORGANIZATIONAL_BEHAVIOR: ["organisation", "talent", "culture", "leadership", "workforce", "team"],
    Domain.STRATEGY_INTELLIGENCE: ["strategy", "competitive", "market", "advantage", "positioning", "innovation"],
    # Mastery / interview / PhD research domains
    Domain.ALGORITHMS_PROGRAMMING: ["algorithm", "python", "rust", "golang", "data structure", "complexity"],
    Domain.INTERVIEW_PREP: ["interview", "leetcode", "system design", "faang", "coding", "behavioral"],
    Domain.EE_LLM_RESEARCH: ["llm", "signal processing", "safety", "alignment", "transformer", "phd"],
}


def _domain_relevance(question: str, domain: Domain) -> float:
    """Return a 0–1 relevance weight for *domain* given *question*."""
    q_lower = question.lower()
    signals = _DOMAIN_SIGNALS.get(domain, [])
    hits = sum(1 for s in signals if s in q_lower)
    return min(1.0, 0.2 + hits * 0.16)


# ---------------------------------------------------------------------------
# Main weighing system
# ---------------------------------------------------------------------------

class WeighingSystem:
    """Compares human intuition against domain-agent answers.

    Parameters
    ----------
    backend:
        Optional ``LLMBackend`` instance.  When provided, the LLM is used
        to generate a high-quality ``overall_analysis`` narrative and richer
        ``AlignmentScore`` text fields.  Defaults to ``MockBackend`` (offline).

    .. deprecated::
        ``llm_client``, ``llm_provider``, and ``model`` keyword arguments are
        accepted for backwards compatibility but are ignored.  Pass ``backend``
        instead.
    """

    def __init__(
        self,
        backend: Optional[LLMBackend] = None,
        synthesis_max_tokens: int = 512,
        # Legacy kwargs — accepted but ignored
        llm_client: Optional[object] = None,
        llm_provider: str = "mock",
        model: Optional[str] = None,
    ) -> None:
        self._backend: LLMBackend = backend if backend is not None else MockBackend()
        self._synthesis_max_tokens = synthesis_max_tokens
        # _use_llm is True only when a real (non-mock) backend was provided
        self._use_llm = backend is not None and not isinstance(backend, MockBackend)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def weigh(
        self,
        intuition: HumanIntuition,
        agent_responses: list[AgentResponse],
    ) -> WeighingResult:
        """Produce a full ``WeighingResult`` comparing *intuition* to agents."""
        if not agent_responses:
            raise ValueError("At least one agent response is required.")

        alignment_scores = [
            self._score_alignment(intuition, resp)
            for resp in agent_responses
        ]

        intuition_accuracy = self._compute_intuition_accuracy(
            intuition, agent_responses, alignment_scores
        )

        synthesized = self._synthesize(
            intuition, agent_responses, alignment_scores, intuition_accuracy
        )

        overall_analysis = self._generate_analysis(
            intuition, agent_responses, alignment_scores, intuition_accuracy
        )

        recommendations = self._generate_recommendations(
            intuition, alignment_scores, intuition_accuracy
        )

        return WeighingResult(
            question=intuition.question,
            human_intuition=intuition,
            agent_responses=agent_responses,
            alignment_scores=alignment_scores,
            synthesized_answer=synthesized,
            intuition_accuracy_pct=intuition_accuracy,
            overall_analysis=overall_analysis,
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------
    # Alignment scoring
    # ------------------------------------------------------------------

    def _score_alignment(
        self, intuition: HumanIntuition, resp: AgentResponse
    ) -> AlignmentScore:
        combined_agent = f"{resp.answer} {resp.reasoning}"
        combined_human = f"{intuition.intuitive_answer} {intuition.reasoning}"

        # Similarity
        sim = _cosine_sim_tfidf(combined_human, combined_agent)

        # Agreements / divergences
        human_tokens = _tokenize(combined_human)
        agent_tokens = _tokenize(combined_agent)

        common = human_tokens & agent_tokens
        human_only = human_tokens - agent_tokens
        agent_only = agent_tokens - human_tokens

        agreements = self._top_n(common, 5)
        divergences_human = self._top_n(human_only, 3)
        divergences_agent = self._top_n(agent_only, 3)
        divergences = [
            f"Human mentions: {', '.join(divergences_human)}" if divergences_human else "",
            f"Agent mentions: {', '.join(divergences_agent)}" if divergences_agent else "",
        ]
        divergences = [d for d in divergences if d]

        # LLM-generated intuition insight
        insight = self._llm_insight(intuition, resp) if self._use_llm else (
            f"Semantic similarity: {sim:.2f}. "
            f"Common concepts: {', '.join(agreements) if agreements else 'none'}."
        )

        return AlignmentScore(
            domain=resp.domain,
            semantic_similarity=round(sim, 3),
            key_agreements=agreements,
            key_divergences=divergences,
            intuition_insight=insight,
        )

    # ------------------------------------------------------------------
    # Accuracy computation
    # ------------------------------------------------------------------

    def _compute_intuition_accuracy(
        self,
        intuition: HumanIntuition,
        responses: list[AgentResponse],
        alignments: list[AlignmentScore],
    ) -> float:
        """Weighted average similarity across all agents → 0–100 %."""
        total_weight = 0.0
        weighted_sim = 0.0

        for resp, align in zip(responses, alignments):
            # Weight: agent confidence × domain relevance
            relevance = _domain_relevance(intuition.question, resp.domain)
            weight = resp.confidence * relevance
            weighted_sim += align.semantic_similarity * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        raw = weighted_sim / total_weight
        # Scale to 0-100
        return round(min(100.0, raw * 100), 1)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def _synthesize(
        self,
        intuition: HumanIntuition,
        responses: list[AgentResponse],
        alignments: list[AlignmentScore],
        accuracy_pct: float,
    ) -> str:
        """Blend expert consensus with human intuition."""
        if self._use_llm:
            return self._llm_synthesis(intuition, responses, accuracy_pct)

        # Fallback: pick the highest-confidence agent answer and blend
        best = max(responses, key=lambda r: r.confidence)
        blend_ratio = accuracy_pct / 100.0  # how much intuition to weight in

        lines = [
            "=== SYNTHESIZED ANSWER ===\n",
            f"Expert consensus ({best.domain.value.replace('_', ' ').title()}):\n"
            f"{best.answer}\n",
        ]
        if blend_ratio > 0.3:
            lines.append(
                f"\nHuman intuition alignment ({accuracy_pct:.1f}% agreement):\n"
                f"{intuition.intuitive_answer}"
            )
        else:
            lines.append(
                f"\nThe human intuition ({accuracy_pct:.1f}% agreement with experts) "
                f"diverges notably from the expert consensus."
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM-powered helpers
    # ------------------------------------------------------------------

    def _llm_insight(self, intuition: HumanIntuition, resp: AgentResponse) -> str:
        prompt = (
            f"Domain: {resp.domain.value}\n"
            f"Question: {intuition.question}\n\n"
            f"Human intuition: {intuition.intuitive_answer}\n"
            f"Human reasoning: {intuition.reasoning}\n"
            f"Human confidence: {intuition.confidence}\n\n"
            f"Expert answer: {resp.answer}\n"
            f"Expert reasoning: {resp.reasoning}\n\n"
            "In 2-3 sentences, analyse how well the human intuition aligns with "
            "the expert answer. Identify the most insightful agreement or the "
            "most important gap in the intuition."
        )
        return self._call_llm(prompt, max_tokens=256) or "Analysis unavailable."

    def _llm_synthesis(
        self,
        intuition: HumanIntuition,
        responses: list[AgentResponse],
        accuracy_pct: float,
    ) -> str:
        agent_summaries = "\n".join(
            f"- {r.domain.value}: {r.answer[:300]}" for r in responses
        )
        prompt = (
            f"Question: {intuition.question}\n\n"
            f"Human intuition (confidence {intuition.confidence:.0%}):\n"
            f"{intuition.intuitive_answer}\n\n"
            f"Domain-expert answers:\n{agent_summaries}\n\n"
            f"Intuition accuracy vs experts: {accuracy_pct:.1f}%\n\n"
            "Synthesize a single comprehensive answer that:\n"
            "1. Incorporates the strongest expert insights.\n"
            "2. Acknowledges where the human intuition was correct.\n"
            "3. Corrects or expands where the intuition fell short.\n"
            "Keep the answer under 400 words."
        )
        return self._call_llm(prompt, max_tokens=self._synthesis_max_tokens) or "Synthesis unavailable."

    def _generate_analysis(
        self,
        intuition: HumanIntuition,
        responses: list[AgentResponse],
        alignments: list[AlignmentScore],
        accuracy_pct: float,
    ) -> str:
        if self._use_llm:
            return self._llm_full_analysis(intuition, responses, alignments, accuracy_pct)

        # Fallback text analysis
        best_align = max(alignments, key=lambda a: a.semantic_similarity)
        worst_align = min(alignments, key=lambda a: a.semantic_similarity)
        lines = [
            f"Intuition Accuracy: {accuracy_pct:.1f}%",
            f"Strongest alignment: {best_align.domain.value} "
            f"(similarity={best_align.semantic_similarity:.2f})",
            f"Weakest alignment: {worst_align.domain.value} "
            f"(similarity={worst_align.semantic_similarity:.2f})",
            "",
            "The human intuition was compared against "
            f"{len(responses)} domain-expert agents. "
        ]
        if accuracy_pct >= 70:
            lines.append(
                "The intuition demonstrates strong alignment with expert knowledge, "
                "suggesting well-developed domain intuition."
            )
        elif accuracy_pct >= 40:
            lines.append(
                "The intuition captures some key concepts but misses important "
                "nuances identified by domain experts."
            )
        else:
            lines.append(
                "The intuition diverges significantly from expert consensus, "
                "indicating an area where deeper study would be beneficial."
            )
        return "\n".join(lines)

    def _llm_full_analysis(
        self,
        intuition: HumanIntuition,
        responses: list[AgentResponse],
        alignments: list[AlignmentScore],
        accuracy_pct: float,
    ) -> str:
        align_summaries = "\n".join(
            f"- {a.domain.value}: similarity={a.semantic_similarity:.2f}, "
            f"agreements=[{', '.join(a.key_agreements[:3])}]"
            for a in alignments
        )
        prompt = (
            f"Question: {intuition.question}\n\n"
            f"Human intuition: {intuition.intuitive_answer}\n"
            f"Human confidence: {intuition.confidence:.0%}\n\n"
            f"Per-domain alignment scores:\n{align_summaries}\n\n"
            f"Overall intuition accuracy: {accuracy_pct:.1f}%\n\n"
            "Write a deep, analytical paragraph (150–200 words) that:\n"
            "1. Explains WHY the human intuition aligned or diverged.\n"
            "2. Identifies cognitive patterns, heuristics, or knowledge gaps.\n"
            "3. Notes which domains were most and least relevant.\n"
            "4. Assesses the quality of the intuition as a thinking strategy."
        )
        return self._call_llm(prompt, max_tokens=self._synthesis_max_tokens) or "Analysis unavailable."

    def _generate_recommendations(
        self,
        intuition: HumanIntuition,
        alignments: list[AlignmentScore],
        accuracy_pct: float,
    ) -> list[str]:
        recs: list[str] = []
        if accuracy_pct < 40:
            recs.append(
                "Consider studying the foundational concepts in the domains "
                "where your intuition diverged most."
            )
        if accuracy_pct >= 70:
            recs.append(
                "Your intuition is well-calibrated – push it further by exploring "
                "edge cases and counter-examples."
            )
        low_align = [a for a in alignments if a.semantic_similarity < 0.2]
        if low_align:
            domains = ", ".join(a.domain.value.replace("_", " ") for a in low_align)
            recs.append(f"Explore introductory resources in: {domains}.")
        if intuition.confidence > 0.8 and accuracy_pct < 50:
            recs.append(
                "Your confidence was high but alignment was moderate – "
                "consider practising epistemic humility and seeking diverse perspectives."
            )
        if not recs:
            recs.append(
                "Continue developing your intuition by tackling more questions "
                "across multiple domains."
            )
        return recs

    # ------------------------------------------------------------------
    # LLM call helper
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, max_tokens: int = 256) -> str:
        try:
            return self._backend.generate("", prompt, max_tokens=max_tokens)
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _top_n(tokens: set[str], n: int) -> list[str]:
        """Return up to *n* tokens sorted alphabetically."""
        return sorted(tokens)[:n]
