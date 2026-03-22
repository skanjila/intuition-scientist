"""Debate Engine — structured multi-party debate harness.

This module implements the core debate harness that pits three perspectives
against each other for every question:

1. **Human intuition** — the user's own answer, reasoning, and confidence.
2. **Tool / MCP evidence** — factual grounding retrieved from web search or
   other MCP-connected tools.
3. **Domain-agent reasoning** — expert analysis from one or more domain agents
   driven by the configured LLM backend.

The engine surfaces agreements and divergences explicitly, producing a
:class:`DebateResult` that makes the human-vs-machine-vs-tool comparison
transparent and auditable.

Debate algorithm
----------------
1. Convert human intuition, tool evidence, and each agent response into
   :class:`DebatePosition` objects.
2. Generate :class:`DebateRound` instances by examining three analytical
   dimensions of the question: factual accuracy, causal reasoning, and
   practical implications.
3. Within each round, identify token-level agreements and divergences between
   the human position and the combined agent/tool positions.
4. Compute an overall *intuition accuracy* score using the same cosine-TF-IDF
   similarity used by :class:`WeighingSystem`.
5. Produce a synthesised verdict and key insights.
6. When an LLM backend is available (non-Mock), upgrade the synthesis and
   insights with model-generated text.
"""

from __future__ import annotations

import re
from typing import Optional

from src.llm.base import LLMBackend
from src.llm.mock_backend import MockBackend
from src.models import (
    AgentResponse,
    DebatePosition,
    DebateResult,
    DebateRound,
    HumanIntuition,
    SearchResult,
)


# ---------------------------------------------------------------------------
# Token-level helpers (reuse logic from weighing_system without circular import)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", text.lower()))


def _cosine_sim(text_a: str, text_b: str) -> float:
    import math

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
    norm_a = math.sqrt(sum(v**2 for v in va.values()))
    norm_b = math.sqrt(sum(v**2 for v in vb.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _top_tokens(common: set[str], n: int) -> list[str]:
    return sorted(common)[:n]


# ---------------------------------------------------------------------------
# Debate engine
# ---------------------------------------------------------------------------

_DEBATE_ASPECTS = [
    "factual accuracy and evidence",
    "causal reasoning and mechanisms",
    "practical implications and recommendations",
]


class DebateEngine:
    """Orchestrates a structured multi-party debate.

    Parameters
    ----------
    backend:
        LLM backend used to upgrade synthesis text when available.
        Defaults to :class:`MockBackend` (offline-safe).
    """

    def __init__(self, backend: Optional[LLMBackend] = None) -> None:
        self._backend: LLMBackend = backend if backend is not None else MockBackend()
        self._use_llm = backend is not None and not isinstance(backend, MockBackend)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def debate(
        self,
        intuition: HumanIntuition,
        agent_responses: list[AgentResponse],
        tool_results: list[SearchResult],
    ) -> DebateResult:
        """Run the full debate and return a :class:`DebateResult`.

        Parameters
        ----------
        intuition:
            The human's pre-captured intuition.
        agent_responses:
            Responses from all domain agents that were queried.
        tool_results:
            MCP/web-search results providing factual grounding.
        """
        if not agent_responses:
            raise ValueError("At least one agent response is required for debate.")

        # Build positions
        human_pos = self._build_human_position(intuition)
        tool_pos = self._build_tool_position(intuition.question, tool_results)
        agent_positions = [self._build_agent_position(r) for r in agent_responses]

        # Structured rounds
        all_positions = [human_pos, tool_pos] + agent_positions
        rounds = [
            self._build_round(aspect, human_pos, tool_pos, agent_positions)
            for aspect in _DEBATE_ASPECTS
        ]

        # Accuracy score
        agent_combined = " ".join(
            f"{r.answer} {r.reasoning}" for r in agent_responses
        )
        tool_combined = tool_pos.position
        consensus = f"{agent_combined} {tool_combined}"
        human_combined = f"{intuition.intuitive_answer} {intuition.reasoning}"
        raw_sim = _cosine_sim(human_combined, consensus)
        accuracy_pct = round(min(100.0, raw_sim * 100), 1)

        # Verdict and insights
        verdict, confidence = self._synthesise_verdict(
            intuition, agent_responses, tool_pos, accuracy_pct
        )
        insights = self._extract_insights(rounds, accuracy_pct)
        recommendations = self._generate_recommendations(rounds, accuracy_pct)

        return DebateResult(
            question=intuition.question,
            human_position=human_pos,
            tool_evidence=tool_pos,
            agent_positions=agent_positions,
            rounds=rounds,
            synthesized_verdict=verdict,
            verdict_confidence=confidence,
            intuition_accuracy_pct=accuracy_pct,
            key_insights=insights,
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------
    # Position builders
    # ------------------------------------------------------------------

    def _build_human_position(self, intuition: HumanIntuition) -> DebatePosition:
        return DebatePosition(
            source="human",
            position=(
                f"{intuition.intuitive_answer}"
                + (f"\n\nReasoning: {intuition.reasoning}" if intuition.reasoning else "")
            ),
            confidence=intuition.confidence,
            evidence=[],
        )

    def _build_tool_position(
        self, question: str, results: list[SearchResult]
    ) -> DebatePosition:
        if not results:
            return DebatePosition(
                source="tool",
                position="No tool/web evidence gathered (MCP disabled or no results).",
                confidence=0.0,
                evidence=[],
            )
        snippets = [r.snippet for r in results if r.snippet]
        urls = [r.url for r in results if r.url]
        combined = " ".join(snippets[:5])
        return DebatePosition(
            source="tool",
            position=combined[:1500] if combined else "No snippet content available.",
            confidence=0.6,
            evidence=urls[:5],
        )

    def _build_agent_position(self, response: AgentResponse) -> DebatePosition:
        return DebatePosition(
            source=f"agent:{response.domain.value}",
            position=(
                f"{response.answer}"
                + (f"\n\nReasoning: {response.reasoning}" if response.reasoning else "")
            ),
            confidence=response.confidence,
            evidence=response.sources,
        )

    # ------------------------------------------------------------------
    # Debate round construction
    # ------------------------------------------------------------------

    def _build_round(
        self,
        aspect: str,
        human_pos: DebatePosition,
        tool_pos: DebatePosition,
        agent_positions: list[DebatePosition],
    ) -> DebateRound:
        all_positions = [human_pos, tool_pos] + agent_positions

        human_tokens = _tokenize(human_pos.position)
        machine_text = " ".join(p.position for p in [tool_pos] + agent_positions)
        machine_tokens = _tokenize(machine_text)

        common = human_tokens & machine_tokens
        human_only = human_tokens - machine_tokens
        machine_only = machine_tokens - human_tokens

        agreements = _top_tokens(common, 6)
        divergences: list[str] = []
        if human_only:
            divergences.append(
                f"Human emphasises: {', '.join(_top_tokens(human_only, 4))}"
            )
        if machine_only:
            divergences.append(
                f"Agents/tools emphasise: {', '.join(_top_tokens(machine_only, 4))}"
            )

        synthesis = self._round_synthesis(aspect, agreements, divergences)

        return DebateRound(
            aspect=aspect,
            positions=all_positions,
            agreements=agreements,
            disagreements=divergences,
            round_synthesis=synthesis,
        )

    def _round_synthesis(
        self, aspect: str, agreements: list[str], divergences: list[str]
    ) -> str:
        if self._use_llm:
            prompt = (
                f"Debate aspect: {aspect}\n\n"
                f"Points of agreement: {', '.join(agreements) if agreements else 'none identified'}\n"
                f"Points of divergence:\n"
                + ("\n".join(f"  - {d}" for d in divergences) if divergences else "  none\n")
                + "\n\nIn 2 sentences, summarise what this round established about "
                f"the aspect '{aspect}' and what remains unresolved."
            )
            result = self._call_llm(prompt, max_tokens=120)
            if result:
                return result
        # Fallback
        parts = [f"Round on '{aspect}':"]
        if agreements:
            parts.append(f"Consensus on: {', '.join(agreements[:4])}.")
        if divergences:
            parts.append(" | ".join(divergences))
        else:
            parts.append("No major divergences detected.")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Verdict and insight generation
    # ------------------------------------------------------------------

    def _synthesise_verdict(
        self,
        intuition: HumanIntuition,
        responses: list[AgentResponse],
        tool_pos: DebatePosition,
        accuracy_pct: float,
    ) -> tuple[str, float]:
        if self._use_llm:
            agent_summaries = "\n".join(
                f"  [{r.domain.value}] {r.answer[:250]}" for r in responses
            )
            tool_summary = tool_pos.position[:400]
            prompt = (
                f"Question: {intuition.question}\n\n"
                f"Human intuition (confidence {intuition.confidence:.0%}):\n"
                f"{intuition.intuitive_answer}\n\n"
                f"Tool/web evidence:\n{tool_summary}\n\n"
                f"Domain-expert answers:\n{agent_summaries}\n\n"
                f"Human intuition accuracy vs. expert+tool consensus: {accuracy_pct:.1f}%\n\n"
                "Write a balanced, authoritative synthesised verdict (under 350 words) that:\n"
                "1. Integrates the strongest expert insights and tool evidence.\n"
                "2. Acknowledges where the human intuition was correct.\n"
                "3. Corrects or expands where intuition fell short.\n"
                "4. Ends with one concrete actionable takeaway."
            )
            text = self._call_llm(prompt, max_tokens=450)
            if text:
                confidence = min(1.0, 0.5 + accuracy_pct / 200)
                return text, round(confidence, 2)

        # Fallback (no LLM)
        best = max(responses, key=lambda r: r.confidence)
        lines = [
            "=== DEBATE VERDICT ===\n",
            f"Leading expert view [{best.domain.value.replace('_', ' ').title()}]:\n"
            f"{best.answer}\n",
        ]
        if tool_pos.position and "No tool" not in tool_pos.position:
            lines.append(f"\nTool/web grounding:\n{tool_pos.position[:300]}\n")
        lines.append(
            f"\nHuman intuition aligned {accuracy_pct:.1f}% with expert+tool consensus."
        )
        if accuracy_pct >= 70:
            lines.append(" Your intuition was well-calibrated on this question.")
        elif accuracy_pct >= 40:
            lines.append(" Your intuition captured key concepts but missed important nuances.")
        else:
            lines.append(
                " Your intuition diverged significantly from expert+tool consensus."
            )
        confidence = min(1.0, 0.5 + accuracy_pct / 200)
        return "\n".join(lines), round(confidence, 2)

    def _extract_insights(
        self, rounds: list[DebateRound], accuracy_pct: float
    ) -> list[str]:
        insights: list[str] = []
        # One insight per round that had disagreements
        for rnd in rounds:
            if rnd.disagreements:
                insights.append(
                    f"On '{rnd.aspect}': {rnd.disagreements[0]}"
                )
        if accuracy_pct >= 70:
            insights.append(
                "Strong human–expert alignment suggests well-developed intuition "
                "on this topic."
            )
        elif accuracy_pct < 40:
            insights.append(
                "Significant divergence between human intuition and expert+tool "
                "consensus — targeted study recommended."
            )
        return insights[:5]

    def _generate_recommendations(
        self, rounds: list[DebateRound], accuracy_pct: float
    ) -> list[str]:
        recs: list[str] = []
        # Surface topics the human missed
        missed: list[str] = []
        for rnd in rounds:
            for div in rnd.disagreements:
                if div.startswith("Agents/tools emphasise:"):
                    tokens = div.replace("Agents/tools emphasise:", "").strip()
                    missed.extend(tokens.split(", ")[:2])
        if missed:
            recs.append(
                f"Deepen understanding of: {', '.join(missed[:4])}."
            )
        if accuracy_pct < 40:
            recs.append(
                "Review foundational concepts in the domains where divergence was greatest."
            )
        if accuracy_pct >= 70:
            recs.append(
                "Your intuition is well-calibrated — challenge it further with edge cases."
            )
        if not recs:
            recs.append(
                "Continue building intuition by exploring multi-domain questions."
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
