"""Abstract base class for all domain agents.

Every domain agent inherits from ``BaseAgent`` and must implement the
``_build_system_prompt`` method.  The public ``answer`` method implements a
**dual-pipeline** that combines:

Pipeline A — Pure Intuition
    The agent calls the LLM with domain expertise only, no external context.
    This captures deep theoretical reasoning and pattern recognition.

Pipeline B — Tool / MCP Grounded
    The agent first retrieves relevant evidence via MCP web search, then calls
    the LLM with that evidence as additional context.  This captures up-to-date
    facts and cited sources.

Intelligent Weight Blending
    The two answers are combined using weights computed from:
    - Domain type (interpretive domains favour intuition; empirical / legal /
      medical domains favour tool evidence)
    - MCP result quality (richer evidence → higher tool weight)
    - Question type heuristic (factual "when/who/what" → tools; analytical
      "why/how/should" → intuition)

The ``AgentResponse`` records both the final blended answer and the weights
that produced it, giving the ``WeighingSystem`` full transparency.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Optional

from src.llm.base import LLMBackend
from src.llm.mock_backend import MockBackend
from src.mcp.mcp_client import MCPClient
from src.models import AgentResponse, Domain, SearchResult


# ---------------------------------------------------------------------------
# Per-domain base intuition weight (before MCP quality adjustment)
# ---------------------------------------------------------------------------

# Domains where the agent's deep knowledge outweighs raw web evidence
_INTUITION_HEAVY: frozenset[Domain] = frozenset({
    Domain.SOCIAL_SCIENCE,
    Domain.STRATEGY_INTELLIGENCE,
    Domain.ORGANIZATIONAL_BEHAVIOR,
    Domain.MARKETING_GROWTH,
    Domain.INTERVIEW_PREP,
    Domain.ALGORITHMS_PROGRAMMING,
    Domain.EE_LLM_RESEARCH,
    Domain.PHYSICS,
    Domain.NEURAL_NETWORKS,
    Domain.DEEP_LEARNING,
})

# Domains where verified external evidence is especially valuable
_TOOL_HEAVY: frozenset[Domain] = frozenset({
    Domain.HEALTHCARE,
    Domain.CYBERSECURITY,
    Domain.LEGAL_COMPLIANCE,
    Domain.SUPPLY_CHAIN,
    Domain.FINANCE_ECONOMICS,
    Domain.CLIMATE_ENERGY,
    Domain.BIOTECH_GENOMICS,
})

# Factual question signals → boost tool weight
_FACTUAL_RE = re.compile(
    r"\b(what is|what are|when did|who invented|how many|list|"
    r"which company|what year|where is|define)\b",
    re.IGNORECASE,
)

# Analytical question signals → boost intuition weight
_ANALYTICAL_RE = re.compile(
    r"\b(why|how does|how should|what causes|what would|"
    r"explain|analyse|analyze|compare|evaluate|design)\b",
    re.IGNORECASE,
)


class BaseAgent(ABC):
    """Base for all domain-specific agents.

    Parameters
    ----------
    mcp_client:
        MCP internet-search client.  When ``None`` the tool pipeline is
        skipped and the agent relies entirely on its trained knowledge.
    backend:
        Free/open LLM backend.  Defaults to ``MockBackend`` (offline).
    """

    #: Override in subclasses to declare which domain this agent covers.
    domain: Domain

    def __init__(
        self,
        mcp_client: Optional[MCPClient] = None,
        backend: Optional[LLMBackend] = None,
        # Deprecated parameters kept for backwards compatibility; ignored.
        llm_provider: str = "mock",
        model: Optional[str] = None,
    ) -> None:
        self.mcp_client = mcp_client
        self._backend: LLMBackend = backend if backend is not None else MockBackend()
        self.llm_provider = "mock" if backend is None else "custom"
        self.model = model or "mock"
        self._llm_client = None  # legacy attribute kept for compat

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def answer(self, question: str) -> AgentResponse:
        """Produce a domain-expert answer using the dual intuition+tool pipeline.

        Steps
        -----
        1. **Intuition path** — call LLM with domain knowledge only (no MCP).
        2. **Tool path** — retrieve MCP evidence, then call LLM with that
           context.  Skipped when MCP is unavailable or returns no results.
        3. **Weight computation** — derive ``intuition_weight`` and
           ``tool_weight`` from domain type, MCP quality, and question type.
        4. **Blend** — return the answer from whichever pipeline carries more
           weight; record both weights in the ``AgentResponse``.
        """
        # ── Pipeline A: Pure intuition (no MCP context) ──────────────────
        intuition_raw = self._call_llm(question, mcp_context="")
        intuition_data = self._extract_json(intuition_raw) or {}

        # ── MCP retrieval ─────────────────────────────────────────────────
        search_results: list[SearchResult] = []
        mcp_context = ""
        if self.mcp_client is not None:
            query = f"{self.domain.value.replace('_', ' ')} {question}"
            try:
                search_results = self.mcp_client.search(query, num_results=4)
                mcp_context = self._format_search_context(search_results)
            except Exception:
                mcp_context = ""

        # ── Compute weights ───────────────────────────────────────────────
        intuition_w, tool_w = self._compute_weights(question, search_results)

        # ── Pipeline B: Tool-grounded (only if weight is meaningful) ──────
        tool_data: dict = {}
        if tool_w >= 0.2 and mcp_context:
            tool_raw = self._call_llm(question, mcp_context)
            tool_data = self._extract_json(tool_raw) or {}

        # ── Blend and return ──────────────────────────────────────────────
        return self._blend_and_build(
            question=question,
            intuition_data=intuition_data,
            tool_data=tool_data,
            intuition_weight=intuition_w,
            tool_weight=tool_w,
            search_results=search_results,
            mcp_context=mcp_context,
        )

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_system_prompt(self) -> str:
        """Return the system-level prompt that gives this agent its persona."""

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------

    def _compute_weights(
        self,
        question: str,
        mcp_results: list[SearchResult],
    ) -> tuple[float, float]:
        """Return ``(intuition_weight, tool_weight)`` — both sum to 1.0.

        Algorithm
        ---------
        1. Start from a domain-specific base intuition weight.
        2. Adjust based on MCP evidence quality (0 results → no tool boost).
        3. Adjust based on question type (factual → tool; analytical →
           intuition).
        """
        # Step 1: domain base
        if self.domain in _INTUITION_HEAVY:
            base_intuition = 0.65
        elif self.domain in _TOOL_HEAVY:
            base_intuition = 0.40
        else:
            base_intuition = 0.55  # balanced

        # Step 2: MCP quality modifier (0.0–0.20 boost to tool)
        mcp_quality = min(1.0, len(mcp_results) / 4)  # 4 results = max quality
        tool_boost = mcp_quality * 0.20

        # Step 3: question type modifier (±0.10)
        if _FACTUAL_RE.search(question):
            type_mod = +0.10   # factual → more tool
        elif _ANALYTICAL_RE.search(question):
            type_mod = -0.10  # analytical → more intuition
        else:
            type_mod = 0.0

        raw_tool = (1.0 - base_intuition) + tool_boost + type_mod
        raw_tool = max(0.10, min(0.75, raw_tool))  # clamp to [0.10, 0.75]
        intuition_w = round(1.0 - raw_tool, 3)
        tool_w = round(raw_tool, 3)
        return intuition_w, tool_w

    # ------------------------------------------------------------------
    # Response blending
    # ------------------------------------------------------------------

    def _blend_and_build(
        self,
        question: str,
        intuition_data: dict,
        tool_data: dict,
        intuition_weight: float,
        tool_weight: float,
        search_results: list[SearchResult],
        mcp_context: str,
    ) -> AgentResponse:
        sources = [r.url for r in search_results if r.url]

        # Choose dominant pipeline for each field
        if tool_weight >= 0.2 and tool_data:
            # Blend answers: dominant pipeline provides the main text;
            # the other contributes any unique key claims via a short suffix.
            i_ans = str(intuition_data.get("answer", ""))
            t_ans = str(tool_data.get("answer", ""))

            if intuition_weight >= tool_weight:
                primary_ans = i_ans
                secondary_note = (
                    f" [Tool evidence ({tool_weight:.0%}): {t_ans[:200]}]"
                    if t_ans and t_ans != i_ans else ""
                )
            else:
                primary_ans = t_ans
                secondary_note = (
                    f" [Intuition insight ({intuition_weight:.0%}): {i_ans[:200]}]"
                    if i_ans and i_ans != t_ans else ""
                )

            blended_answer = (primary_ans + secondary_note).strip()
            blended_reasoning = str(
                tool_data.get("reasoning", "")
                or intuition_data.get("reasoning", "")
            )
            # Weighted average confidence
            i_conf = float(intuition_data.get("confidence", 0.6))
            t_conf = float(tool_data.get("confidence", 0.6))
            blended_conf = round(
                intuition_weight * i_conf + tool_weight * t_conf, 3
            )
            combined_sources = list(
                dict.fromkeys(
                    list(tool_data.get("sources", [])) + sources
                )
            )
        else:
            # Tool pipeline unavailable or negligible — use intuition only
            raw_fallback = self._mock_response(question)
            blended_answer = str(
                intuition_data.get("answer", raw_fallback)
            )
            blended_reasoning = str(intuition_data.get("reasoning", ""))
            blended_conf = float(intuition_data.get("confidence", 0.5))
            combined_sources = list(intuition_data.get("sources", [])) + sources

        return AgentResponse(
            domain=self.domain,
            answer=blended_answer,
            reasoning=blended_reasoning,
            confidence=max(0.0, min(1.0, blended_conf)),
            sources=combined_sources,
            mcp_context=mcp_context,
            intuition_weight=intuition_weight,
            tool_weight=tool_weight,
        )

    # ------------------------------------------------------------------
    # LLM wiring
    # ------------------------------------------------------------------

    def _call_llm(self, question: str, mcp_context: str) -> str:
        """Call the LLM backend and return the raw text response."""
        system_prompt = self._build_system_prompt()
        user_message = self._build_user_message(question, mcp_context)
        try:
            return self._backend.generate(system_prompt, user_message)
        except Exception as exc:
            return self._mock_response(question, error=str(exc))

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def _build_user_message(self, question: str, mcp_context: str) -> str:
        parts = [f"Question: {question}"]
        if mcp_context:
            parts.append(f"\nAdditional context from the internet:\n{mcp_context}")
        parts.append(
            "\nPlease answer in the following JSON format:\n"
            "{\n"
            '  "answer": "<concise expert answer>",\n'
            '  "reasoning": "<step-by-step reasoning>",\n'
            '  "confidence": <0.0-1.0>,\n'
            '  "sources": ["<source or reference>", ...]\n'
            "}"
        )
        return "\n".join(parts)

    @staticmethod
    def _format_search_context(results: list[SearchResult]) -> str:
        if not results:
            return ""
        lines = ["Web search results:"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. [{r.title}]({r.url})")
            if r.snippet:
                lines.append(f"   {r.snippet[:300]}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Response parsing (kept for external callers that call _parse_response)
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        question: str,
        raw: str,
        search_results: list[SearchResult],
        mcp_context: str,
    ) -> AgentResponse:
        data = self._extract_json(raw)
        sources = [r.url for r in search_results if r.url]
        if data:
            return AgentResponse(
                domain=self.domain,
                answer=str(data.get("answer", raw)),
                reasoning=str(data.get("reasoning", "")),
                confidence=float(data.get("confidence", 0.7)),
                sources=data.get("sources", sources),
                mcp_context=mcp_context,
            )
        return AgentResponse(
            domain=self.domain,
            answer=raw,
            reasoning="",
            confidence=0.6,
            sources=sources,
            mcp_context=mcp_context,
        )

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Extract the first JSON object found in *text*."""
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return None

    # ------------------------------------------------------------------
    # Mock / offline fallback
    # ------------------------------------------------------------------

    def _mock_response(self, question: str, error: str = "") -> str:
        note = (
            f" (LLM unavailable: {error})" if error
            else " (LLM unavailable – offline mode)"
        )
        return json.dumps({
            "answer": (
                f"[{self.domain.value}] A domain-expert analysis of "
                f"'{question}' is required.{note}"
            ),
            "reasoning": "No LLM client configured or reachable.",
            "confidence": 0.3,
            "sources": [],
        })

