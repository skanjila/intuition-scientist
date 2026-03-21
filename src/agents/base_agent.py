"""Abstract base class for all domain agents.

Every domain agent inherits from ``BaseAgent`` and must implement the
``_build_system_prompt`` method.  The public ``answer`` method handles:

1. Optionally querying the internet via the MCP client.
2. Constructing a domain-specific prompt.
3. Calling the configured LLM backend (free/open models only) to produce a
   structured answer.
4. Returning an ``AgentResponse`` dataclass.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Optional

from src.llm.base import LLMBackend
from src.llm.mock_backend import MockBackend
from src.mcp.mcp_client import MCPClient
from src.models import AgentResponse, Domain, SearchResult


class BaseAgent(ABC):
    """Base for all domain-specific agents."""

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
        # Keep llm_provider/model attributes so external code that reads them
        # does not break, but they no longer drive LLM selection.
        self.llm_provider = "mock" if backend is None else "custom"
        self.model = model or "mock"
        # _llm_client kept as None so legacy checks (``if self._llm_client``)
        # fall through to the mock path in any code we have not yet updated.
        self._llm_client = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def answer(self, question: str) -> AgentResponse:
        """Produce a domain-expert answer for *question*.

        Steps:
        1. Search the internet for relevant context (if MCP client available).
        2. Build the full prompt.
        3. Call the LLM.
        4. Return a structured ``AgentResponse``.
        """
        search_results: list[SearchResult] = []
        mcp_context = ""

        if self.mcp_client is not None:
            query = f"{self.domain.value.replace('_', ' ')} {question}"
            try:
                search_results = self.mcp_client.search(query, num_results=3)
                mcp_context = self._format_search_context(search_results)
            except Exception:
                mcp_context = ""

        raw = self._call_llm(question, mcp_context)
        return self._parse_response(question, raw, search_results, mcp_context)

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_system_prompt(self) -> str:
        """Return the system-level prompt that gives this agent its persona."""

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
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        question: str,
        raw: str,
        search_results: list[SearchResult],
        mcp_context: str,
    ) -> AgentResponse:
        """Try to parse a JSON blob from *raw*; fall back to plain text."""
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
        # Fall back: treat the whole text as the answer
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
        note = f" (LLM unavailable: {error})" if error else " (LLM unavailable – offline mode)"
        return json.dumps(
            {
                "answer": (
                    f"[{self.domain.value}] A domain-expert analysis of '{question}' "
                    f"is required.{note}"
                ),
                "reasoning": "No LLM client configured or reachable.",
                "confidence": 0.3,
                "sources": [],
            }
        )
