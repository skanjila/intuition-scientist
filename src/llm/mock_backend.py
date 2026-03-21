"""Mock LLM backend — used for offline/CI runs (no model required)."""

from __future__ import annotations

import json


class MockBackend:
    """Returns deterministic mock responses without any network calls.

    Useful for CI pipelines and unit tests where real model servers are
    unavailable.
    """

    def generate(self, system: str, user: str, max_tokens: int = 1024) -> str:
        # Extract a hint from the user message so the mock is mildly informative
        snippet = user[:80].replace("\n", " ").strip()
        return json.dumps(
            {
                "answer": f"[mock] A domain-expert analysis: {snippet}",
                "reasoning": "Mock backend — no LLM required.",
                "confidence": 0.3,
                "sources": [],
            }
        )
