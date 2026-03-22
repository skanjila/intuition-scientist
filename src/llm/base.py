"""Protocol definition for all free/open LLM backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    """Minimal interface that every LLM backend must satisfy.

    Each backend is responsible for:
    - Sending a system prompt + user message to the underlying model.
    - Returning the model's text reply as a plain string.
    - Raising ``RuntimeError`` with a clear message when unavailable.
    """

    def generate(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """Call the underlying model and return its text response.

        Parameters
        ----------
        system:
            System-level instructions / persona prompt.
        user:
            User message / question.
        max_tokens:
            Soft cap on the number of tokens to generate.

        Returns
        -------
        str
            The model's response text.

        Raises
        ------
        RuntimeError
            When the backend is unavailable (server down, missing env var, etc.).
        """
        ...
