"""Groq hosted free-tier backend.

Requires environment variable:
    GROQ_API_KEY — obtain from https://console.groq.com

Example model IDs (free tier as of 2025):
    llama-3.1-8b-instant
    llama-3.3-70b-versatile
    mixtral-8x7b-32768
    gemma2-9b-it
"""

from __future__ import annotations

import json
import os

import httpx


_GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
_DEFAULT_TIMEOUT = 60.0


class GroqBackend:
    """Calls the Groq inference API (free-tier models).

    Parameters
    ----------
    model:
        Groq model ID, e.g. ``"llama-3.1-8b-instant"``.
    api_key:
        Groq API key — falls back to ``GROQ_API_KEY`` env var.
    timeout:
        HTTP timeout in seconds (default: 60).
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.timeout = timeout

    def generate(self, system: str, user: str, max_tokens: int = 1024) -> str:
        if not self._api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. "
                "Obtain a free key at https://console.groq.com"
            )
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
        }
        try:
            response = httpx.post(
                _GROQ_API_URL, headers=headers, json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Groq API returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc
        except (KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unexpected Groq response format: {exc}") from exc
