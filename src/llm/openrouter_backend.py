"""OpenRouter backend — routes to many free/open models.

Requires environment variable:
    OPENROUTER_API_KEY — obtain from https://openrouter.ai

OpenRouter exposes an OpenAI-compatible API.  Many models are free (marked
``:free`` in the model ID or zero cost per token).

Example free model IDs:
    meta-llama/llama-3.1-8b-instruct:free
    mistralai/mistral-7b-instruct:free
    google/gemma-2-9b-it:free
    qwen/qwen-2.5-7b-instruct:free

See https://openrouter.ai/models for the full list.
"""

from __future__ import annotations

import json
import os

import httpx


_OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_TIMEOUT = 60.0


class OpenRouterBackend:
    """Calls the OpenRouter inference API.

    Parameters
    ----------
    model:
        OpenRouter model ID, e.g.
        ``"meta-llama/llama-3.1-8b-instruct:free"``.
    api_key:
        OpenRouter API key — falls back to ``OPENROUTER_API_KEY`` env var.
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
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.timeout = timeout

    def generate(self, system: str, user: str, max_tokens: int = 1024) -> str:
        if not self._api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY environment variable is not set. "
                "Obtain a free key at https://openrouter.ai"
            )
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/skanjila/intuition-scientist",
            "X-Title": "Intuition Scientist",
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
                _OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"OpenRouter API returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc
        except (KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Unexpected OpenRouter response format: {exc}"
            ) from exc
