"""Together AI hosted free-tier backend.

Requires environment variable:
    TOGETHER_API_KEY — obtain from https://api.together.xyz

Example free-tier model IDs:
    meta-llama/Llama-3.1-8B-Instruct-Turbo
    mistralai/Mixtral-8x7B-Instruct-v0.1
    Qwen/Qwen2.5-7B-Instruct-Turbo
"""

from __future__ import annotations

import json
import os

import httpx


_TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
_DEFAULT_TIMEOUT = 60.0


class TogetherBackend:
    """Calls the Together AI inference API.

    Parameters
    ----------
    model:
        Together model ID, e.g. ``"meta-llama/Llama-3.1-8B-Instruct-Turbo"``.
    api_key:
        Together API key — falls back to ``TOGETHER_API_KEY`` env var.
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
        self._api_key = api_key or os.environ.get("TOGETHER_API_KEY", "")
        self.timeout = timeout

    def generate(self, system: str, user: str, max_tokens: int = 1024) -> str:
        if not self._api_key:
            raise RuntimeError(
                "TOGETHER_API_KEY environment variable is not set. "
                "Obtain a free key at https://api.together.xyz"
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
                _TOGETHER_API_URL, headers=headers, json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Together API returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc
        except (KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Unexpected Together response format: {exc}"
            ) from exc
