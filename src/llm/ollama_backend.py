"""Ollama local backend — calls a locally running Ollama server."""

from __future__ import annotations

import json

import httpx


_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_TIMEOUT = 120.0  # seconds — local inference can be slow


class OllamaBackend:
    """Connects to a locally running Ollama server.

    Parameters
    ----------
    model:
        Ollama model tag, e.g. ``"llama3.1:8b"`` or ``"qwen2.5:7b"``.
    base_url:
        Override the Ollama server URL (default: ``http://localhost:11434``).
    timeout:
        HTTP timeout in seconds (default: 120).
    """

    def __init__(
        self,
        model: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """Send a chat request to Ollama and return the assistant reply."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        try:
            response = httpx.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Ollama server not reachable at {self.base_url}. "
                "Make sure Ollama is running (`ollama serve`)."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        except (KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unexpected Ollama response format: {exc}") from exc
