"""llama.cpp backend — calls a locally running llama.cpp server.

The llama.cpp server exposes an OpenAI-compatible ``/v1/chat/completions``
endpoint when started with ``llama-server`` (formerly ``llama.cpp/server``).

Start the server with:
    llama-server -m models/your-model.gguf --port 8080

Environment variable override:
    LLAMACPP_BASE_URL — default: ``http://localhost:8080``
"""

from __future__ import annotations

import json
import os

import httpx


_DEFAULT_BASE_URL = "http://localhost:8080"
_DEFAULT_TIMEOUT = 180.0


class LlamaCppBackend:
    """Talks to a locally running llama.cpp server (OpenAI-compat API).

    Parameters
    ----------
    model_path:
        Path to the GGUF model file — used only as a display label; the server
        must already be started with the model loaded.
    base_url:
        Override the server URL (also reads ``LLAMACPP_BASE_URL`` env var).
    timeout:
        HTTP timeout in seconds (default: 180).
    """

    def __init__(
        self,
        model_path: str,
        base_url: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.model_path = model_path
        self.base_url = (
            base_url
            or os.environ.get("LLAMACPP_BASE_URL", _DEFAULT_BASE_URL)
        ).rstrip("/")
        self.timeout = timeout

    def generate(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """Send a chat request to the llama.cpp server and return the reply."""
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "stream": False,
        }
        try:
            response = httpx.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"llama.cpp server not reachable at {self.base_url}. "
                "Start it with: llama-server -m <model.gguf> --port 8080"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"llama.cpp server returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc
        except (KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Unexpected llama.cpp server response format: {exc}"
            ) from exc
