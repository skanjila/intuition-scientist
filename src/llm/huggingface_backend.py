"""HuggingFace backend — free inference API and local transformers pipeline.

Supports two modes:
1. HuggingFace Inference API (free tier, rate-limited)
   Requires: HUGGINGFACE_API_KEY environment variable
2. Local transformers pipeline (zero API cost, requires GPU/CPU + model download)
   Requires: pip install transformers torch

Environment variables
---------------------
HUGGINGFACE_API_KEY  — API token from https://huggingface.co/settings/tokens
                       Required for API mode. Free accounts have rate limits.
HF_MODEL_ID          — Default model ID (overridden by constructor argument).
                       Default: "meta-llama/Meta-Llama-3-8B-Instruct"

Usage example
-------------
    # API mode (requires HUGGINGFACE_API_KEY env var)
    from src.llm.huggingface_backend import HuggingFaceBackend
    backend = HuggingFaceBackend(model_id="microsoft/Phi-3-mini-4k-instruct")
    response = backend.generate("You are a helpful assistant.", "What is Python?")

    # Local pipeline mode (requires transformers + torch)
    backend = HuggingFaceBackend(
        model_id="microsoft/Phi-3-mini-4k-instruct",
        use_local_pipeline=True,
        device="cuda",  # or "cpu" or "auto"
    )
    response = backend.generate("You are a helpful assistant.", "Explain recursion.")
"""

from __future__ import annotations

import os
import json
from typing import Optional

from src.llm.base import LLMBackend

_DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
_API_URL_TEMPLATE = "https://api-inference.huggingface.co/models/{model_id}"


class HuggingFaceBackend(LLMBackend):
    """HuggingFace backend supporting both API and local pipeline modes.

    Parameters
    ----------
    model_id: Optional model ID. Defaults to HF_MODEL_ID env var or meta-llama/Meta-Llama-3-8B-Instruct.
    api_key: HuggingFace API token. Defaults to HUGGINGFACE_API_KEY env var.
    use_local_pipeline: If True, uses local transformers pipeline instead of API.
    device: Device for local pipeline ("auto", "cuda", "cpu", "mps").
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        use_local_pipeline: bool = False,
        device: str = "auto",
    ) -> None:
        self._model_id = model_id or os.environ.get("HF_MODEL_ID", _DEFAULT_MODEL)
        self._api_key = api_key or os.environ.get("HUGGINGFACE_API_KEY", "")
        self._use_local = use_local_pipeline
        self._device = device
        self._pipeline = None  # lazy-loaded

    def generate(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """Generate a response.

        In API mode: POSTs to HuggingFace Inference API.
        In local mode: uses transformers.pipeline.
        Falls back to a mock response if both fail.
        """
        if self._use_local:
            return self._local_generate(system, user, max_tokens)
        return self._api_generate(system, user, max_tokens)

    def _api_generate(self, system: str, user: str, max_tokens: int) -> str:
        """Call the HuggingFace Inference API."""
        try:
            import urllib.request
            import urllib.error

            prompt = f"<system>\n{system}\n</system>\n<user>\n{user}\n</user>"
            payload = json.dumps({
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "return_full_text": False,
                }
            }).encode("utf-8")

            url = _API_URL_TEMPLATE.format(model_id=self._model_id)
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            # HF API returns a list of generated texts
            if isinstance(data, list) and data:
                return data[0].get("generated_text", str(data[0]))
            return str(data)

        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str or "rate" in err_str.lower():
                return self._mock_response(user, "HuggingFace API rate limit reached. Try again later.")
            return self._mock_response(user, f"HuggingFace API error: {err_str}")

    def _local_generate(self, system: str, user: str, max_tokens: int) -> str:
        """Use a local transformers pipeline."""
        try:
            if self._pipeline is None:
                from transformers import pipeline  # type: ignore
                self._pipeline = pipeline(
                    "text-generation",
                    model=self._model_id,
                    device_map=self._device if self._device != "auto" else None,
                )

            prompt = f"<system>\n{system}\n</system>\n<user>\n{user}\n</user>"
            outputs = self._pipeline(
                prompt,
                max_new_tokens=max_tokens,
                return_full_text=False,
            )
            if outputs:
                return outputs[0].get("generated_text", "")
            return ""
        except ImportError:
            return self._mock_response(
                user,
                "transformers/torch not installed. Run: pip install transformers torch"
            )
        except Exception as exc:
            return self._mock_response(user, f"Local pipeline error: {exc}")

    @staticmethod
    def _mock_response(question: str, error: str = "") -> str:
        note = f" ({error})" if error else " (HuggingFace unavailable — offline mode)"
        return json.dumps({
            "answer": f"[HuggingFace] Analysis of '{question[:100]}' required.{note}",
            "reasoning": "HuggingFace backend unavailable.",
            "confidence": 0.3,
            "sources": [],
        })
