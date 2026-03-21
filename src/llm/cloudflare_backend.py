"""Cloudflare Workers AI backend (free tier).

Requires environment variables:
    CLOUDFLARE_ACCOUNT_ID — your Cloudflare account ID
    CLOUDFLARE_API_TOKEN  — a Workers AI API token

Example free-tier model IDs:
    @cf/meta/llama-3.1-8b-instruct
    @cf/mistral/mistral-7b-instruct-v0.1
    @cf/qwen/qwen1.5-14b-chat-awq

See: https://developers.cloudflare.com/workers-ai/models/
"""

from __future__ import annotations

import json
import os

import httpx


_DEFAULT_TIMEOUT = 60.0


class CloudflareBackend:
    """Calls Cloudflare Workers AI (free tier included with Cloudflare account).

    Parameters
    ----------
    model:
        Cloudflare model ID, e.g. ``"@cf/meta/llama-3.1-8b-instruct"``.
    account_id:
        Cloudflare account ID — falls back to ``CLOUDFLARE_ACCOUNT_ID`` env var.
    api_token:
        Cloudflare API token — falls back to ``CLOUDFLARE_API_TOKEN`` env var.
    timeout:
        HTTP timeout in seconds (default: 60).
    """

    def __init__(
        self,
        model: str,
        account_id: str | None = None,
        api_token: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.model = model
        self._account_id = account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
        self._api_token = api_token or os.environ.get("CLOUDFLARE_API_TOKEN", "")
        self.timeout = timeout

    def generate(self, system: str, user: str, max_tokens: int = 1024) -> str:
        if not self._account_id or not self._api_token:
            raise RuntimeError(
                "CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN environment "
                "variables must be set to use the Cloudflare Workers AI backend. "
                "See https://developers.cloudflare.com/workers-ai/"
            )
        url = (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{self._account_id}/ai/run/{self.model}"
        )
        headers = {
            "Authorization": f"Bearer {self._api_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
        }
        try:
            response = httpx.post(
                url, headers=headers, json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            # Cloudflare wraps the result in {"result": {"response": "..."}}
            return data["result"]["response"]
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Cloudflare Workers AI returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc
        except (KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Unexpected Cloudflare Workers AI response format: {exc}"
            ) from exc
