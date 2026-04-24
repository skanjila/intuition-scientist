"""Backend registry — parse provider specs and instantiate backends.

Provider spec format:  ``<provider>:<model-or-path>``

Examples
--------
    mock                                          — offline mock (no model)
    ollama:llama3.1:8b                            — Ollama local
    llamacpp:models/llama-3.1-8b-q4_k_m.gguf     — llama.cpp local
    groq:llama-3.1-8b-instant                     — Groq hosted (free-tier)
    together:meta-llama/Llama-3.1-8B-Instruct-Turbo — Together AI
    cloudflare:@cf/meta/llama-3.1-8b-instruct     — Cloudflare Workers AI
    openrouter:meta-llama/llama-3.1-8b-instruct:free — OpenRouter

Anthropic and OpenAI are **not** supported — this project is free-only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm.base import LLMBackend


# ---------------------------------------------------------------------------
# Allowlist — only free/open providers
# ---------------------------------------------------------------------------

SUPPORTED_PROVIDERS: frozenset[str] = frozenset(
    {
        "mock",
        "ollama",
        "llamacpp",
        "groq",
        "together",
        "cloudflare",
        "openrouter",
        "huggingface",
    }
)

# Providers explicitly blocked with a helpful error message
_BLOCKED_PROVIDERS: dict[str, str] = {
    "anthropic": (
        "The Anthropic provider is not supported. "
        "This project uses free/open models only. "
        "Use one of: " + ", ".join(sorted(SUPPORTED_PROVIDERS))
    ),
    "openai": (
        "The OpenAI provider is not supported. "
        "This project uses free/open models only. "
        "Use one of: " + ", ".join(sorted(SUPPORTED_PROVIDERS))
    ),
}


def list_supported_providers() -> list[str]:
    """Return the sorted list of supported (free/open) provider names."""
    return sorted(SUPPORTED_PROVIDERS)


def get_backend(spec: str) -> "LLMBackend":
    """Parse a provider spec string and return the matching backend instance.

    Parameters
    ----------
    spec:
        A provider spec string such as ``"ollama:llama3.1:8b"`` or
        ``"mock"`` or ``"groq:llama-3.1-8b-instant"``.

    Returns
    -------
    LLMBackend
        A configured backend instance ready to call.

    Raises
    ------
    ValueError
        When the provider is unknown or explicitly blocked (Anthropic/OpenAI).
    """
    parts = spec.split(":", 1)
    provider = parts[0].strip().lower()
    model = parts[1].strip() if len(parts) > 1 else ""

    # Check blocked providers first (gives a helpful error message)
    if provider in _BLOCKED_PROVIDERS:
        raise ValueError(_BLOCKED_PROVIDERS[provider])

    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported free providers: {', '.join(sorted(SUPPORTED_PROVIDERS))}"
        )

    if provider == "mock":
        from src.llm.mock_backend import MockBackend
        return MockBackend()

    if provider == "ollama":
        if not model:
            raise ValueError("Ollama provider requires a model name: 'ollama:<model>'")
        from src.llm.ollama_backend import OllamaBackend
        return OllamaBackend(model=model)

    if provider == "llamacpp":
        if not model:
            raise ValueError(
                "llamacpp provider requires a model path: 'llamacpp:<path-to-gguf>'"
            )
        from src.llm.llamacpp_backend import LlamaCppBackend
        return LlamaCppBackend(model_path=model)

    if provider == "groq":
        if not model:
            raise ValueError("Groq provider requires a model ID: 'groq:<model-id>'")
        from src.llm.groq_backend import GroqBackend
        return GroqBackend(model=model)

    if provider == "together":
        if not model:
            raise ValueError(
                "Together provider requires a model ID: 'together:<model-id>'"
            )
        from src.llm.together_backend import TogetherBackend
        return TogetherBackend(model=model)

    if provider == "cloudflare":
        if not model:
            raise ValueError(
                "Cloudflare provider requires a model ID: 'cloudflare:<model-id>'"
            )
        from src.llm.cloudflare_backend import CloudflareBackend
        return CloudflareBackend(model=model)

    if provider == "openrouter":
        if not model:
            raise ValueError(
                "OpenRouter provider requires a model ID: 'openrouter:<model-id>'"
            )
        from src.llm.openrouter_backend import OpenRouterBackend
        return OpenRouterBackend(model=model)

    if provider == "huggingface":
        if not model:
            from src.llm.huggingface_backend import HuggingFaceBackend
            return HuggingFaceBackend()
        from src.llm.huggingface_backend import HuggingFaceBackend
        return HuggingFaceBackend(model_id=model)

    # Should never reach here given the allowlist check above
    raise ValueError(f"Unhandled provider: '{provider}'")  # pragma: no cover
