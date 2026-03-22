"""Tests for the LLM backend registry (offline — no model server required)."""

from __future__ import annotations

import pytest

from src.llm.mock_backend import MockBackend
from src.llm.registry import (
    SUPPORTED_PROVIDERS,
    get_backend,
    list_supported_providers,
    _BLOCKED_PROVIDERS,
)


class TestListSupportedProviders:
    def test_returns_sorted_list(self):
        providers = list_supported_providers()
        assert providers == sorted(providers)

    def test_contains_expected_providers(self):
        providers = set(list_supported_providers())
        assert {"mock", "ollama", "llamacpp", "groq", "together", "cloudflare", "openrouter"} == providers


class TestGetBackendMock:
    def test_mock_spec(self):
        backend = get_backend("mock")
        assert isinstance(backend, MockBackend)

    def test_mock_generate_returns_string(self):
        backend = get_backend("mock")
        result = backend.generate("system prompt", "user message")
        assert isinstance(result, str)
        assert len(result) > 0


class TestGetBackendBlocked:
    def test_anthropic_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            get_backend("anthropic")

    def test_anthropic_with_model_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            get_backend("anthropic:claude-3-haiku-20240307")

    def test_openai_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            get_backend("openai")

    def test_openai_with_model_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            get_backend("openai:gpt-4o")


class TestGetBackendUnknown:
    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_backend("fakeprovider:somemodel")


class TestGetBackendMissingModel:
    """Specs that require a model name but don't provide one should raise."""

    @pytest.mark.parametrize(
        "spec",
        ["ollama", "llamacpp", "groq", "together", "cloudflare", "openrouter"],
    )
    def test_provider_without_model_raises(self, spec: str):
        with pytest.raises(ValueError, match="requires"):
            get_backend(spec)


class TestGetBackendLazy:
    """Backends with model names should instantiate without contacting servers."""

    def test_ollama_instantiates(self):
        from src.llm.ollama_backend import OllamaBackend
        backend = get_backend("ollama:llama3.1:8b")
        assert isinstance(backend, OllamaBackend)
        assert backend.model == "llama3.1:8b"

    def test_llamacpp_instantiates(self):
        from src.llm.llamacpp_backend import LlamaCppBackend
        backend = get_backend("llamacpp:models/test.gguf")
        assert isinstance(backend, LlamaCppBackend)
        assert backend.model_path == "models/test.gguf"

    def test_groq_instantiates(self):
        from src.llm.groq_backend import GroqBackend
        backend = get_backend("groq:llama-3.1-8b-instant")
        assert isinstance(backend, GroqBackend)
        assert backend.model == "llama-3.1-8b-instant"

    def test_together_instantiates(self):
        from src.llm.together_backend import TogetherBackend
        backend = get_backend("together:meta-llama/Llama-3.1-8B-Instruct-Turbo")
        assert isinstance(backend, TogetherBackend)
        assert backend.model == "meta-llama/Llama-3.1-8B-Instruct-Turbo"

    def test_cloudflare_instantiates(self):
        from src.llm.cloudflare_backend import CloudflareBackend
        backend = get_backend("cloudflare:@cf/meta/llama-3.1-8b-instruct")
        assert isinstance(backend, CloudflareBackend)
        assert backend.model == "@cf/meta/llama-3.1-8b-instruct"

    def test_openrouter_instantiates(self):
        from src.llm.openrouter_backend import OpenRouterBackend
        backend = get_backend("openrouter:meta-llama/llama-3.1-8b-instruct:free")
        assert isinstance(backend, OpenRouterBackend)
        assert backend.model == "meta-llama/llama-3.1-8b-instruct:free"
