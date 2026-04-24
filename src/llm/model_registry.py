from __future__ import annotations

"""Open-source model registry for the Business Agent Platform.

All models listed here are free to use. Zero proprietary APIs required.

Cost tiers
----------
free-local  : runs entirely on-device via Ollama or llama.cpp. Zero API cost.
              Requires 8–64 GB RAM depending on model size.
free-api    : hosted inference via Groq, Together AI, Cloudflare, or OpenRouter
              free tiers. Subject to rate limits but no billing required.

Model profiles
--------------
fast        : ≤8B parameters. Sub-second latency on GPU, ~2s on CPU.
              Best for high-volume, latency-sensitive workloads.
balanced    : 8B–14B parameters. 1–5s latency. Good quality/speed tradeoff.
              Recommended default for most use cases.
quality     : 30B–70B parameters. 5–30s latency on GPU.
              Best for complex reasoning, medical, legal, and long-form tasks.

Usage example
-------------
    from src.llm.model_registry import get_backend_with_fallback, setup_instructions

    # Auto-select best available free model for code review
    backend = get_backend_with_fallback("code_review", profile="balanced")
    response = backend.generate("You are a code reviewer.", "Review this diff: ...")

    # Print setup instructions
    print(setup_instructions("clinical_decision_support", "quality"))
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm.base import LLMBackend


class ModelProfile(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"


class CostTier(str, Enum):
    FREE_LOCAL = "free-local"
    FREE_API = "free-api"


@dataclass
class ModelSpec:
    provider: str
    model_id: str
    profile: ModelProfile
    cost_tier: CostTier
    context_window_k: int
    strengths: list[str] = field(default_factory=list)
    setup_command: str = ""
    notes: str = ""


MODEL_CATALOG: dict[str, ModelSpec] = {
    "ollama:llama3.1:8b": ModelSpec(
        provider="ollama", model_id="llama3.1:8b",
        profile=ModelProfile.FAST, cost_tier=CostTier.FREE_LOCAL,
        context_window_k=128,
        strengths=["general reasoning", "instruction following", "fast inference"],
        setup_command="ollama pull llama3.1:8b",
    ),
    "ollama:llama3.1:70b": ModelSpec(
        provider="ollama", model_id="llama3.1:70b",
        profile=ModelProfile.QUALITY, cost_tier=CostTier.FREE_LOCAL,
        context_window_k=128,
        strengths=["complex reasoning", "long-form writing", "medical analysis"],
        setup_command="ollama pull llama3.1:70b",
    ),
    "ollama:llama3.2:3b": ModelSpec(
        provider="ollama", model_id="llama3.2:3b",
        profile=ModelProfile.FAST, cost_tier=CostTier.FREE_LOCAL,
        context_window_k=128,
        strengths=["ultra-fast", "low memory", "edge deployment"],
        setup_command="ollama pull llama3.2:3b",
    ),
    "ollama:deepseek-r1:7b": ModelSpec(
        provider="ollama", model_id="deepseek-r1:7b",
        profile=ModelProfile.BALANCED, cost_tier=CostTier.FREE_LOCAL,
        context_window_k=64,
        strengths=["chain-of-thought reasoning", "mathematics", "code analysis"],
        setup_command="ollama pull deepseek-r1:7b",
    ),
    "ollama:deepseek-r1:32b": ModelSpec(
        provider="ollama", model_id="deepseek-r1:32b",
        profile=ModelProfile.QUALITY, cost_tier=CostTier.FREE_LOCAL,
        context_window_k=64,
        strengths=["deep reasoning", "complex analysis", "research"],
        setup_command="ollama pull deepseek-r1:32b",
    ),
    "ollama:qwen2.5-coder:7b": ModelSpec(
        provider="ollama", model_id="qwen2.5-coder:7b",
        profile=ModelProfile.BALANCED, cost_tier=CostTier.FREE_LOCAL,
        context_window_k=128,
        strengths=["code review", "bug detection", "code generation"],
        setup_command="ollama pull qwen2.5-coder:7b",
    ),
    "ollama:qwen2.5-coder:14b": ModelSpec(
        provider="ollama", model_id="qwen2.5-coder:14b",
        profile=ModelProfile.QUALITY, cost_tier=CostTier.FREE_LOCAL,
        context_window_k=128,
        strengths=["advanced code review", "security analysis", "refactoring"],
        setup_command="ollama pull qwen2.5-coder:14b",
    ),
    "ollama:medllama2:7b": ModelSpec(
        provider="ollama", model_id="medllama2:7b",
        profile=ModelProfile.BALANCED, cost_tier=CostTier.FREE_LOCAL,
        context_window_k=4,
        strengths=["medical terminology", "clinical reasoning", "drug knowledge"],
        setup_command="ollama pull medllama2:7b",
    ),
    "ollama:meditron:7b": ModelSpec(
        provider="ollama", model_id="meditron:7b",
        profile=ModelProfile.BALANCED, cost_tier=CostTier.FREE_LOCAL,
        context_window_k=4,
        strengths=["clinical guidelines", "evidence-based medicine", "medical Q&A"],
        setup_command="ollama pull meditron:7b",
    ),
    "ollama:mistral:7b": ModelSpec(
        provider="ollama", model_id="mistral:7b",
        profile=ModelProfile.FAST, cost_tier=CostTier.FREE_LOCAL,
        context_window_k=32,
        strengths=["instruction following", "European compliance", "multilingual"],
        setup_command="ollama pull mistral:7b",
    ),
    "ollama:mixtral:8x7b": ModelSpec(
        provider="ollama", model_id="mixtral:8x7b",
        profile=ModelProfile.QUALITY, cost_tier=CostTier.FREE_LOCAL,
        context_window_k=32,
        strengths=["mixture-of-experts", "diverse tasks", "long context"],
        setup_command="ollama pull mixtral:8x7b",
    ),
    "ollama:phi3:mini": ModelSpec(
        provider="ollama", model_id="phi3:mini",
        profile=ModelProfile.FAST, cost_tier=CostTier.FREE_LOCAL,
        context_window_k=128,
        strengths=["extremely fast", "low memory 3.8B", "surprising capability"],
        setup_command="ollama pull phi3:mini",
    ),
    "ollama:gemma2:9b": ModelSpec(
        provider="ollama", model_id="gemma2:9b",
        profile=ModelProfile.BALANCED, cost_tier=CostTier.FREE_LOCAL,
        context_window_k=8,
        strengths=["Google-tuned", "factual accuracy", "safety-tuned"],
        setup_command="ollama pull gemma2:9b",
    ),
    "groq:llama-3.1-8b-instant": ModelSpec(
        provider="groq", model_id="llama-3.1-8b-instant",
        profile=ModelProfile.FAST, cost_tier=CostTier.FREE_API,
        context_window_k=128,
        strengths=["sub-100ms latency", "free tier 14k TPM", "best for triage"],
        setup_command="export GROQ_API_KEY=gsk_...",
    ),
    "groq:llama-3.3-70b-versatile": ModelSpec(
        provider="groq", model_id="llama-3.3-70b-versatile",
        profile=ModelProfile.QUALITY, cost_tier=CostTier.FREE_API,
        context_window_k=128,
        strengths=["state-of-art quality", "free tier 6k TPM", "complex reasoning"],
        setup_command="export GROQ_API_KEY=gsk_...",
    ),
    "groq:mixtral-8x7b-32768": ModelSpec(
        provider="groq", model_id="mixtral-8x7b-32768",
        profile=ModelProfile.QUALITY, cost_tier=CostTier.FREE_API,
        context_window_k=32,
        strengths=["excellent reasoning", "32k context", "free tier"],
        setup_command="export GROQ_API_KEY=gsk_...",
    ),
    "groq:gemma2-9b-it": ModelSpec(
        provider="groq", model_id="gemma2-9b-it",
        profile=ModelProfile.BALANCED, cost_tier=CostTier.FREE_API,
        context_window_k=8,
        strengths=["fast", "safe", "Google quality"],
        setup_command="export GROQ_API_KEY=gsk_...",
    ),
    "together:meta-llama/Llama-3.1-8B-Instruct-Turbo": ModelSpec(
        provider="together", model_id="meta-llama/Llama-3.1-8B-Instruct-Turbo",
        profile=ModelProfile.FAST, cost_tier=CostTier.FREE_API,
        context_window_k=128,
        strengths=["fast inference", "free $25 credit", "good instruction following"],
        setup_command="export TOGETHER_API_KEY=...",
    ),
    "cloudflare:@cf/meta/llama-3.1-8b-instruct": ModelSpec(
        provider="cloudflare", model_id="@cf/meta/llama-3.1-8b-instruct",
        profile=ModelProfile.FAST, cost_tier=CostTier.FREE_API,
        context_window_k=128,
        strengths=["edge inference", "10k requests/day free", "global CDN"],
        setup_command="export CF_API_TOKEN=... CF_ACCOUNT_ID=...",
    ),
    "openrouter:meta-llama/llama-3.1-8b-instruct:free": ModelSpec(
        provider="openrouter", model_id="meta-llama/llama-3.1-8b-instruct:free",
        profile=ModelProfile.FAST, cost_tier=CostTier.FREE_API,
        context_window_k=128,
        strengths=["free tier", "no key required for some models", "fallback option"],
        setup_command="export OPENROUTER_API_KEY=...",
    ),
}

USE_CASE_MODEL_RECOMMENDATIONS: dict[str, dict[str, str]] = {
    "customer_support": {
        "fast": "groq:llama-3.1-8b-instant",
        "balanced": "ollama:llama3.1:8b",
        "quality": "groq:llama-3.3-70b-versatile",
    },
    "incident_response": {
        "fast": "groq:llama-3.1-8b-instant",
        "balanced": "ollama:llama3.1:8b",
        "quality": "groq:mixtral-8x7b-32768",
    },
    "finance_reconciliation": {
        "fast": "groq:gemma2-9b-it",
        "balanced": "ollama:deepseek-r1:7b",
        "quality": "groq:llama-3.3-70b-versatile",
    },
    "code_review": {
        "fast": "groq:llama-3.1-8b-instant",
        "balanced": "ollama:qwen2.5-coder:7b",
        "quality": "ollama:qwen2.5-coder:14b",
    },
    "analytics": {
        "fast": "groq:llama-3.1-8b-instant",
        "balanced": "ollama:deepseek-r1:7b",
        "quality": "groq:llama-3.3-70b-versatile",
    },
    "rfp_response": {
        "fast": "groq:mixtral-8x7b-32768",
        "balanced": "ollama:mixtral:8x7b",
        "quality": "groq:llama-3.3-70b-versatile",
    },
    "compliance_qa": {
        "fast": "groq:mixtral-8x7b-32768",
        "balanced": "ollama:llama3.1:8b",
        "quality": "groq:llama-3.3-70b-versatile",
    },
    "sales_outreach": {
        "fast": "groq:llama-3.1-8b-instant",
        "balanced": "ollama:llama3.1:8b",
        "quality": "groq:llama-3.3-70b-versatile",
    },
    "supply_chain_exception": {
        "fast": "groq:llama-3.1-8b-instant",
        "balanced": "ollama:llama3.1:8b",
        "quality": "groq:mixtral-8x7b-32768",
    },
    "general": {
        "fast": "groq:llama-3.1-8b-instant",
        "balanced": "ollama:llama3.1:8b",
        "quality": "groq:llama-3.3-70b-versatile",
    },
    "clinical_decision_support": {
        "fast": "ollama:meditron:7b",
        "balanced": "ollama:meditron:7b",
        "quality": "groq:llama-3.3-70b-versatile",
    },
    "drug_interaction": {
        "fast": "ollama:medllama2:7b",
        "balanced": "ollama:medllama2:7b",
        "quality": "groq:mixtral-8x7b-32768",
    },
    "medical_literature": {
        "fast": "groq:mixtral-8x7b-32768",
        "balanced": "ollama:llama3.1:8b",
        "quality": "groq:llama-3.3-70b-versatile",
    },
    "patient_risk": {
        "fast": "groq:llama-3.1-8b-instant",
        "balanced": "ollama:meditron:7b",
        "quality": "groq:llama-3.3-70b-versatile",
    },
    "healthcare_access": {
        "fast": "groq:llama-3.1-8b-instant",
        "balanced": "ollama:llama3.1:8b",
        "quality": "groq:llama-3.3-70b-versatile",
    },
    "genomics_medicine": {
        "fast": "groq:mixtral-8x7b-32768",
        "balanced": "ollama:llama3.1:8b",
        "quality": "groq:llama-3.3-70b-versatile",
    },
    "mental_health_triage": {
        "fast": "groq:llama-3.1-8b-instant",
        "balanced": "ollama:llama3.1:8b",
        "quality": "groq:llama-3.3-70b-versatile",
    },
    "clinical_trials": {
        "fast": "groq:llama-3.1-8b-instant",
        "balanced": "ollama:llama3.1:8b",
        "quality": "groq:mixtral-8x7b-32768",
    },
}

FREE_FALLBACK_CHAIN: list[str] = [
    "ollama:llama3.1:8b",
    "groq:llama-3.1-8b-instant",
    "together:meta-llama/Llama-3.1-8B-Instruct-Turbo",
    "cloudflare:@cf/meta/llama-3.1-8b-instruct",
    "openrouter:meta-llama/llama-3.1-8b-instruct:free",
    "mock",
]


def get_model_for_use_case(use_case: str, profile: str = "balanced") -> str:
    """Return the recommended model spec string for a use case and profile.

    Falls back to balanced if quality unavailable, then fast.
    Falls back to general if use_case not in recommendations.
    """
    recommendations = USE_CASE_MODEL_RECOMMENDATIONS.get(
        use_case, USE_CASE_MODEL_RECOMMENDATIONS["general"]
    )
    fallback_order = [profile, "balanced", "fast", "quality"]
    for p in fallback_order:
        if p in recommendations:
            return recommendations[p]
    # Should never reach here since all entries have fast/balanced/quality
    return USE_CASE_MODEL_RECOMMENDATIONS["general"]["balanced"]


def get_backend_with_fallback(use_case: str, profile: str = "balanced") -> "LLMBackend":
    """Try each model in the fallback chain, return first that instantiates successfully.

    Always falls back to MockBackend as last resort.
    Catches ImportError and RuntimeError from backends that need missing deps.

    This is the key function for open-source-first operation — call this to
    get a working backend without needing any API keys.
    """
    from src.llm.registry import get_backend
    from src.llm.mock_backend import MockBackend

    # Try the recommended model first
    recommended = get_model_for_use_case(use_case, profile)
    try:
        return get_backend(recommended)
    except (ImportError, RuntimeError, ValueError, Exception):
        pass

    # Try the full fallback chain
    for spec in FREE_FALLBACK_CHAIN:
        if spec == "mock":
            return MockBackend()
        try:
            return get_backend(spec)
        except (ImportError, RuntimeError, ValueError, Exception):
            continue

    return MockBackend()


def list_free_models() -> list[ModelSpec]:
    """Return all models sorted by profile then cost_tier."""
    profile_order = {ModelProfile.FAST: 0, ModelProfile.BALANCED: 1, ModelProfile.QUALITY: 2}
    tier_order = {CostTier.FREE_LOCAL: 0, CostTier.FREE_API: 1}
    return sorted(
        MODEL_CATALOG.values(),
        key=lambda m: (profile_order[m.profile], tier_order[m.cost_tier])
    )


def setup_instructions(use_case: str = "", profile: str = "balanced") -> str:
    """Return human-readable setup instructions for the recommended model."""
    if use_case:
        model_key = get_model_for_use_case(use_case, profile)
    else:
        model_key = get_model_for_use_case("general", profile)

    spec = MODEL_CATALOG.get(model_key)
    if not spec:
        return f"Model '{model_key}' not found in catalog. Use MockBackend for offline testing."

    lines = [
        f"Recommended model for '{use_case or 'general'}' ({profile} profile):",
        f"  Model: {model_key}",
        f"  Provider: {spec.provider}",
        f"  Cost tier: {spec.cost_tier.value}",
        f"  Context window: {spec.context_window_k}k tokens",
        f"  Strengths: {', '.join(spec.strengths)}",
        f"  Setup: {spec.setup_command}",
    ]
    if spec.notes:
        lines.append(f"  Notes: {spec.notes}")
    return "\n".join(lines)
