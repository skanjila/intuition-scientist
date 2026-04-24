# Open-Source Models Guide

This platform is designed to run entirely on **free, open-source models** with no API key required. It uses [Ollama](https://ollama.com) for local inference and gracefully falls back to the built-in mock backend.

---

## Model Registry

The `src/llm/model_registry.py` module provides:

- `MODEL_CATALOG` — dictionary of all supported model specs
- `USE_CASE_MODEL_RECOMMENDATIONS` — per-use-case model profiles
- `FREE_FALLBACK_CHAIN` — ordered list of models to try (last: `mock`)
- `get_backend_with_fallback(use_case, profile)` — returns best available backend
- `list_free_models()` — enumerate all catalog entries
- `setup_instructions(use_case, profile)` — get install instructions

---

## Model Profiles

| Profile | Description | Use Cases |
|---------|-------------|-----------|
| `fast` | Smaller, faster models (7B) | Triage, simple Q&A |
| `balanced` | Mid-size models (8–13B) | Most use cases |
| `quality` | Large models (70B+) | Clinical, legal, finance |

---

## Recommended Models by Use Case

| Use Case | Fast | Balanced | Quality |
|----------|------|----------|---------|
| Customer Support | llama3.1:8b | llama3.1:8b | llama3.1:70b |
| Incident Response | llama3.1:8b | mistral:7b | llama3.1:70b |
| Code Review | deepseek-coder:6.7b | deepseek-coder:33b | deepseek-coder:33b |
| Clinical Decision | llama3.1:8b | llama3.1:70b | llama3.1:70b |
| Drug Interaction | llama3.1:8b | llama3.1:70b | llama3.1:70b |
| Finance/Stock | llama3.1:8b | mixtral:8x7b | llama3.1:70b |

---

## Installation

### 1. Install Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start the service
ollama serve
```

### 2. Pull a Model

```bash
# Fast (4GB RAM)
ollama pull llama3.1:8b

# Balanced (8GB RAM)
ollama pull mistral:7b

# Quality (40GB RAM)
ollama pull llama3.1:70b

# Code-specialized
ollama pull deepseek-coder:6.7b
```

### 3. Use with the Platform

```bash
# CLI
python main.py --model ollama:llama3.1:8b triage --ticket "API is down"

# Python API
from src.orchestrator.business_orchestrator import BusinessOrchestrator
orch = BusinessOrchestrator(backend_spec="ollama:llama3.1:8b")
```

---

## Backend Hierarchy

```
1. Explicit backend_spec (e.g., "ollama:llama3.1:8b")
2. Use-case recommendation for profile (tries Ollama)
3. FREE_FALLBACK_CHAIN (tries each in order)
4. MockBackend (always available, returns deterministic responses)
```

---

## Adding Custom Models

Add entries to `MODEL_CATALOG` in `src/llm/model_registry.py`:

```python
"ollama:my-custom-model": ModelSpec(
    provider="ollama",
    model_id="my-custom-model",
    profile=ModelProfile.BALANCED,
    cost_tier=CostTier.FREE,
    context_window_k=32,
    supports_tools=False,
)
```

---

## Mock Backend

The `MockBackend` is always available and returns deterministic responses for testing. It simulates confidence scores and reasoning without any external dependencies.

```python
from src.llm.mock_backend import MockBackend
from src.orchestrator.business_orchestrator import BusinessOrchestrator

orch = BusinessOrchestrator(backend=MockBackend())
result = orch.triage("test ticket")
```

---

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OpenTelemetry collector | `http://localhost:4318` |

---

## Performance Benchmarks (approximate)

| Model | Tokens/sec | RAM | Quality |
|-------|-----------|-----|---------|
| llama3.1:8b | 40-60 | 6GB | Good |
| mistral:7b | 45-65 | 6GB | Good |
| mixtral:8x7b | 20-30 | 26GB | Excellent |
| llama3.1:70b | 5-10 | 40GB | Excellent |
| deepseek-coder:33b | 10-15 | 20GB | Excellent (code) |
