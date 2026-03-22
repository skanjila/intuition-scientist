# Running the Test Suite

All tests run fully **offline** — no LLM, no network connection, and no API keys are required. The mock backend is used automatically.

---

## Table of contents

1. [Prerequisites](#prerequisites)
2. [Running the full suite](#running-the-full-suite)
3. [Filtering and targeting tests](#filtering-and-targeting-tests)
4. [Running with coverage](#running-with-coverage)
5. [Running specific test groups](#running-specific-test-groups)
6. [Model sweep tests](#model-sweep-tests)
7. [Test file reference](#test-file-reference)

---

## Prerequisites

```bash
# All test dependencies are in requirements.txt
pip install -r requirements.txt

# Optional extras
pip install pytest-cov      # for coverage reports
pip install pytest-xdist    # for parallel execution (-n auto)
```

No API keys are needed. The mock backend responds instantly and is 100 % deterministic.

---

## Running the full suite

```bash
python -m pytest tests/ -v
```

Expected result: **all tests pass** (sweep tests are skipped unless explicitly enabled — see [Model sweep tests](#model-sweep-tests)).

---

## Filtering and targeting tests

```bash
# Run a single test file
python -m pytest tests/test_orchestrator.py -v

# Run a single test class
python -m pytest tests/test_orchestrator.py::TestOrchestatorRunOffline -v

# Run tests whose name contains a substring
python -m pytest tests/ -k "timeout" -v
python -m pytest tests/ -k "escalation" -v

# Stop immediately on the first failure
python -m pytest tests/ -x -v

# Run in parallel (requires pytest-xdist)
pip install pytest-xdist
python -m pytest tests/ -n auto -v
```

---

## Running with coverage

```bash
# Terminal summary with missing lines
python -m pytest tests/ --cov=src --cov-report=term-missing

# HTML report (opens in browser)
python -m pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html   # macOS
xdg-open htmlcov/index.html  # Linux
```

---

## Running specific test groups

### Core orchestrator

```bash
python -m pytest tests/test_orchestrator.py -v
```

Covers: orchestrator initialisation, domain selection, agent count, timeout handling, and the basic CLI smoke test.

### Human involvement policy (new)

```bash
python -m pytest tests/test_human_policy.py -v
```

Covers:
- Domain selection returns > 0 domains for any question
- Orchestrator queries the correct number of agents
- Escalation triggers for high-stakes domains, low confidence, and high disagreement
- `--non-interactive` / `--auto-intuition` / `--human-policy never` never read stdin
- `--verbose` and `--quiet` flags complete without error
- New flags appear in `--help` output

### Agent feature tests

```bash
python -m pytest tests/test_agents.py tests/test_new_agents.py -v
```

Covers all 24 domain agents including the Signal Processing iterative tutor and the Experiment Runner.

### Weighing system

```bash
python -m pytest tests/test_weighing_system.py -v
```

Covers `WeighingSystem.weigh()`, alignment scoring, synthesis, and intuition accuracy calculation.

### Auto-intuition and adaptive loop

```bash
python -m pytest tests/test_auto_intuition_adaptive.py -v
```

Covers the `--auto-intuition` and `--adaptive-agents` code paths.

### Workflow rendering

```bash
python -m pytest tests/test_workflow.py -v
```

Covers `build_workflow_trace()` and `render_workflow()` for all four modes: `off`, `compact`, `standard`, `deep`.

### Large domain × mode matrix

```bash
python -m pytest tests/test_large_suite.py -v
```

Runs 200+ tests across all domains × modes (run / debate / evaluate_models). These tests pass fully offline.

### Experiment Runner agent

```bash
python -m pytest tests/test_experiment_runner.py -v
```

Covers question classification (experimentable vs. non-experimentable), experiment type selection, and plan generation.

### LLM registry

```bash
python -m pytest tests/test_llm_registry.py -v
```

Covers provider registration, blocked-provider detection (Anthropic, OpenAI), and backend instantiation.

### Performance / adaptive features

```bash
python -m pytest tests/test_performance_features.py -v
```

Covers per-agent timeout enforcement, adaptive loop expansion, and latency-budget stopping criteria.

---

## Model sweep tests

Model sweep tests cycle through real model backends and compare results across providers. They are **disabled by default** to keep CI fast and free.

### Enable with mock backend only (no model needed)

```bash
RUN_MODEL_SWEEP=1 python -m pytest tests/test_model_sweep.py -v
```

### Enable with local Ollama models

```bash
# Ollama must be running and models pulled
ollama pull llama3.1:8b
ollama pull qwen2.5:7b

RUN_MODEL_SWEEP=1 \
INTUITION_SCIENTIST_MODELS="ollama:llama3.1:8b,ollama:qwen2.5:7b" \
python -m pytest tests/test_model_sweep.py -v
```

### Enable with hosted free-tier backends

```bash
RUN_MODEL_SWEEP=1 \
INTUITION_SCIENTIST_MODELS="ollama:llama3.1:8b,groq:llama-3.1-8b-instant,together:meta-llama/Llama-3.1-8B-Instruct-Turbo" \
GROQ_API_KEY=your_key \
TOGETHER_API_KEY=your_key \
python -m pytest tests/test_model_sweep.py tests/test_large_suite.py -v
```

Sweep tests are parametrised: each `model × domain × mode` combination runs as a separate test case. Backends that are unavailable (server down, missing key, etc.) are skipped gracefully and never fail the suite.

---

## Test file reference

| File | What it covers |
|---|---|
| `test_agents.py` | All 24 domain agents, answer structure, confidence range |
| `test_auto_intuition_adaptive.py` | `--auto-intuition`, `--adaptive-agents`, adaptive loop stopping criteria |
| `test_experiment_runner.py` | Question classification, experiment type selection, plan generation |
| `test_human_policy.py` | Human involvement policy, escalation triggers, CLI flags (`--interactive`, `--non-interactive`, `--human-policy`, `--verbose`, `--quiet`) |
| `test_intuition.py` | `IntuitionCapture`, `generate_auto_intuition`, domain keyword inference |
| `test_large_suite.py` | 200+ tests: all domains × all modes |
| `test_llm_registry.py` | Provider registry, blocked providers, backend instantiation |
| `test_model_sweep.py` | Opt-in real-backend sweep (requires `RUN_MODEL_SWEEP=1`) |
| `test_models.py` | Dataclass validation (`HumanIntuition`, `AgentResponse`, `WeighingResult`, etc.) |
| `test_new_agents.py` | Signal Processing iterative tutor, Experiment Runner |
| `test_orchestrator.py` | Orchestrator init, domain selection, timeout, CLI smoke test |
| `test_performance_features.py` | Per-agent timeout, adaptive expansion, latency budget |
| `test_weighing_system.py` | `WeighingSystem.weigh()`, alignment, synthesis, accuracy scoring |
| `test_workflow.py` | Workflow trace building and rendering (all four modes) |

---

## Environment variables used by tests

| Variable | Effect |
|---|---|
| `RUN_MODEL_SWEEP` | Set to `1` to run model sweep tests (default: skipped) |
| `INTUITION_SCIENTIST_MODELS` | Comma-separated model spec list for sweep tests (default: `mock`) |
| `GROQ_API_KEY` | Groq API key — required for `groq:*` specs in sweep tests |
| `TOGETHER_API_KEY` | Together AI key — required for `together:*` specs |
| `CF_ACCOUNT_ID` / `CF_API_TOKEN` | Cloudflare credentials — required for `cloudflare:*` specs |
| `OPENROUTER_API_KEY` | OpenRouter key — required for `openrouter:*` specs |

---

## Further reading

- 🏗️ [README.md](README.md) — system architecture and design
- 📋 [SCENARIOS.md](SCENARIOS.md) — how to run scenarios with expected output
- 🤖 [docs/AGENT_WORKFLOWS.md](docs/AGENT_WORKFLOWS.md) — per-agent workflow optimization guide
- 🎛️ [docs/ORCHESTRATOR_WORKFLOWS.md](docs/ORCHESTRATOR_WORKFLOWS.md) — orchestrator configuration and composition patterns
