# Orchestrator Workflow Optimization Guide

Detailed instructions for building the most effective workflows with the `AgentOrchestrator` — covering initialization, domain selection, thread-pool tuning, adaptive loops, timeout strategy, human policy integration, entry-point selection, and composing the orchestrator into larger systems.

---

## Table of contents

1. [Orchestrator initialization reference](#orchestrator-initialization-reference)
2. [Choosing an entry point](#choosing-an-entry-point)
3. [Domain selection strategies](#domain-selection-strategies)
4. [Thread-pool and concurrency tuning](#thread-pool-and-concurrency-tuning)
5. [Per-agent timeout strategy](#per-agent-timeout-strategy)
6. [Adaptive agent loop configuration](#adaptive-agent-loop-configuration)
7. [Human involvement policy integration](#human-involvement-policy-integration)
8. [Progress and observability](#progress-and-observability)
9. [MCP configuration](#mcp-configuration)
10. [Token budget optimization](#token-budget-optimization)
11. [Composing the orchestrator into larger systems](#composing-the-orchestrator-into-larger-systems)
12. [Batch and pipeline workflows](#batch-and-pipeline-workflows)
13. [Model evaluation workflow](#model-evaluation-workflow)
14. [Debate and multi-party workflows](#debate-and-multi-party-workflows)
15. [Interview prep workflow](#interview-prep-workflow)
16. [Complete workflow recipes](#complete-workflow-recipes)

---

## Orchestrator initialization reference

```python
from src.orchestrator.agent_orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator(
    # ── Backend ────────────────────────────────────────────────────────
    backend_spec="mock",           # str: "mock" | "ollama:<model>" | "groq:<model>" | …
    backend=None,                  # Optional[LLMBackend]: pass a pre-built backend instance

    # ── MCP ────────────────────────────────────────────────────────────
    use_mcp=True,                  # bool: enable DuckDuckGo web search for all agents

    # ── Concurrency ────────────────────────────────────────────────────
    max_workers=7,                 # int: thread-pool size for parallel agent calls
    agent_timeout_seconds=30.0,    # float: per-agent deadline before placeholder response

    # ── Domain selection ───────────────────────────────────────────────
    max_domains=None,              # Optional[int]: hard cap on number of agents

    # ── Token budgets ──────────────────────────────────────────────────
    agent_max_tokens=1024,         # int: token budget per agent LLM call
    synthesis_max_tokens=512,      # int: token budget for synthesis + analysis calls

    # ── Human intuition mode ───────────────────────────────────────────
    auto_intuition=False,          # bool: True → skip interactive prompt; auto-generate

    # ── Adaptive loop ──────────────────────────────────────────────────
    adaptive_agents=False,         # bool: True → expanding adaptive domain loop
    target_latency_ms=None,        # Optional[int]: wall-clock cap (ms) for adaptive loop

    # ── Observability ──────────────────────────────────────────────────
    verbose=False,                 # bool: emit DEBUG-level logging
    progress_callback=None,        # Optional[Callable[[str], None]]: per-event hook

    # ── Legacy (ignored) ───────────────────────────────────────────────
    llm_provider="mock",
    model=None,
)
```

### Minimal non-interactive setup

```python
orch = AgentOrchestrator(use_mcp=False, auto_intuition=True, max_domains=3)
result = orch.run("What is gradient descent?")
orch.close()
```

### Context manager (recommended)

```python
with AgentOrchestrator(use_mcp=False, auto_intuition=True) as orch:
    result = orch.run("What is gradient descent?")
```

The context manager calls `orch.close()` on exit, releasing the MCP HTTP client and backend connections cleanly.

---

## Choosing an entry point

The orchestrator exposes four public entry points. Choose based on your use case:

| Entry point | Returns | Use when |
|---|---|---|
| `run(question, …)` | `WeighingResult` | Standard: weighing human intuition against domain agents |
| `debate(question, …)` | `DebateResult` | You want a structured three-way debate (human vs. tool vs. agents) |
| `interview_prep(question, …)` | `InterviewResult` | FAANG coaching: technical + algorithmic + psychological |
| `evaluate_models(question, model_specs, …)` | `ModelEvaluationResult` | Comparing multiple LLM backends side-by-side |

### `run()` — standard weighing workflow

```python
from src.models import HumanIntuition

intuition = HumanIntuition(
    question="How does RLHF work?",
    intuitive_answer="It uses human feedback to fine-tune a reward model.",
    confidence=0.7,
    reasoning="Based on the InstructGPT paper.",
)

with AgentOrchestrator(use_mcp=True) as orch:
    result = orch.run(
        "How does RLHF work?",
        prefilled_intuition=intuition,   # skip the interactive prompt
        domains=[Domain.DEEP_LEARNING, Domain.NEURAL_NETWORKS],
    )

print(result.synthesized_answer)
print(f"Intuition accuracy: {result.intuition_accuracy_pct:.1f}%")
```

### `debate()` — structured debate workflow

```python
with AgentOrchestrator(use_mcp=True) as orch:
    debate = orch.debate(
        "Is microservices always the right default architecture?",
        prefilled_intuition=intuition,
        domains=[Domain.ENTERPRISE_ARCHITECTURE, Domain.STRATEGY_INTELLIGENCE],
    )

for round_ in debate.rounds:
    print(f"\n--- Round: {round_.topic} ---")
    print(f"Agreements: {round_.agreements}")
    print(f"Divergences: {round_.divergences}")
print(f"\nVerdict: {debate.verdict}")
```

### `interview_prep()` — coaching workflow

```python
with AgentOrchestrator(use_mcp=False, auto_intuition=True) as orch:
    result = orch.interview_prep(
        "How do you find the kth largest element in an unsorted array?"
    )

print(f"Technical score: {result.technical_score:.0%}")
print(f"Technical feedback: {result.technical_feedback}")
print(f"Algorithmic insight: {result.algorithmic_insight}")
print(f"Mental preparation: {result.mental_preparation}")
```

### `evaluate_models()` — model comparison workflow

```python
with AgentOrchestrator(use_mcp=False, auto_intuition=True) as orch:
    eval_result = orch.evaluate_models(
        "Explain the bias-variance tradeoff",
        model_specs=["mock", "ollama:llama3.1:8b", "groq:llama-3.1-8b-instant"],
        domains=[Domain.DEEP_LEARNING, Domain.COMPUTER_SCIENCE],
    )

print(f"Best model: {eval_result.best_model_spec}")
print(f"Mean accuracy: {eval_result.mean_intuition_accuracy_pct:.1f}%")
for r in eval_result.model_results:
    status = "✓" if r.backend_available else "✗"
    acc = f"{r.weighing_result.intuition_accuracy_pct:.1f}%" if r.weighing_result else "N/A"
    print(f"  {status} {r.model_spec}: {acc} in {r.duration_seconds:.1f}s")
```

---

## Domain selection strategies

### Strategy 1: Auto-detect (default)

The orchestrator uses `IntuitionCapture.infer_domains()` to score all domains by keyword overlap with the question + intuition text. The top-scoring domains are selected, with a minimum floor of 3.

```python
# Auto-detect: let the system choose
result = orch.run("How does quantum entanglement affect cryptography?")
# → typically selects: Physics, Computer Science, Cybersecurity, EE LLM Research
```

**When to use:** exploratory questions where you don't know which domains are most relevant.

**Tuning:** increase keyword overlap for a domain by adding more domain keywords to `_DOMAIN_KEYWORDS` in `src/intuition/human_intuition.py`.

### Strategy 2: Explicit domains

```python
result = orch.run(
    "Design a noise-cancelling algorithm for audio processing",
    domains=[Domain.SIGNAL_PROCESSING, Domain.ELECTRICAL_ENGINEERING],
)
```

**When to use:** when you know exactly which domain(s) are relevant. Explicit domains also disable the adaptive loop (even if `adaptive_agents=True`).

### Strategy 3: Max-domains cap

```python
# Auto-detect but limit to the top 3 domains
orch = AgentOrchestrator(use_mcp=False, max_domains=3)
```

**When to use:** latency-sensitive applications, CI, or local GPU with limited VRAM.

### Strategy 4: Adaptive loop

```python
orch = AgentOrchestrator(
    use_mcp=False,
    adaptive_agents=True,
    max_domains=8,
    target_latency_ms=10000,  # stop after 10 s regardless
)
result = orch.run("Explain the bias-variance tradeoff")
```

The loop starts with the top 3 domains and expands by 2 at a time until mean agent confidence ≥ 0.65 or a stopping criterion fires. See [Adaptive agent loop configuration](#adaptive-agent-loop-configuration) for details.

### Domain selection decision matrix

| Question type | Recommended strategy |
|---|---|
| Broad / exploratory | Auto-detect, no `max_domains` cap |
| Focused technical | Explicit domains, 1–3 |
| Latency-sensitive | Auto-detect + `max_domains=3` or adaptive loop with budget |
| Batch processing | Auto-detect + `max_domains=3`, `auto_intuition=True` |
| CI smoke test | Explicit 2 domains, `auto_intuition=True`, `--no-mcp` |

---

## Thread-pool and concurrency tuning

The orchestrator runs all agent LLM calls in parallel using a `ThreadPoolExecutor` with `max_workers` threads.

### Choosing `max_workers`

| Backend | Recommended `max_workers` | Rationale |
|---|---|---|
| Mock | 7–24 | Instant responses; parallelism is free |
| Ollama (local GPU) | **1** | Single GPU; concurrent calls increase queuing latency |
| llama.cpp (local CPU) | **1–2** | CPU-bound; concurrency adds thrashing |
| Groq / Together / Cloudflare | **3–7** | API concurrency limits; 3–5 usually optimal |
| OpenRouter | **3–5** | Rate-limited; check provider limits |

### Why `max_workers=1` is often faster on local GPUs

With a single GPU, the LLM processes one prompt at a time. Submitting 7 parallel requests forces the GPU to context-switch between batches (or queue them), often increasing total time by 2–5×. Sequential execution (`max_workers=1`) with a shorter per-agent token budget (`--agent-max-tokens 256`) is usually the fastest profile for local hardware.

### Apple Silicon preset

```python
orch = AgentOrchestrator(
    backend_spec="ollama:qwen2.5:7b",
    use_mcp=False,
    max_workers=1,
    max_domains=3,
    agent_max_tokens=256,
    synthesis_max_tokens=384,
    auto_intuition=True,
)
```

Equivalent CLI: `python main.py --provider ollama:qwen2.5:7b --fast`

### High-throughput API preset

```python
orch = AgentOrchestrator(
    backend_spec="groq:llama-3.1-8b-instant",
    use_mcp=False,
    max_workers=5,
    max_domains=7,
    agent_max_tokens=512,
    auto_intuition=True,
)
```

---

## Per-agent timeout strategy

The `agent_timeout_seconds` parameter sets a hard per-agent deadline. Agents that exceed it receive a low-confidence placeholder response (`confidence=0.1`, `answer="Agent timed out after Xs"`).

### How timeout interacts with `max_workers`

The timeout is applied **per future** in submission order. With `max_workers=7` and 7 agents, all 7 run in parallel and each has `agent_timeout_seconds` to complete. With `max_workers=1` and 7 agents, they run sequentially — the total wall-clock time is `7 × agent_timeout_seconds` in the worst case.

### Recommended timeout values

| Context | Recommended `agent_timeout_seconds` |
|---|---|
| CI smoke test (mock) | 5.0–10.0 |
| Local Ollama (fast questions) | 30.0–60.0 |
| Local Ollama (long derivations) | 120.0–300.0 |
| Groq / hosted API | 15.0–30.0 |
| Production with SLA | Equal to your SLA minus synthesis time (~5–10 s) |

### Timeout behaviour on timed-out agents

```python
# A timed-out agent produces this placeholder:
AgentResponse(
    domain=Domain.PHYSICS,
    answer="Agent timed out after 30.0s.",
    reasoning="",
    confidence=0.1,
)
```

The `WeighingSystem` will still include timed-out agents in its scoring, but their low confidence (0.1) gives them negligible weight. The `intuition_accuracy_pct` is not meaningfully affected.

**Best practice:** if you consistently see timeouts for a specific domain, either increase the timeout for that domain or reduce its token budget.

---

## Adaptive agent loop configuration

### When to use the adaptive loop

| Scenario | Adaptive loop? |
|---|---|
| Exploratory questions with unknown scope | ✅ Yes |
| Latency-sensitive queries with quality floor | ✅ Yes + `target_latency_ms` |
| Well-scoped questions with known domains | ❌ No — use explicit domains |
| CI / fast smoke tests | ❌ No — use `max_domains=2` |
| Batch processing large datasets | ❌ No — too unpredictable per-question latency |

### Configuration parameters

```python
orch = AgentOrchestrator(
    adaptive_agents=True,
    max_domains=10,          # absolute ceiling on total agents queried
    target_latency_ms=8000,  # stop expanding after 8 seconds
)
```

### Adaptive loop thresholds (internal constants)

| Constant | Value | Meaning |
|---|---|---|
| `_ADAPTIVE_INITIAL` | 3 | Number of agents in the first batch |
| `_ADAPTIVE_STEP` | 2 | Number of new domains added per expansion round |
| `_ADAPTIVE_CONF_THRESHOLD` | 0.65 | Mean confidence above which expansion stops |

**Override by subclassing:**
```python
from src.orchestrator.agent_orchestrator import AgentOrchestrator

class AggressiveOrchestrator(AgentOrchestrator):
    """Expands more aggressively: starts with 5, adds 3 at a time."""

    def _adaptive_select_and_run(self, question, intuition):
        # Inject custom constants before calling super
        import src.orchestrator.agent_orchestrator as mod
        original_initial = mod._ADAPTIVE_INITIAL if hasattr(mod, '_ADAPTIVE_INITIAL') else 3
        # Note: constants are local variables in the method; override by overriding the method
        # Copy the method body with modified constants, or monkey-patch
        ...
```

Since the constants are local to the method body, the cleanest override is to copy `_adaptive_select_and_run` with your values.

### Adaptive loop stopping criteria

The loop halts as soon as **any one** condition is met:

1. **Confidence met:** mean confidence across all collected responses ≥ 0.65.
2. **No candidates left:** all ranked domains have been queried.
3. **Max domains reached:** `max_domains` ceiling hit.
4. **Time budget exceeded:** `target_latency_ms` elapsed since start of first round.

### Logging the adaptive loop

The adaptive loop emits `INFO`-level logs via `logging.getLogger(__name__)`. Enable them:

```python
import logging
logging.basicConfig(level=logging.INFO)

with AgentOrchestrator(adaptive_agents=True, use_mcp=False) as orch:
    result = orch.run("Explain the bias-variance tradeoff", prefilled_intuition=intuition)
```

Expected log output:
```
INFO [adaptive] Round 1: querying 3 new agent(s): ['deep_learning', 'neural_networks', 'computer_science']
INFO [adaptive] Round 1 complete: 3 domains queried, mean_conf=0.72, elapsed=45 ms
INFO [adaptive] Stopping: mean confidence 0.72 >= threshold 0.65
```

---

## Human involvement policy integration

The orchestrator's `auto_intuition` flag controls whether it prompts interactively. The `HumanPolicy` module in `src/intuition/human_policy.py` provides higher-level policy logic that the CLI uses to set `auto_intuition` automatically.

### Using the policy programmatically

```python
from src.intuition.human_policy import HumanPolicy, decide_interactive, should_escalate
from src.intuition.human_intuition import IntuitionCapture
from src.orchestrator.agent_orchestrator import AgentOrchestrator
from src.models import Domain

question = "What drug interactions should I watch for with warfarin?"

# 1. Infer domains
domains = IntuitionCapture.infer_domains(question)
# → [Domain.HEALTHCARE, Domain.BIOTECH_GENOMICS, ...]

# 2. Decide interaction mode based on AUTO policy
interactive = decide_interactive(HumanPolicy.AUTO, domains)
# → True (healthcare is a high-stakes domain → escalate)

# 3. Build orchestrator accordingly
with AgentOrchestrator(
    use_mcp=True,
    auto_intuition=not interactive,
) as orch:
    result = orch.run(question, domains=domains)
```

### Building a post-run escalation check

For workflows that need to re-prompt the human after seeing agent responses:

```python
from src.intuition.human_policy import should_escalate

with AgentOrchestrator(use_mcp=False, auto_intuition=True) as orch:
    # Run with auto-intuition first
    result = orch.run(question, prefilled_intuition=auto_intuition_obj)

# Check if agents disagreed or had low confidence
if should_escalate(
    domains=[r.domain for r in result.agent_responses],
    responses=result.agent_responses,
    use_mcp=False,
):
    print("⚠  Low confidence or high disagreement detected.")
    print("   Consider re-running with --interactive for human input.")
```

### Escalation triggers summary

| Trigger | `should_escalate` fires when |
|---|---|
| High-stakes domain | `Domain.HEALTHCARE`, `LEGAL_COMPLIANCE`, or `FINANCE_ECONOMICS` in domain list |
| Low confidence | Mean agent confidence < 0.40 |
| High disagreement | Max − min confidence spread > 0.45 |
| MCP expected but absent | MCP enabled + tool-heavy domain + all `mcp_context` fields empty |

---

## Progress and observability

### Using `progress_callback`

The `progress_callback` parameter is a `Callable[[str], None]` that receives human-readable event messages. This decouples the orchestrator from stdout and lets you route progress to any sink (terminal, log file, web socket, etc.).

```python
events = []

def capture_progress(msg: str) -> None:
    events.append(msg)
    # or: logging.info(msg)
    # or: websocket.send(msg)
    # or: print(f"[{datetime.now():%H:%M:%S}] {msg}")

with AgentOrchestrator(
    use_mcp=False,
    auto_intuition=True,
    progress_callback=capture_progress,
) as orch:
    result = orch.run("What is gradient descent?", domains=[Domain.DEEP_LEARNING])

# Review captured events
for event in events:
    print(event)
```

**Events emitted (in order):**
1. `"🔍  MCP internet search: enabled"` or `"disabled"`
2. `"📋  Selected N domain(s): Domain A, Domain B, …"`
3. `"  ▶  [Domain Name] querying…"` — one per agent, as they start
4. `"  ✓  [Domain Name] done — conf=X%, pipeline=intuition/tool (MCP: results/none)"` — one per agent as they finish
5. `"  ⚠  [Domain Name] timed out after Xs"` — only if timeout fires

### Python `logging` integration

The orchestrator also logs via `logging.getLogger("src.orchestrator.agent_orchestrator")`. Enable at DEBUG level to see all internal events:

```python
import logging
logging.getLogger("src.orchestrator.agent_orchestrator").setLevel(logging.DEBUG)
logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
```

DEBUG events include: agent start/finish with domain name, confidence, pipeline dominance, and MCP result status.

### Timing individual agents

```python
import time

timings = {}

def timed_progress(msg: str) -> None:
    if "querying" in msg:
        domain = msg.split("[")[1].split("]")[0]
        timings[domain] = {"start": time.monotonic()}
    elif "done" in msg:
        domain = msg.split("[")[1].split("]")[0]
        if domain in timings:
            timings[domain]["elapsed"] = time.monotonic() - timings[domain]["start"]

with AgentOrchestrator(
    use_mcp=False, auto_intuition=True, progress_callback=timed_progress
) as orch:
    result = orch.run("What is RLHF?")

for domain, t in timings.items():
    print(f"{domain}: {t.get('elapsed', '?'):.2f}s")
```

---

## MCP configuration

### Enable / disable MCP

```python
# Enabled (default): retrieves web evidence for all agents
orch = AgentOrchestrator(use_mcp=True)

# Disabled: agents use knowledge only; fully offline
orch = AgentOrchestrator(use_mcp=False)
```

### How MCP affects agent behavior

When `use_mcp=True`, the orchestrator instantiates a single `MCPClient` (DuckDuckGo search, zero-auth) and passes it to every agent. Each agent constructs its search query as:

```python
query = f"{domain.value.replace('_', ' ')} {question}"
# e.g. "healthcare What is the efficacy of mRNA vaccines for RSV?"
```

The domain prefix improves search relevance. Up to 4 results are retrieved and formatted as the MCP context for Pipeline B.

### MCP search failure handling

MCP failures (network down, rate-limited, no results) are caught silently by each agent:
- If the search raises an exception → `mcp_context = ""`
- If `mcp_context` is empty → Pipeline B is skipped
- The agent falls back cleanly to Pipeline A with no error

This means `use_mcp=True` is safe in all environments — network unavailability is handled gracefully.

### When to disable MCP

| Scenario | Use `--no-mcp` / `use_mcp=False` |
|---|---|
| CI / offline testing | ✅ Always |
| Apple Silicon with `--fast` | ✅ Default (override with `--use-mcp`) |
| Derivation / proof questions | ✅ Web evidence rarely helps |
| High-stakes domains needing current data | ❌ Keep MCP on |
| Time-critical questions needing current facts | ❌ Keep MCP on |

---

## Token budget optimization

### Two separate token budgets

```python
orch = AgentOrchestrator(
    agent_max_tokens=1024,      # per-agent LLM call (× 2 with both pipelines)
    synthesis_max_tokens=512,   # synthesis + overall analysis calls
)
```

**Total LLM tokens per question ≈:**
```
N_agents × agent_max_tokens × pipeline_calls_per_agent
+ synthesis_max_tokens × 2 (synthesis + analysis)
```

For a typical run with 5 agents, both pipelines active, and standard budgets:
```
5 agents × 1024 × 2 pipelines = 10,240 agent tokens
+ 512 × 2 = 1,024 synthesis tokens
Total ≈ 11,264 tokens per question
```

### Budget presets

| Preset | `agent_max_tokens` | `synthesis_max_tokens` | Total (5 agents) | Use case |
|---|---|---|---|---|
| Minimal (`--fast`) | 256 | 384 | 3,328 | Local GPU, max speed |
| Standard (default) | 1024 | 512 | 11,264 | Balanced |
| Rich | 1536 | 768 | 16,896 | Derivations, experiments |
| Maximum (tutors) | 2048 | 1024 | 21,504 | Full iterative tutor protocol |

---

## Composing the orchestrator into larger systems

### Pattern 1: Pre-supply intuition from another system

```python
from src.models import HumanIntuition

def get_user_belief_from_database(question: str) -> HumanIntuition:
    # Pull from a user profile, survey response, or prior interaction
    return HumanIntuition(
        question=question,
        intuitive_answer="...",
        confidence=0.6,
        reasoning="...",
        domain_guesses=["deep_learning", "computer_science"],
    )

with AgentOrchestrator(use_mcp=True) as orch:
    intuition = get_user_belief_from_database(question)
    result = orch.run(question, prefilled_intuition=intuition)
```

### Pattern 2: Chaining orchestrators for multi-step reasoning

```python
# Step 1: Get a broad analysis
with AgentOrchestrator(use_mcp=False, auto_intuition=True, max_domains=5) as orch:
    broad_result = orch.run(question)

# Step 2: Extract the most uncertain domain and drill deeper
uncertain_domain = min(
    broad_result.agent_responses,
    key=lambda r: r.confidence,
).domain

with AgentOrchestrator(use_mcp=True, auto_intuition=True) as orch:
    deep_result = orch.run(
        question,
        domains=[uncertain_domain],
        prefilled_intuition=broad_result.human_intuition,
    )

print(f"Deep analysis of {uncertain_domain.value}: {deep_result.synthesized_answer}")
```

### Pattern 3: Progressive confidence — escalate to more agents

```python
def run_with_confidence_floor(question, min_accuracy=60.0):
    """Run with 3 domains; if accuracy is too low, re-run with more."""
    intuition_obj = HumanIntuition(question=question, intuitive_answer="auto", confidence=0.5)

    with AgentOrchestrator(use_mcp=False, auto_intuition=True, max_domains=3) as orch:
        result = orch.run(question, prefilled_intuition=intuition_obj)

    if result.intuition_accuracy_pct >= min_accuracy:
        return result

    # Accuracy too low — retry with more agents
    with AgentOrchestrator(use_mcp=True, auto_intuition=True, max_domains=8) as orch:
        result = orch.run(question, prefilled_intuition=intuition_obj)

    return result
```

### Pattern 4: Inject a custom backend

```python
from src.llm.ollama_backend import OllamaBackend

backend = OllamaBackend(model="qwen2.5:7b", base_url="http://localhost:11434")

with AgentOrchestrator(backend=backend, use_mcp=False, max_workers=1) as orch:
    result = orch.run("Explain attention", prefilled_intuition=intuition)
```

Using `backend=` instead of `backend_spec=` avoids re-initializing the backend for every question in a loop — a significant performance saving.

### Pattern 5: Separate backend instances per agent type

```python
# Fast backend for most agents, slow but powerful for the PhD agent
fast_backend = OllamaBackend(model="qwen2.5:7b")
deep_backend = OllamaBackend(model="llama3.1:70b")

# Run the bulk of agents with the fast backend
with AgentOrchestrator(backend=fast_backend, use_mcp=False, max_domains=3) as orch:
    fast_result = orch.run(question, prefilled_intuition=intuition,
                           domains=[Domain.COMPUTER_SCIENCE, Domain.DEEP_LEARNING])

# Run the research agent with the deep backend separately
from src.agents.ee_llm_research_agent import EELLMResearchAgent
deep_agent = EELLMResearchAgent(backend=deep_backend)
deep_resp = deep_agent.answer(question)
```

---

## Batch and pipeline workflows

### Batch processing multiple questions

```python
questions = [
    "How does RLHF work?",
    "Explain the vanishing gradient problem.",
    "What is the transformer attention mechanism?",
]

results = []
with AgentOrchestrator(
    use_mcp=False,
    auto_intuition=True,
    max_domains=3,
    agent_timeout_seconds=10.0,
) as orch:
    for q in questions:
        r = orch.run(q)
        results.append(r)
        print(f"{q[:50]}: {r.intuition_accuracy_pct:.1f}%")
```

**Why re-use the orchestrator across questions:** the backend connection, thread pool, and MCP client are created once and shared. Creating a new orchestrator per question wastes ~100–200 ms in setup time.

### Parallel batch (independent questions only)

```python
from concurrent.futures import ThreadPoolExecutor

def run_one(question: str) -> WeighingResult:
    # Each thread gets its own orchestrator to avoid shared state
    with AgentOrchestrator(use_mcp=False, auto_intuition=True, max_domains=2) as orch:
        return orch.run(question)

with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(run_one, q) for q in questions]
    results = [f.result() for f in futures]
```

**Warning:** when using Ollama, parallel orchestrators will serialize at the GPU anyway. Only use this pattern with API-based backends (Groq, Together, OpenRouter).

---

## Model evaluation workflow

The `evaluate_models()` entry point runs the same question through multiple backends and compares results.

```python
with AgentOrchestrator(use_mcp=False, auto_intuition=True) as orch:
    comparison = orch.evaluate_models(
        "Explain gradient descent",
        model_specs=[
            "mock",
            "ollama:llama3.1:8b",
            "groq:llama-3.1-8b-instant",
        ],
        domains=[Domain.DEEP_LEARNING, Domain.COMPUTER_SCIENCE],
    )

print(f"Best model: {comparison.best_model_spec}")
print(f"Consensus answer: {comparison.consensus_answer[:200]}")
print(f"Divergence: {comparison.divergence_summary}")

for r in comparison.model_results:
    if r.backend_available and r.weighing_result:
        print(
            f"{r.model_spec}: {r.weighing_result.intuition_accuracy_pct:.1f}% "
            f"({r.duration_seconds:.1f}s)"
        )
    else:
        print(f"{r.model_spec}: UNAVAILABLE — {r.error}")
```

### Model selection via environment variable

```bash
export INTUITION_SCIENTIST_MODELS="mock,ollama:llama3.1:8b,groq:llama-3.1-8b-instant"

python -c "
import os
from src.orchestrator.agent_orchestrator import AgentOrchestrator
from src.models import HumanIntuition

models = os.environ['INTUITION_SCIENTIST_MODELS'].split(',')
with AgentOrchestrator(use_mcp=False, auto_intuition=True) as orch:
    r = orch.evaluate_models('What is gradient descent?', model_specs=models)
print('Best:', r.best_model_spec)
"
```

---

## Debate and multi-party workflows

```python
with AgentOrchestrator(use_mcp=True) as orch:
    debate = orch.debate(
        "Should we always default to microservices for new products?",
        prefilled_intuition=HumanIntuition(
            question="Should we always default to microservices?",
            intuitive_answer="No — monoliths are simpler to start.",
            confidence=0.8,
        ),
        domains=[Domain.ENTERPRISE_ARCHITECTURE, Domain.STRATEGY_INTELLIGENCE],
    )

# Three rounds: factual accuracy, causal reasoning, practical implications
for round_ in debate.rounds:
    print(f"\n=== {round_.topic} ===")
    print(f"Human position: {round_.human_position.claim[:200]}")
    print(f"Agent position: {round_.agent_position.claim[:200]}")
    print(f"Tool position:  {round_.tool_position.claim[:200]}")
    print(f"Agreements: {', '.join(round_.agreements[:3])}")
    print(f"Divergences: {', '.join(round_.divergences[:3])}")

print(f"\nVerdict: {debate.verdict}")
print(f"Key insights: {debate.key_insights}")
```

### When to use `debate()` vs `run()`

| Use `run()` | Use `debate()` |
|---|---|
| Question has a clear answer | Question has multiple defensible positions |
| You want a synthesized answer | You want to see *why* human and agents disagree |
| Accuracy scoring is the goal | Transparency and auditability are the goal |
| Most STEM questions | Strategy, ethics, architecture decisions |

---

## Interview prep workflow

```python
with AgentOrchestrator(use_mcp=False) as orch:
    result = orch.interview_prep(
        "Implement a LRU cache with O(1) get and put",
        prefilled_intuition=HumanIntuition(
            question="Implement a LRU cache with O(1) get and put",
            intuitive_answer="I'd use a doubly linked list and a hash map.",
            confidence=0.8,
            reasoning="The hash map gives O(1) lookup; the linked list gives O(1) eviction.",
        ),
    )

print(f"Technical score: {result.technical_score:.0%}")
print(f"\nFeedback: {result.technical_feedback}")
print(f"\nAlgorithmic depth: {result.algorithmic_insight}")
print(f"\nMental prep: {result.mental_preparation}")
print(f"\nRecommendations:")
for i, rec in enumerate(result.recommendations, 1):
    print(f"  {i}. {rec}")
```

The `interview_prep()` entry point always routes through:
- `InterviewPrepAgent` — technical correctness, LeetCode patterns, system design
- `AlgorithmsProgrammingAgent` — algorithmic depth and implementation patterns
- `SocialScienceAgent` — psychological readiness, communication, STAR framing

---

## Complete workflow recipes

### Recipe 1: Fast CI smoke test

```python
# Runs in < 2 seconds with mock backend; zero network, zero stdin
from src.orchestrator.agent_orchestrator import AgentOrchestrator
from src.models import Domain

with AgentOrchestrator(
    use_mcp=False,
    auto_intuition=True,
    max_domains=2,
    agent_timeout_seconds=5.0,
    max_workers=2,
) as orch:
    result = orch.run(
        "What is 2+2?",
        domains=[Domain.COMPUTER_SCIENCE, Domain.PHYSICS],
    )

assert result.synthesized_answer
assert len(result.agent_responses) == 2
assert 0.0 <= result.intuition_accuracy_pct <= 100.0
```

### Recipe 2: High-quality deep analysis

```python
# Full analysis: all relevant domains, MCP on, generous token budgets
with AgentOrchestrator(
    backend_spec="ollama:llama3.1:8b",
    use_mcp=True,
    max_workers=3,
    auto_intuition=False,       # prompt for human intuition
    agent_max_tokens=1024,
    synthesis_max_tokens=768,
    agent_timeout_seconds=120.0,
) as orch:
    result = orch.run(
        "How should a startup architect its data platform for ML workloads?"
    )
```

### Recipe 3: Adaptive quality floor with time budget

```python
# Expand domains until confident, but never take longer than 20 seconds
with AgentOrchestrator(
    use_mcp=False,
    auto_intuition=True,
    adaptive_agents=True,
    target_latency_ms=20000,
    max_domains=10,
    agent_timeout_seconds=8.0,
    progress_callback=print,
) as orch:
    result = orch.run("Explain the information bottleneck principle in deep learning")
```

### Recipe 4: Longitudinal intuition tracking

```python
# Track how a user's intuition accuracy improves over time
import json, datetime

history = []
questions = [
    "What is gradient descent?",
    "Why does batch normalisation help training?",
    "How does dropout prevent overfitting?",
]

with AgentOrchestrator(use_mcp=False, auto_intuition=False, max_domains=3) as orch:
    for q in questions:
        result = orch.run(q)    # prompts for intuition interactively
        history.append({
            "question": q,
            "accuracy": result.intuition_accuracy_pct,
            "timestamp": datetime.datetime.now().isoformat(),
        })

print(json.dumps(history, indent=2))
# → track whether the user's accuracy improves across sessions
```

### Recipe 5: Research pipeline with escalation

```python
from src.intuition.human_policy import should_escalate

# Phase 1: auto-scan all domains
with AgentOrchestrator(use_mcp=True, auto_intuition=True) as orch:
    result = orch.run("What are the key risks of deploying LLMs in clinical decision support?")

# Phase 2: if escalation triggers, get human input before final synthesis
if should_escalate(
    [r.domain for r in result.agent_responses],
    result.agent_responses,
    use_mcp=True,
):
    print("⚠  High-stakes question with uncertain agents — requesting human review.")
    print(f"\nAuto-synthesized answer: {result.synthesized_answer}\n")
    human_review = input("Enter your expert assessment: ")
    # Store and use for downstream reporting / audit trail
```
