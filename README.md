# Human Intuition Scientist — Architecture and Design

A multi-agent AI system that weighs **human intuition** against **domain-expert AI reasoning** and **MCP/tool-sourced evidence** to produce structured, transparent answers to complex questions.

> **Quick links**
> - 📋 [SCENARIOS.md](SCENARIOS.md) — how to run every mode with expected output
> - 🧪 [RUNNING_TESTS.md](RUNNING_TESTS.md) — step-by-step test instructions
> - 🤖 [docs/AGENT_WORKFLOWS.md](docs/AGENT_WORKFLOWS.md) — optimized workflow guide for all 24 agents
> - 🎛️ [docs/ORCHESTRATOR_WORKFLOWS.md](docs/ORCHESTRATOR_WORKFLOWS.md) — orchestrator configuration and composition patterns

---

## Table of contents

1. [System overview](#system-overview)
2. [End-to-end architecture](#end-to-end-architecture)
3. [Human involvement policy](#human-involvement-policy)
4. [Dual-pipeline weight blending](#dual-pipeline-weight-blending)
5. [Domain agents](#domain-agents)
6. [Weighing and synthesis pipeline](#weighing-and-synthesis-pipeline)
7. [Adaptive agent loop](#adaptive-agent-loop)
8. [Supported model backends](#supported-model-backends)
9. [Project structure](#project-structure)
10. [Installation](#installation)
11. [Environment variable reference](#environment-variable-reference)

---

## System overview

| Feature | Description |
|---|---|
| **24 domain agents** | Science, industry, enterprise, mastery, research, and experiment domains |
| **Dual-pipeline** | Every agent runs an *intuition path* (knowledge only) **and** a *tool/MCP path* (web-grounded), then blends them with intelligent weights |
| **Human involvement policy** | Configurable AUTO / ALWAYS / NEVER policy; AUTO escalates to interactive prompting only when the question is high-stakes, confidence is low, or agents strongly disagree |
| **Iterative hard-problem tutors** | Physics and Signal Processing agents select the *hardest applicable problem type* and guide learners step-by-step with checkpoints and targeted hints |
| **Experiment Runner** | Classifies which questions warrant experiments, then generates a structured set of targeted experiments (hypotheses, variables, runnable Python/NumPy snippets) |
| **Auto-intuition mode** | `--non-interactive` (or legacy `--auto-intuition`) skips the interactive prompt; the system auto-generates a human perspective via keyword heuristics + optional LLM quick-think |
| **Adaptive agent loop** | `--adaptive-agents` starts with 3 agents and expands only when confidence is insufficient, with optional `--target-latency-ms` budget |
| **Debate harness** | Structured multi-party debate: human vs. tool evidence vs. agents |
| **Interview coach** | Socratic FAANG interview prep with 100+ practice problems and hints |
| **Model sweep** | Cycle across any open-source or free-tier model backends; compare results |
| **Free-only models** | Anthropic and OpenAI are disabled; only free/open backends supported |
| **Per-agent progress** | Observable logging of domain selection, agent start/finish, MCP results, and pipeline dominance |

---

## End-to-end architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  CLI (main.py)                                                  │
│  • Parses flags: --interactive, --non-interactive,              │
│    --human-policy, --verbose, --quiet, --domains, --fast, …    │
│  • Resolves HumanPolicy (AUTO / ALWAYS / NEVER)                 │
│  • Wires progress_callback → terminal output                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  AgentOrchestrator (src/orchestrator/agent_orchestrator.py)    │
│                                                                 │
│  Step 1 — Human Intuition                                       │
│    ├── AUTO policy + high-stakes domain → interactive capture   │
│    ├── AUTO policy + normal domain → auto-generate intuition    │
│    ├── ALWAYS policy → interactive capture (IntuitionCapture)   │
│    └── NEVER policy → auto-generate (generate_auto_intuition)  │
│                                                                 │
│  Step 2 — Domain selection                                      │
│    ├── Explicit --domains → use as given                        │
│    ├── --adaptive-agents → _adaptive_select_and_run()           │
│    └── Default → _select_domains() (keyword inference, ≥3)     │
│                                                                 │
│  Step 3 — Agent querying (_query_agents)                        │
│    ├── ThreadPoolExecutor (max_workers parallel calls)          │
│    ├── Per-agent progress emitted via progress_callback         │
│    ├── Per-agent timeout (agent_timeout_seconds)                │
│    └── Returns list[AgentResponse] in submission order          │
│                                                                 │
│  Step 4 — Weighing                                              │
│    └── WeighingSystem.weigh(intuition, responses)               │
│        → WeighingResult                                         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────────┐
          ▼               ▼                   ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐
│ BaseAgent    │  │ MCPClient    │  │ WeighingSystem           │
│ (per domain) │  │ (web search) │  │                          │
│              │  │              │  │ • Compute alignment       │
│ Pipeline A:  │  │ DuckDuckGo   │  │   scores (semantic sim.)  │
│  LLM-only    │  │ zero-auth    │  │ • Synthesize final answer │
│              │  │              │  │ • Score intuition accuracy│
│ Pipeline B:  │  │ Returns MCP  │  │ • Generate recommendations│
│  MCP context │  │ context used │  │                          │
│  → LLM call  │  │ in Pipeline B│  └──────────────────────────┘
│              │  │              │
│ Blend A+B    │  └──────────────┘
│ (weight pair)│
└──────────────┘
```

### Data flow

```
Question
    │
    ├─ HumanIntuition (interactive or auto)
    │       question, intuitive_answer, confidence, reasoning, domain_guesses
    │
    ├─ AgentResponse × N (one per domain)
    │       domain, answer, reasoning, confidence,
    │       mcp_context, intuition_weight, tool_weight
    │
    └─ WeighingResult
            question, human_intuition, agent_responses,
            alignment_scores, synthesized_answer,
            overall_analysis, intuition_accuracy_pct, recommendations
```

---

## Human involvement policy

The policy is implemented in `src/intuition/human_policy.py` and is fully testable with no side-effects.

### Policy levels

| Policy | Flag | Behaviour |
|---|---|---|
| `AUTO` | *(default)* | Interactive only when an escalation trigger fires |
| `ALWAYS` | `--interactive` / `--human-policy always` | Always prompt |
| `NEVER` | `--non-interactive` / `--human-policy never` | Never prompt; always auto-generate |

### Escalation triggers (AUTO mode)

The system escalates to interactive prompting when **any one** of these conditions fires:

| Trigger | Condition |
|---|---|
| **High-stakes domain** | The question is inferred to belong to Healthcare, Legal & Compliance, or Finance & Economics |
| **Low overall confidence** | Mean agent confidence < 0.40 (after agents run) |
| **High disagreement** | Max − min agent confidence spread > 0.45 |
| **MCP expected but absent** | MCP is enabled, the question touches a tool-heavy domain, but no agent received any MCP context |

For domain-level escalation (high-stakes), the check runs *before* agents are queried so the system can switch to interactive capture immediately. The confidence and disagreement checks are available for post-run re-prompting in workflows that need it.

### Design rationale

- **Default is non-interactive**: most questions do not need deep human judgment; making the default interactive forces users to type before seeing any results.
- **Escalation keeps the human in the loop where it matters**: a question about warfarin interactions is different from "what is 2+2?".
- **Policy is decoupled from the orchestrator**: `human_policy.py` has no imports from the orchestrator, making it independently testable.

---

## Dual-pipeline weight blending

Every agent runs **both** pipelines for every question and blends the results:

```
Pipeline A — intuition (knowledge only)
    System prompt: domain expert persona
    LLM call: question → expert answer

Pipeline B — tool (MCP-grounded)
    MCP search: retrieve web evidence
    System prompt: domain expert + evidence context
    LLM call: question + evidence → grounded answer

Weight computation:
    base_intuition  ←  domain type
                        interpretive (legal, social, strategy): 0.65
                        empirical (healthcare, finance, cyber):  0.40
                        balanced (cs, physics, dl):              0.55
    tool_boost      ←  MCP result quality × 0.20
    type_modifier   ←  ±0.10 (factual "what/when" → +tool;
                                analytical "why/how" → +intuition)

    tool_weight      = clamp(base_tool + tool_boost + type_modifier, 0.10, 0.75)
    intuition_weight = 1.0 − tool_weight
```

The dominant pipeline provides the main answer text; the minority pipeline contributes a bracketed insight. Both weights are recorded in `AgentResponse` for full transparency and are visible in the per-agent progress log:

```
  ✓  [Deep Learning] done — conf=82%, pipeline=tool (MCP: results)
  ✓  [Legal Compliance] done — conf=74%, pipeline=intuition (MCP: none)
```

---

## Domain agents

The system has **24 domain agents** in four groups.

### Core science & engineering

| Domain | Shortcut | Description |
|---|---|---|
| Electrical Engineering | `ee` | Circuits, signals, power, control systems |
| Computer Science | `cs` | Algorithms, complexity, software systems |
| Neural Networks | `nn` | Deep theory, signal processing connections, next-gen architectures |
| Social Science | `social` | Behaviour, psychology, sociology, game theory |
| Space Science | `space` | Astrophysics + 7 planetary exploration scenarios |
| Physics | `physics` | Quantum, relativity, condensed matter — iterative hard-problem tutor |
| Deep Learning | `dl` | Transformers, SSMs, diffusion, scaling, alignment |
| Signal Processing | `signal` / `dsp` | Filter design, spectral estimation, adaptive filters — iterative hard-problem tutor |

### High-economic-value industry (high-stakes for AUTO policy)

| Domain | Shortcut | Description |
|---|---|---|
| Healthcare | `healthcare` | Drug discovery, diagnostics, clinical trials |
| Climate & Energy | `climate` | Renewables, grid, decarbonisation, tipping points |
| Finance & Economics | `finance` | Quant finance, risk, macro economics |
| Cybersecurity | `cyber` | Threat analysis, adversarial ML, cryptography |
| Biotech & Genomics | `biotech` | CRISPR, protein folding, synthetic biology |
| Supply Chain | `supply_chain` | Logistics, resilience, demand forecasting |

### Enterprise problems

| Domain | Shortcut | Description |
|---|---|---|
| Legal & Compliance | `legal` | Contract law, GDPR, IP, regulatory risk |
| Enterprise Architecture | `architecture` | Cloud, microservices, technical debt |
| Marketing & Growth | `marketing` | Attribution, CLV, PLG, A/B testing |
| Organisational Behaviour | `org` | Talent, culture, leadership, workforce planning |
| Strategy & Intelligence | `strategy` | Competitive moats, M&A, scenario planning |

### Mastery, interview, research, and experiments

| Domain | Shortcut | Description |
|---|---|---|
| Algorithms & Programming | `algo` | Python, Rust, Go; all DS&A patterns |
| Interview Prep | `interview` / `faang` | 100+ LeetCode patterns + system design + STAR coaching |
| EE LLM Research | `phd` / `ee_llm` | LLMs, signal processing, LLM safety — PhD-level advisor |
| Experiment Runner | `experiment` / `simulate` | Classifies questions by experimentability; generates experiment specs with runnable Python/NumPy snippets |

---

## Weighing and synthesis pipeline

`WeighingSystem.weigh(intuition, agent_responses)` produces a `WeighingResult`:

1. **Alignment scoring** — for each agent response, compute semantic similarity to the human intuition using token-overlap heuristics. Extract key agreements and divergences.
2. **Synthesis** — combine agent answers (weighted by confidence) into a single synthesized answer via an LLM call.
3. **Overall analysis** — deep analysis of the collective response, surfacing tensions and supporting evidence via an LLM call.
4. **Intuition accuracy** — weighted mean of alignment scores, expressed as a percentage. Weights are proportional to agent confidence.
5. **Recommendations** — actionable next steps derived from the analysis.

The full `WeighingResult` is printed to the terminal with sections for: Human Intuition, Domain-by-Domain Alignment (table), Expert Agent Answers, Synthesized Answer, Deep Analysis, Intuition Accuracy Score, Recommendations, and Agentic Workflow (if `--workflow-map` is not `off`).

---

## Adaptive agent loop

When `--adaptive-agents` is set, the orchestrator uses an expanding loop instead of a fixed domain set:

```
1. Rank all domains by keyword relevance to question + intuition.
2. Query the top 3 (initial batch) in parallel.
3. Compute mean confidence.
4. If mean ≥ 0.65 → stop.
5. Otherwise add next 2 highest-ranked domains and re-query only those.
6. Repeat until a stopping criterion is met:
     • Mean confidence ≥ 0.65
     • No remaining candidate domains
     • max_domains ceiling reached
     • target_latency_ms wall-clock budget exceeded
```

This balances thoroughness against latency: a question with a clear, confident answer in 3 domains never wastes time on 20 more.

---

## Supported model backends

> **Note:** Anthropic (`claude-*`) and OpenAI (`gpt-*`) are intentionally **not supported**. Only free and open-source providers are allowed.

### Local backends (always free)

| Backend | Spec format | Notes |
|---|---|---|
| **Mock** | `mock` | Built-in offline backend; instant, deterministic; used in all tests |
| **Ollama** | `ollama:<model>` | Local inference via [Ollama](https://ollama.com) |
| **llama.cpp** | `llamacpp:<path/to/model.gguf>` | Local inference from a GGUF file |

### Hosted free-tier backends

| Backend | Spec format | Required env var(s) |
|---|---|---|
| **Groq** | `groq:<model>` | `GROQ_API_KEY` |
| **Together AI** | `together:<model>` | `TOGETHER_API_KEY` |
| **Cloudflare Workers AI** | `cloudflare:<model>` | `CF_ACCOUNT_ID` + `CF_API_TOKEN` |
| **OpenRouter** | `openrouter:<model>` | `OPENROUTER_API_KEY` |

---

## Project structure

```
intuition-scientist/
├── main.py                             # CLI entry point
├── requirements.txt
├── README.md                           # Architecture and design (this file)
├── SCENARIOS.md                        # How to run scenarios with expected output
├── RUNNING_TESTS.md                    # Step-by-step test instructions
├── src/
│   ├── agents/
│   │   ├── base_agent.py               # Dual-pipeline base (intuition + MCP weights)
│   │   ├── algorithms_programming_agent.py
│   │   ├── biotech_genomics_agent.py
│   │   ├── climate_energy_agent.py
│   │   ├── computer_science_agent.py
│   │   ├── cybersecurity_agent.py
│   │   ├── deep_learning_agent.py
│   │   ├── ee_llm_research_agent.py
│   │   ├── electrical_engineering_agent.py
│   │   ├── enterprise_architecture_agent.py
│   │   ├── experiment_runner_agent.py  # Experiment classification + plan generation
│   │   ├── finance_economics_agent.py
│   │   ├── healthcare_agent.py
│   │   ├── interview_prep_agent.py     # 100+ DS&A practice problems + system design
│   │   ├── legal_compliance_agent.py
│   │   ├── marketing_growth_agent.py
│   │   ├── neural_networks_agent.py
│   │   ├── organizational_behavior_agent.py
│   │   ├── physics_agent.py            # Iterative hard-problem tutor (10 problem types)
│   │   ├── signal_processing_agent.py  # Iterative hard-problem tutor (10 problem types)
│   │   ├── social_science_agent.py
│   │   ├── space_science_agent.py
│   │   ├── strategy_intelligence_agent.py
│   │   └── supply_chain_agent.py
│   ├── analysis/
│   │   ├── debate_engine.py            # Multi-party debate harness
│   │   └── weighing_system.py          # Human intuition vs. agent weighing + synthesis
│   ├── intuition/
│   │   ├── human_intuition.py          # IntuitionCapture + domain inference + auto-generate
│   │   └── human_policy.py             # HumanPolicy enum + escalation triggers
│   ├── llm/
│   │   ├── base.py                     # LLMBackend protocol
│   │   ├── registry.py                 # Free-only provider registry (blocks Anthropic/OpenAI)
│   │   ├── mock_backend.py             # Offline CI backend
│   │   ├── ollama_backend.py
│   │   ├── llamacpp_backend.py
│   │   ├── groq_backend.py
│   │   ├── together_backend.py
│   │   ├── cloudflare_backend.py
│   │   └── openrouter_backend.py
│   ├── mcp/
│   │   └── mcp_client.py               # DuckDuckGo web search (zero-auth)
│   ├── models.py                       # All dataclasses (WeighingResult, AgentResponse, …)
│   ├── orchestrator/
│   │   └── agent_orchestrator.py       # run() / debate() / interview_prep() / evaluate_models()
│   └── workflow/
│       └── …                           # Workflow trace + rendering (off/compact/standard/deep)
└── tests/
    ├── fixtures/
    │   └── qa.yaml                     # Complex Q&A with human intuition prefills
    ├── test_agents.py
    ├── test_auto_intuition_adaptive.py
    ├── test_experiment_runner.py
    ├── test_human_policy.py            # Human policy + escalation + new CLI flags
    ├── test_intuition.py
    ├── test_large_suite.py             # 200+ tests across all domains × modes
    ├── test_llm_registry.py
    ├── test_model_sweep.py             # Opt-in: RUN_MODEL_SWEEP=1
    ├── test_models.py
    ├── test_new_agents.py
    ├── test_orchestrator.py
    ├── test_performance_features.py
    ├── test_weighing_system.py
    └── test_workflow.py
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/skanjila/intuition-scientist.git
cd intuition-scientist

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

No API keys are required to run in **mock mode** (the default).

---

## Environment variable reference

| Variable | Default | Description |
|---|---|---|
| `RUN_MODEL_SWEEP` | *(unset)* | Set to `1` to enable model sweep tests |
| `INTUITION_SCIENTIST_MODELS` | `mock` | Comma-separated model spec list for sweep tests |
| `GROQ_API_KEY` | *(unset)* | Groq API key (free at console.groq.com) |
| `TOGETHER_API_KEY` | *(unset)* | Together AI API key |
| `CF_ACCOUNT_ID` | *(unset)* | Cloudflare account ID |
| `CF_API_TOKEN` | *(unset)* | Cloudflare API token |
| `OPENROUTER_API_KEY` | *(unset)* | OpenRouter API key |

---

*Anthropic and OpenAI are intentionally not supported. This project is free-only.*
