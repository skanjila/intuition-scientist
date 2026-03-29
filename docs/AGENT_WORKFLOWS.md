# Agent Workflow Optimization Guide

Detailed instructions for building the most effective workflows with each of the 24 domain agents — covering system-prompt design, dual-pipeline tuning, token budgets, MCP weight calibration, question routing, and how to add a new agent.

---

## Table of contents

1. [BaseAgent architecture](#baseagent-architecture)
2. [Dual-pipeline mechanics](#dual-pipeline-mechanics)
3. [Writing an optimal system prompt](#writing-an-optimal-system-prompt)
4. [Pipeline weight tuning reference](#pipeline-weight-tuning-reference)
5. [Token budget guidelines](#token-budget-guidelines)
6. [Per-agent workflow guide](#per-agent-workflow-guide)
   - [Core science & engineering agents](#core-science--engineering-agents)
   - [High-economic-value industry agents](#high-economic-value-industry-agents)
   - [Enterprise problem agents](#enterprise-problem-agents)
   - [Mastery, research, and experiment agents](#mastery-research-and-experiment-agents)
7. [Routing questions to the right agent](#routing-questions-to-the-right-agent)
8. [Adding a new custom agent](#adding-a-new-custom-agent)
9. [Common pitfalls and fixes](#common-pitfalls-and-fixes)
10. [Solver Policy — strategy-selection toggle](#solver-policy--strategy-selection-toggle)

---

## BaseAgent architecture

Every agent inherits from `src/agents/base_agent.py`. The only method you **must** implement is:

```python
def _build_system_prompt(self) -> str:
    """Return the system-level prompt that defines this agent's domain expertise."""
```

Everything else — MCP retrieval, JSON extraction, weight computation, blending — is handled by `BaseAgent.answer()`.

### Full call graph for `agent.answer(question)`

```
answer(question)
  │
  ├─ Pipeline A — pure intuition
  │     _call_llm(question, mcp_context="")
  │       _build_system_prompt()       ← YOUR ONLY REQUIRED HOOK
  │       _build_user_message()        ← asks for JSON output
  │       backend.generate()           ← LLM call
  │     _extract_json(raw)             ← parse {"answer", "reasoning", "confidence", "sources"}
  │
  ├─ MCP retrieval (if mcp_client is not None)
  │     mcp_client.search(domain + question, num_results=4)
  │     _format_search_context(results)
  │
  ├─ _compute_weights(question, search_results)
  │     domain classification  →  base_intuition ∈ {0.40, 0.55, 0.65}
  │     MCP quality modifier   →  +0.00 to +0.20 tool boost
  │     question-type modifier →  ±0.10
  │
  ├─ Pipeline B — tool-grounded (only if tool_w ≥ 0.20 and mcp_context non-empty)
  │     _call_llm(question, mcp_context)
  │     _extract_json(raw)
  │
  └─ _blend_and_build()  →  AgentResponse
```

### AgentResponse fields

| Field | Type | Meaning |
|---|---|---|
| `domain` | `Domain` | Which domain this agent covers |
| `answer` | `str` | Blended final answer |
| `reasoning` | `str` | Step-by-step reasoning |
| `confidence` | `float` | Weighted confidence (0.0–1.0) |
| `sources` | `list[str]` | URLs from MCP results |
| `mcp_context` | `str` | Raw MCP search context (empty when MCP disabled) |
| `intuition_weight` | `float` | Weight given to Pipeline A |
| `tool_weight` | `float` | Weight given to Pipeline B |

---

## Dual-pipeline mechanics

### When both pipelines run

Pipeline B (tool-grounded) only runs when **both** conditions are true:
1. `tool_weight >= 0.20` — the domain/question type warrants external evidence.
2. `mcp_context` is non-empty — the MCP search returned results.

If MCP returns nothing (network down, `--no-mcp` flag, or search failure), the agent falls back entirely to Pipeline A. This means Pipeline A must always be capable of producing a complete, high-quality answer on its own.

### Blending logic

```python
if tool_weight >= intuition_weight:
    primary_answer = tool_data["answer"]
    secondary_note = f" [Intuition insight ({intuition_weight:.0%}): {intuition_answer[:200]}]"
else:
    primary_answer = intuition_data["answer"]
    secondary_note = f" [Tool evidence ({tool_weight:.0%}): {tool_answer[:200]}]"

final_answer = primary_answer + secondary_note
final_confidence = intuition_weight * i_conf + tool_weight * t_conf  # weighted average
```

The minority pipeline's contribution is appended as a bracketed note. This means the secondary pipeline must produce **something distinct and useful** — otherwise the note adds noise.

**Optimization tip**: write your system prompt so the agent produces terse, high-signal answers (≤ 3 sentences per section). The blending will truncate the secondary answer at 200 chars, so front-load the most important claim.

---

## Writing an optimal system prompt

A great system prompt has three sections in this order:

### 1. Role declaration (1–2 sentences)
State the expert identity, credibility signals, and the most important capability.

```python
"You are a world-class [DOMAIN] expert with deep knowledge of [CORE AREAS]. "
"You [KEY CAPABILITY that differentiates this agent]."
```

**Do:**
- Cite specific sub-fields, named techniques, and key papers/frameworks.
- State what the agent *does* rather than what it *knows*.

**Do not:**
- Use vague phrases like "knowledgeable in many areas".
- Pad with adjectives that don't constrain the LLM's behavior.

### 2. Knowledge map (structured list)
Cover the most important sub-domains with enough specificity to steer the LLM away from generic answers. Use `===` section headers so the LLM treats them as reference categories.

```python
"=== CORE THEORY ===\n"
"- [Concept A]: [specific detail, formula, or named result]\n"
"- [Concept B]: [specific detail]\n"
"\n=== ADVANCED TOPICS ===\n"
"- [Topic C]: [named algorithm, paper, or method]\n"
```

**Why this matters**: the LLM's attention mechanism treats explicit enumeration of named concepts as a strong prior. A prompt that names `Mamba, RWKV, xLSTM` elicits far more specific answers than one that says "state-space models".

### 3. Output contract (1–3 sentences)
Instruct the agent on output style and format.

```python
"Ground every answer in [TYPE OF EVIDENCE]. "
"Distinguish between [ESTABLISHED FACT] and [SPECULATION]. "
"Respond only with the requested JSON structure."
```

The last line — `"Respond only with the requested JSON structure."` — is **mandatory** on every agent. The `_build_user_message` method appends the JSON schema, but the system prompt must reinforce it or some LLMs will add markdown or prose.

### Full template

```python
def _build_system_prompt(self) -> str:
    return (
        "You are a world-class [ROLE] with expertise in:\n"
        "- [Sub-field A]: [specific named techniques/results]\n"
        "- [Sub-field B]: [specific named techniques/results]\n"
        "- [Sub-field C]: [specific named techniques/results]\n"
        "\n"
        "=== [SECTION A] ===\n"
        "- [Key concept 1]: [detail]\n"
        "- [Key concept 2]: [detail]\n"
        "\n"
        "=== [SECTION B] ===\n"
        "- [Key concept 3]: [detail]\n"
        "\n"
        "[Output instructions]. "
        "Respond only with the requested JSON structure."
    )
```

---

## Pipeline weight tuning reference

The base weight is set by domain classification in `base_agent.py`. Override the domain's classification by subclassing `_compute_weights`:

```python
def _compute_weights(self, question, mcp_results):
    # Example: force high tool weight for this domain regardless of question type
    mcp_quality = min(1.0, len(mcp_results) / 4)
    tool_w = max(0.10, min(0.75, 0.55 + mcp_quality * 0.20))
    return round(1.0 - tool_w, 3), round(tool_w, 3)
```

### Domain classification table

| Classification | Base intuition weight | Domains |
|---|---|---|
| **Intuition-heavy** | 0.65 | Social Science, Strategy, Org Behaviour, Marketing, Interview Prep, Algorithms, EE LLM Research, Physics, Neural Networks, Deep Learning, Signal Processing, Experiment Runner |
| **Tool-heavy** | 0.40 | Healthcare, Cybersecurity, Legal & Compliance, Supply Chain, Finance & Economics, Climate & Energy, Biotech & Genomics |
| **Balanced** | 0.55 | Electrical Engineering, Computer Science, Space Science, Enterprise Architecture |

### Full weight formula

```
raw_tool = (1.0 - base_intuition)
           + min(1.0, num_mcp_results / 4) * 0.20   # MCP quality boost: 0–0.20
           ± 0.10                                    # question-type modifier

tool_weight      = clamp(raw_tool, 0.10, 0.75)
intuition_weight = 1.0 - tool_weight
```

### Question-type modifier

| Pattern matched | Modifier | Examples |
|---|---|---|
| Factual regex: `what is`, `when did`, `who invented`, `how many`, `list`, `define` | +0.10 tool | "What is the cure rate for melanoma?" |
| Analytical regex: `why`, `how does`, `explain`, `analyse`, `compare`, `design` | −0.10 tool | "Why does dropout improve generalisation?" |
| Neither | 0.00 | "Calculate the Fourier transform of a Gaussian" |

---

## Token budget guidelines

| Agent group | Recommended `--agent-max-tokens` | Rationale |
|---|---|---|
| Science & Engineering (non-iterative) | 512–1024 | Technical answers are dense; need room for derivations |
| Physics / Signal Processing (iterative tutor) | 1024–2048 | Step-by-step checkpoints require more tokens |
| Healthcare / Legal / Finance | 768–1024 | Clinical/legal precision requires careful hedging |
| Enterprise / Strategy | 512–768 | Decision-focused; shorter answers are more actionable |
| Interview Prep | 1024–1536 | Must include problem + hints + pattern name |
| Experiment Runner | 1024–2048 | Full experiment spec + Python snippet |

Set per-run via: `python main.py --agent-max-tokens 768`

Set per-agent by overriding `__init__`:
```python
class MyAgent(BaseAgent):
    def __init__(self, **kwargs):
        kwargs.setdefault("max_tokens", 768)
        super().__init__(**kwargs)
```

---

## Per-agent workflow guide

### Core science & engineering agents

---

#### Electrical Engineering (`ee`)

**Best for:** circuit analysis, control systems, power electronics, RF/antenna design, semiconductor devices.

**Optimal question forms:**
```bash
# Derivation — triggers analytical modifier, high intuition weight
python main.py --domains ee --question "Derive the transfer function of a second-order RLC filter"

# Design — triggers analytical modifier
python main.py --domains ee --question "Design a PI controller for a DC motor with 50 ms rise time"

# Factual + MCP — triggers tool-boost
python main.py --domains ee --question "What is the current efficiency record for perovskite solar cells?"
```

**Workflow tip:** EE questions involving numerical design (Bode plots, Nyquist, stability margins) benefit from `--no-mcp` because MCP results for EE tend to be textbook excerpts rather than derivations. Pure knowledge-mode answers are often more precise.

```bash
python main.py --domains ee --no-mcp \
  --question "Derive the Nyquist stability criterion from the argument principle"
```

**Weight profile:** Balanced (0.55 base intuition). For "design" questions, analytical modifier pushes to ~0.65 intuition. For "what is the standard voltage for X" questions, factual modifier pushes to ~0.55 tool.

---

#### Computer Science (`cs`)

**Best for:** algorithms, complexity, operating systems, distributed systems, networks, databases.

**Optimal question forms:**
```bash
# Complexity / proof — pure knowledge, disable MCP
python main.py --domains cs --no-mcp \
  --question "Prove that 3-SAT is NP-complete via reduction from Circuit-SAT"

# Practical design — MCP useful for current best practices
python main.py --domains cs \
  --question "What are the consistency models in modern distributed databases?"

# Multi-domain: CS + algorithms
python main.py --domains cs algo --no-mcp \
  --question "Implement and analyse Dijkstra's algorithm for sparse graphs"
```

**Weight profile:** Balanced (0.55). Multi-domain combination with `algo` is recommended for coding-heavy questions.

---

#### Neural Networks (`nn`)

**Best for:** deep learning theory, backpropagation, gradient flow, RNN/LSTM/GRU, attention mechanisms, next-generation architectures.

**Optimal question forms:**
```bash
# Theory — disable MCP for first-principles derivation
python main.py --domains nn --no-mcp \
  --question "Derive the vanishing gradient problem in deep RNNs from first principles"

# Architecture comparison — MCP useful for recent results
python main.py --domains nn \
  --question "How does Mamba's selective state-space compare to attention in terms of expressive power?"

# Combine with Deep Learning for frontier questions
python main.py --domains nn dl \
  --question "What makes xLSTM different from standard LSTM in terms of memory capacity?"
```

**Weight profile:** Intuition-heavy (0.65). Theoretical derivations should always use `--no-mcp`. For "latest results" questions, MCP adds value.

---

#### Social Science (`social`)

**Best for:** group dynamics, cognitive biases, social influence, institutional economics, organisational psychology.

**Optimal question forms:**
```bash
# Interpretive — high intuition weight, no MCP needed
python main.py --domains social --no-mcp \
  --question "Why do organisations resist change even when change is clearly beneficial?"

# Evidence-based — use MCP for recent studies
python main.py --domains social \
  --question "What does the evidence say about the effectiveness of four-day work weeks?"

# Combine with org for workforce questions
python main.py --domains social org \
  --question "How do psychological safety and team performance relate?"
```

**Weight profile:** Intuition-heavy (0.65). For questions starting with "what does the evidence say", the factual modifier will push slightly toward tool. Interpretive/analytical questions stay intuition-dominant.

---

#### Space Science (`space`)

**Best for:** astrophysics, planetary science, cosmology, orbital mechanics, NASA missions.

**Optimal question forms:**
```bash
# Current mission data — MCP adds real value
python main.py --domains space \
  --question "What have the James Webb Space Telescope findings revealed about the early universe?"

# Theoretical — disable MCP
python main.py --domains space --no-mcp \
  --question "Derive the Schwarzschild radius for a black hole of mass M"

# Exploration scenarios (7 built-in): Mars, Europa, Titan, Venus, Moon, asteroids, exoplanet imaging
python main.py --domains space \
  --question "What are the engineering challenges for a Europa subsurface ocean lander?"
```

**Weight profile:** Balanced (0.55). Current-event questions (JWST, Mars missions) benefit significantly from MCP. Theoretical questions are better served without it.

---

#### Physics (`physics`) — iterative hard-problem tutor

**Best for:** quantum mechanics, statistical mechanics, general relativity, condensed matter, field theory. Excels at step-by-step derivations with checkpoints.

**Optimal workflow — iterative tutor mode:**
```bash
# Always use --interactive for the iterative tutor; the agent will ask for
# your intuitive approach before selecting the hardest applicable problem type.
python main.py --domains physics --interactive \
  --question "Derive the path integral propagator for a free particle"
```

**10 problem categories (hardest to most accessible):**
1. Path-integral formulation
2. Gauge field theory
3. Many-body quantum mechanics
4. Non-linear dynamics / chaos
5. Tensor / GR calculation
6. Statistical mechanics (phase transitions)
7. Variational / Lagrangian mechanics
8. Electromagnetic boundary-value problem
9. Quantum scattering / perturbation theory
10. Dimensional analysis and scaling

**Token budget:** Use `--agent-max-tokens 2048` for full derivations with all checkpoints.

**Workflow tip:** For the tutor to work best, supply a non-trivial question that maps to one of the 10 problem categories. Generic questions ("explain quantum mechanics") trigger the lowest category; targeted derivation questions unlock the hardest problems.

---

#### Deep Learning (`dl`)

**Best for:** transformer architectures, SSMs, diffusion models, RL/RLHF, training at scale, LLM inference optimization.

**Optimal question forms:**
```bash
# Architecture internals — disable MCP for precise derivation
python main.py --domains dl --no-mcp \
  --question "Explain the mathematical equivalence between linear attention and SSMs"

# Frontier research — MCP useful for recent papers
python main.py --domains dl \
  --question "What are the most recent findings on emergent capabilities in frontier LLMs?"

# System design — combine with EE LLM for PhD-level depth
python main.py --domains dl phd \
  --question "Design a training curriculum for a 70B parameter model from scratch"
```

**Weight profile:** Intuition-heavy (0.65). The system prompt covers 200+ specific named techniques (RoPE, FlashAttention, Chinchilla, RLHF, DPO, etc.) so knowledge-only mode is often superior to MCP for architectural questions.

---

#### Signal Processing (`signal` / `dsp`) — iterative hard-problem tutor

**Best for:** filter design, spectral estimation, adaptive filtering, compressed sensing, wavelets, multirate systems.

**Optimal workflow — iterative tutor mode:**
```bash
python main.py --domains signal --interactive \
  --question "Design an optimal Wiener filter for a speech denoising application"
```

**10 problem categories (hardest to most accessible):**
1. Optimal Wiener / Kalman filter derivation
2. Compressed sensing and sparse recovery
3. Multirate filter-bank design
4. Spectral estimation (MUSIC / ESPRIT)
5. Adaptive filtering (LMS / RLS)
6. Nonlinear signal processing (Volterra series)
7. Statistical signal detection (Neyman-Pearson / CFAR)
8. Stochastic process analysis (WSS, PSD, Wiener-Khinchin)
9. Digital filter design (FIR/IIR: Parks-McClellan, bilinear transform)
10. Discrete Fourier analysis and FFT

**Token budget:** Use `--agent-max-tokens 2048`. Signal processing derivations require full mathematical notation.

**Workflow tip for iterative mode:** phrase your question as a design or derivation task ("design", "derive", "optimise") rather than a definition ("what is"). Design tasks unlock the hardest categories.

---

### High-economic-value industry agents

These agents have `base_intuition = 0.40` (tool-heavy). MCP evidence is especially important. **Always run with MCP enabled** for these domains.

---

#### Healthcare (`healthcare`)

**Best for:** drug discovery, precision medicine, clinical trials, diagnostics AI, epidemiology, health economics.

**Critical workflow rule:** Healthcare is a high-stakes domain. The AUTO human policy will escalate to interactive mode for healthcare questions — this is intentional. Use `--non-interactive` only in CI/scripting contexts where you understand the limitation.

```bash
# Patient safety research — keep MCP on for current evidence
python main.py --domains healthcare \
  --question "What is the current evidence for GLP-1 agonists in non-alcoholic steatohepatitis?"

# Drug mechanism — disable MCP for first-principles biochemistry
python main.py --domains healthcare --no-mcp \
  --question "Explain the mechanism by which PARP inhibitors cause synthetic lethality in BRCA-mutated tumours"

# Health policy — MCP useful for recent data
python main.py --domains healthcare \
  --question "What are the key drivers of variation in ICU utilisation across OECD countries?"
```

**Token budget:** `--agent-max-tokens 1024`. Healthcare answers must hedge appropriately ("established practice vs. emerging evidence") and this requires space.

**MCP tip:** MCP for healthcare retrieves PubMed abstracts, FDA press releases, and news. For mechanism questions the LLM's trained knowledge is more reliable; save MCP for current-events questions.

---

#### Climate & Energy (`climate`)

**Best for:** renewable energy, grid decarbonisation, carbon markets, tipping points, climate modelling.

```bash
# Policy + current data — MCP essential
python main.py --domains climate \
  --question "What is the current trajectory of global solar LCOE and what are the major cost drivers?"

# Physical modelling — MCP optional
python main.py --domains climate --no-mcp \
  --question "Explain how permafrost thawing creates a self-reinforcing climate feedback loop"

# Combined with supply chain for grid infrastructure
python main.py --domains climate supply_chain \
  --question "What are the critical materials bottlenecks for scaling grid-scale battery storage?"
```

---

#### Finance & Economics (`finance`)

**Best for:** quantitative finance, risk management, macroeconomics, derivatives pricing, portfolio theory.

**Critical workflow rule:** Finance is a high-stakes domain; AUTO policy escalates to interactive.

```bash
# Market microstructure — disable MCP for rigorous theory
python main.py --domains finance --no-mcp \
  --question "Derive the Black-Scholes equation from the risk-neutral pricing argument"

# Current market conditions — MCP essential
python main.py --domains finance \
  --question "What factors are currently driving the compression of credit spreads in IG corporate debt?"

# Risk modelling — combine with strategy
python main.py --domains finance strategy \
  --question "How should a pension fund rebalance its fixed-income allocation during a rate-hiking cycle?"
```

---

#### Cybersecurity (`cyber`)

**Best for:** threat analysis, adversarial machine learning, cryptography, incident response, zero-trust architecture.

```bash
# Current threats — MCP essential for up-to-date CVEs and TTPs
python main.py --domains cyber \
  --question "What are the most exploited attack vectors in cloud-native Kubernetes deployments in 2024?"

# Cryptographic proofs — disable MCP
python main.py --domains cyber --no-mcp \
  --question "Prove that RSA with PKCS#1 v1.5 padding is vulnerable to Bleichenbacher's oracle attack"

# Architecture review — combine with enterprise architecture
python main.py --domains cyber architecture \
  --question "Design a zero-trust network architecture for a hybrid cloud environment"
```

---

#### Biotech & Genomics (`biotech`)

**Best for:** CRISPR, protein folding, synthetic biology, genomic sequencing, gene therapy.

```bash
# Mechanism — knowledge-heavy, disable MCP
python main.py --domains biotech --no-mcp \
  --question "Explain the PAM-dependent and PAM-independent differences between Cas9 and Cas12a"

# Clinical pipeline — MCP useful
python main.py --domains biotech healthcare \
  --question "What is the current clinical status of base-editing therapies for sickle cell disease?"
```

---

#### Supply Chain (`supply_chain`)

**Best for:** logistics optimisation, demand forecasting, resilience analysis, procurement strategy, last-mile delivery.

```bash
# Operational optimisation — knowledge-heavy
python main.py --domains supply_chain --no-mcp \
  --question "Compare newsvendor model vs. base-stock policy for managing perishable inventory"

# Real-world disruption — MCP useful
python main.py --domains supply_chain \
  --question "What supply chain strategies have proven most resilient to geopolitical disruptions since 2020?"
```

---

### Enterprise problem agents

These agents have a balanced (0.55) base. Strategy, Legal, and Org Behaviour lean intuition-heavy in practice because analytical questions dominate.

---

#### Legal & Compliance (`legal`)

**Critical workflow rule:** Legal is a high-stakes domain; AUTO policy escalates to interactive.

```bash
# Regulatory analysis — keep MCP on for current regulations
python main.py --domains legal \
  --question "What are the key data residency obligations under EU GDPR for a US SaaS company?"

# Contract interpretation — knowledge-heavy
python main.py --domains legal --no-mcp \
  --question "Explain the material adverse change clause and when courts have upheld it"

# Compliance programme design — combine with org
python main.py --domains legal org \
  --question "Design a third-party vendor risk management programme for a Series B fintech"
```

---

#### Enterprise Architecture (`architecture`)

**Best for:** cloud-native design, microservices, event-driven architecture, technical debt, platform engineering.

```bash
# System design — knowledge-heavy
python main.py --domains architecture --no-mcp \
  --question "Design a multi-tenant SaaS platform with per-tenant data isolation on Kubernetes"

# Technology selection — MCP useful for current ecosystem
python main.py --domains architecture \
  --question "What are the current trade-offs between Kafka and Pulsar for high-throughput event streaming?"

# Migration strategy — combine with strategy
python main.py --domains architecture strategy \
  --question "Design a strangler-fig migration from a monolith to microservices for a 10-year-old codebase"
```

---

#### Marketing & Growth (`marketing`)

```bash
# Attribution modelling — knowledge-heavy
python main.py --domains marketing --no-mcp \
  --question "Compare multi-touch attribution models and explain when to use data-driven attribution"

# Current channel performance — MCP useful
python main.py --domains marketing \
  --question "What are the current CAC trends across paid and organic channels for B2B SaaS?"
```

---

#### Organisational Behaviour (`org`)

```bash
# Culture and change — interpretive, high intuition weight
python main.py --domains org --no-mcp \
  --question "What are the leading indicators that a post-merger integration will fail culturally?"

# Workforce planning — combine with strategy
python main.py --domains org strategy \
  --question "How should a 500-person company structure its org for a product-led growth motion?"
```

---

#### Strategy & Intelligence (`strategy`)

```bash
# Competitive analysis — knowledge-heavy
python main.py --domains strategy --no-mcp \
  --question "Apply Porter's Five Forces to the large language model infrastructure market"

# Market intelligence — MCP useful
python main.py --domains strategy \
  --question "What strategic moves are the hyperscalers making in the enterprise AI agent market?"
```

---

### Mastery, research, and experiment agents

---

#### Algorithms & Programming (`algo`)

**Best for:** Python, Rust, Go; all DS&A patterns; competitive programming; system programming.

```bash
# Algorithm derivation — disable MCP
python main.py --domains algo --no-mcp \
  --question "Implement and prove correctness of a segment tree with lazy propagation in Python"

# Language-specific — disable MCP
python main.py --domains algo --no-mcp \
  --question "Explain Rust's ownership model and how it prevents data races at compile time"

# Combine with CS for systems questions
python main.py --domains algo cs --no-mcp \
  --question "Implement a lock-free concurrent hash map and analyse its correctness"
```

**Token budget:** `--agent-max-tokens 1536` — implementations need room for code + explanation.

---

#### Interview Prep (`interview` / `faang`)

**Best for:** LeetCode patterns, FAANG system design, behavioural coaching, STAR framing.

**Optimal workflow — Socratic coaching:**
```bash
# Always use --interactive; the agent is designed to coach, not to give answers
python main.py --domains interview --interactive \
  --question "How do I solve the Trapping Rain Water problem?"
```

**Practice problem categories covered:**
- Arrays & Strings (easy/medium/hard)
- Linked Lists
- Trees & Binary Search Trees
- Graphs (BFS, DFS, topological sort, union-find)
- Dynamic Programming (all patterns: 0/1 knapsack, LCS, intervals, digit DP)
- Backtracking
- Heap / Priority Queue
- Binary Search
- Stack / Monotonic Stack
- Sliding Window
- Two Pointers
- System Design (full guided framework)
- Behavioural (STAR method)

**Token budget:** `--agent-max-tokens 1536`. Problems need: description + hint progression + pattern name.

**Workflow tip:** Route DS&A depth questions to `algo` and interview coaching to `interview`. Use both together for the deepest coverage:
```bash
python main.py --domains interview algo --no-mcp --interactive \
  --question "Design and implement a time-based key-value store with O(log n) retrieval"
```

---

#### EE LLM Research (`phd` / `ee_llm`)

**Best for:** frontier LLM research, signal processing ↔ LLM connections, LLM safety, mechanistic interpretability.

```bash
# PhD-level research synthesis — disable MCP for rigour
python main.py --domains phd --no-mcp \
  --question "Explain the mathematical duality between SSMs and linear attention transformers"

# Safety research — MCP useful for current papers
python main.py --domains phd \
  --question "What is the current state of the art in activation steering for LLM alignment?"

# Signal processing ↔ LLM — combine with signal
python main.py --domains phd signal --no-mcp \
  --question "How does Mamba's selective scan algorithm relate to adaptive IIR filter design?"
```

**Weight profile:** Intuition-heavy (0.65). This agent covers a very specific niche; knowledge-mode is almost always superior.

---

#### Experiment Runner (`experiment` / `simulate`)

**Best for:** converting quantitative/causal questions into structured experiment plans with runnable Python/NumPy snippets.

**Classification threshold:** Questions scoring ≥ 0.15 on the experimentability scale receive a full experiment plan. Questions below the threshold receive direct expert analysis.

**Optimal question forms for experiment plans:**
```bash
# Causal + quantitative → numeric sweep + perturbation
python main.py --domains experiment --non-interactive \
  --question "Does gradient descent converge faster with momentum on a quadratic loss surface?"

# Probabilistic → Monte Carlo
python main.py --domains experiment --non-interactive \
  --question "What is the probability of a random binary tree of depth 10 being balanced?"

# Optimisation → parametric sweep
python main.py --domains experiment --non-interactive \
  --question "What is the optimal learning rate for a ResNet-50 trained on CIFAR-10 with SGD?"
```

**Questions that route to direct analysis (no experiment plan):**
```bash
# Pure definition — score −0.40, below 0.15 threshold
python main.py --domains experiment --non-interactive \
  --question "What is the definition of backpropagation?"

# Historical fact — score −0.30
python main.py --domains experiment --non-interactive \
  --question "Who invented the perceptron?"
```

**Token budget:** `--agent-max-tokens 2048` — full experiment specs include Python snippets.

**Programmatic API:**
```python
from src.agents.experiment_runner_agent import ExperimentRunnerAgent

agent = ExperimentRunnerAgent()

# 1. Classify first (deterministic, no LLM needed)
result = ExperimentRunnerAgent.classify_question(
    "How does batch size affect gradient noise during training?"
)
print(result.is_experimentable)   # True
print(result.score)               # 0.55
print(result.question_type)       # "quantitative-causal"

# 2. Generate full plan
plan = agent.plan_experiments("How does batch size affect gradient noise?")
for spec in plan.experiments:
    print(spec.id, spec.category.value)  # e.g. "E1" "numeric_sweep"
    print(spec.hypothesis)
    print(spec.python_snippet)
```

---

## Routing questions to the right agent

### Decision tree

```
Is the question about a STEM topic?
  ├── Math/physics derivation → physics, signal, ee
  ├── Algorithm/code implementation → algo, cs
  ├── ML theory or architecture → nn, dl, phd
  └── Experimental/simulation → experiment

Is the question about an industry domain?
  ├── Medical / clinical → healthcare
  ├── Legal / regulatory → legal
  ├── Financial markets → finance
  ├── Security / adversarial → cyber
  ├── Biology / genetics → biotech
  └── Logistics / procurement → supply_chain

Is the question about an enterprise decision?
  ├── Technical platform / cloud → architecture
  ├── Go-to-market / growth → marketing
  ├── People / culture / org design → org
  └── Competitive position / M&A → strategy

Is the question about skill-building?
  ├── FAANG prep → interview (+ algo for depth)
  ├── Research paper / PhD question → phd
  └── DSP / filter design → signal (tutor mode)
```

### Multi-domain combinations that work well

| Combination | Good for |
|---|---|
| `nn dl` | Frontier architecture theory and recent LLM research |
| `algo cs` | Full-stack algorithms + systems context |
| `interview algo` | Deep FAANG coaching with algorithmic rigour |
| `phd signal` | SSM/LLM ↔ signal processing bridge |
| `healthcare biotech` | Drug discovery with clinical translation |
| `finance strategy` | Investment thesis + competitive analysis |
| `legal org` | Compliance programme design |
| `architecture strategy` | Platform-as-product decisions |
| `climate supply_chain` | Energy transition + critical materials |
| `social org` | Organisational psychology |

---

## Adding a new custom agent

### Step 1: Create the agent file

```python
# src/agents/my_domain_agent.py
from src.agents.base_agent import BaseAgent
from src.models import Domain

class MyDomainAgent(BaseAgent):
    """Expert in [description]."""

    domain = Domain.MY_DOMAIN  # must match the Domain enum

    def _build_system_prompt(self) -> str:
        return (
            "You are a world-class expert in [DOMAIN] with deep expertise in:\n"
            "- [Sub-field A]: [specific techniques/results]\n"
            "- [Sub-field B]: [specific techniques/results]\n"
            "\n"
            "=== [CORE THEORY] ===\n"
            "- [Named concept]: [detail]\n"
            "\n"
            "Ground answers in [evidence type]. "
            "Respond only with the requested JSON structure."
        )
```

### Step 2: Add the domain to the enum

```python
# src/models.py  (add to the Domain class)
class Domain(str, Enum):
    ...
    MY_DOMAIN = "my_domain"
```

### Step 3: Register in the orchestrator

```python
# src/orchestrator/agent_orchestrator.py
from src.agents.my_domain_agent import MyDomainAgent

_AGENT_CLASSES: dict[Domain, type[BaseAgent]] = {
    ...
    Domain.MY_DOMAIN: MyDomainAgent,
}
```

### Step 4: Add keyword mappings

```python
# src/intuition/human_intuition.py  (_DOMAIN_KEYWORDS dict)
Domain.MY_DOMAIN: [
    "keyword1", "keyword2", "keyword3",  # triggers domain selection
],
```

```python
# main.py  (_DOMAIN_MAP dict)
"myalias": Domain.MY_DOMAIN,
"my_domain": Domain.MY_DOMAIN,
```

### Step 5: Set the pipeline weight classification

In `src/agents/base_agent.py`, add the domain to the appropriate set:

```python
_INTUITION_HEAVY: frozenset[Domain] = frozenset({
    ...
    Domain.MY_DOMAIN,  # add here if knowledge-heavy
})
```

Or `_TOOL_HEAVY` if external evidence is critical.

### Step 6: Write a test

```python
# tests/test_agents.py (or a new test file)
def test_my_domain_agent_answers():
    from src.agents.my_domain_agent import MyDomainAgent
    agent = MyDomainAgent()
    resp = agent.answer("What is the core principle of [domain]?")
    assert resp.domain == Domain.MY_DOMAIN
    assert resp.answer
    assert 0.0 <= resp.confidence <= 1.0
```

---

## Common pitfalls and fixes

### Pitfall 1: LLM ignores the JSON format

**Symptom:** `_extract_json` returns `None`, answer falls back to `_mock_response`.

**Fix:** Add `"Respond only with the requested JSON structure."` as the **last sentence** of your system prompt. Some models also need explicit reinforcement in the user message; override `_build_user_message` to repeat the JSON schema.

---

### Pitfall 2: Very low confidence on all answers (0.3)

**Symptom:** All agent responses have `confidence=0.3` — this is the offline fallback value.

**Cause:** The backend is raising an exception silently and `_mock_response` is being returned.

**Fix:** Run with `--verbose` and check stderr for backend errors. For Ollama: ensure `ollama serve` is running and the model is pulled.

---

### Pitfall 3: MCP always returns empty context

**Symptom:** `mcp_context` is always `""`, tool pipeline never runs.

**Cause A:** `--no-mcp` flag is set. Remove it.  
**Cause B:** DuckDuckGo rate-limiting (common in CI). Add a retry delay or cache results.  
**Cause C:** Network is blocked. Use `--no-mcp` explicitly and accept knowledge-only mode.

---

### Pitfall 4: Iterative tutor gives a direct answer instead of checkpoints

**Symptom:** Physics or Signal Processing agent answers the question directly without the step-by-step protocol.

**Fix:** Ensure you are using `--interactive` mode. In non-interactive/auto-intuition mode the tutor protocol is bypassed because there is no human to interact with. The iterative protocol requires a human in the loop.

---

### Pitfall 5: Domain not selected for a relevant question

**Symptom:** A medical question does not trigger `healthcare` in auto-detect mode.

**Fix:** Check whether your question uses the healthcare keywords from `_DOMAIN_KEYWORDS`. Obscure medical terminology may not match. Two options:
1. Add synonyms to `_DOMAIN_KEYWORDS[Domain.HEALTHCARE]`.
2. Specify the domain explicitly: `--domains healthcare`.

---

### Pitfall 6: Agent answer is too short / truncated

**Symptom:** Answers end mid-sentence; JSON is malformed.

**Cause:** `max_tokens` is too low for the agent's verbosity.

**Fix:** Increase `--agent-max-tokens`. For iterative tutors (physics, signal) use at least 1024; for experiment runner use 2048.

---

## Solver Policy — strategy-selection toggle

The **solver policy** is a meta-level toggle that decides *which execution approach* to use for each question.  It lives in `src/solver/` and is wired into `main.py` via five new CLI flags.

### Available policies

| Policy | Behaviour |
|--------|-----------|
| `auto` | **(default)** Deterministic routing from question features + small ε-greedy exploration (ε=0.05). Exploration is disabled for high-stakes questions. |
| `baseline` | Deterministic routing only; no exploration. Preserves existing behaviour. |
| `explore` | Higher exploration rate (ε=0.30 by default). Useful for discovering stronger strategies or gathering performance data. |
| `fixed` | Always use the approach specified by `--solver-approach`. |

### Available approaches

| Approach | Execution strategy |
|----------|--------------------|
| `direct` | Standard non-adaptive pipeline (`AgentOrchestrator.run`). |
| `adaptive` | Force the adaptive agent-expansion loop (same as `--adaptive-agents`). |
| `debate` | Structured multi-party debate (`AgentOrchestrator.debate`) with the synthesised verdict surfaced as the primary answer. |
| `experiment` | Classify with `ExperimentRunnerAgent.classify_question`; if experimentable, generate an experiment plan; otherwise fall back to `direct`. |
| `portfolio` | Run `direct` + optionally `experiment` and/or extra debate agents; reconcile via `WeighingSystem`. |

### CLI flags

```
--solver-policy   POLICY   auto|baseline|explore|fixed   (default: auto)
--solver-approach APPROACH direct|adaptive|debate|experiment|portfolio (default: direct)
--explore-epsilon ε        Float in [0,1]; overrides per-policy default epsilon.
--explore-topk    K        Sample from top-K approaches during exploration (default: 3).
--no-explore-high-stakes   Disable exploration when high-stakes signals detected (default: on).
--explore-high-stakes      Allow exploration even on high-stakes questions.
```

### Examples

#### Default auto policy (recommended)

```bash
python main.py --question "How does gradient descent converge?"
# Solver policy: auto | approach: experiment  (or direct, depending on features)
```

#### Force the debate approach

```bash
python main.py --question "Pros and cons of microservices vs monolith?" \
               --solver-policy fixed --solver-approach debate
```

#### Force the portfolio approach (ensemble)

```bash
python main.py --question "How should I design this distributed system?" \
               --solver-policy fixed --solver-approach portfolio
```

#### Increase exploration to discover alternative approaches

```bash
python main.py --question "What optimisation method converges fastest?" \
               --solver-policy explore --explore-epsilon 0.50 --explore-topk 4
```

#### Baseline (preserve legacy behaviour)

```bash
python main.py --question "What is quantum entanglement?" \
               --solver-policy baseline
# Equivalent to omitting --solver-policy if the recommended approach is direct.
```

#### Allow exploration on high-stakes questions (disabled by default)

```bash
python main.py --question "What are the financial risks of this investment?" \
               --solver-policy explore --explore-high-stakes
```

### How routing works

1. **Feature extraction** (no LLM): length, high-stakes keywords, experiment signals, debate signals.
2. **Deterministic recommendation**:
   - experimentable *and* debate-worthy → `portfolio`
   - experimentable only → `experiment`
   - debate-worthy only → `debate`
   - long question (> 300 chars) → `adaptive`
   - otherwise → `direct`
3. **High-stakes gate**: if the question contains legal/medical/financial/security keywords and `--no-explore-high-stakes` is active, skip exploration.
4. **Epsilon-greedy draw**: with probability ε, sample from the top-K approaches (weighted by feature scores); otherwise use the recommendation.
5. **Dispatch**: the selected approach is passed to `AgentOrchestrator.run_solver`.

### Programmatic usage

```python
from src.solver import SolverPolicy, SolverApproach, StrategyRouter

router = StrategyRouter(
    policy=SolverPolicy.AUTO,
    auto_epsilon=0.05,
    no_explore_high_stakes=True,
)
selection = router.select("How does gradient descent converge?")
print(selection.approach)      # SolverApproach.EXPERIMENT
print(selection.explored)      # False (or True if ε-greedy fired)
print(selection.high_stakes_gate)  # False

# With an orchestrator
from src.orchestrator.agent_orchestrator import AgentOrchestrator
with AgentOrchestrator(use_mcp=False) as orch:
    result = orch.run_solver("How does gradient descent converge?", selection=selection)
    print(result.synthesized_answer)
```
