# Intuition Scientist 🧪🧠

A **Human Intuition Scientist** that leverages an agent orchestrator and a suite of domain-expert agents to systematically compare human intuitive answers against machine-generated expert knowledge — then weighs them against each other with deep analytical insight.

---

## What it does

1. **Captures human intuition** — asks you for your intuitive answer to a question, plus your confidence and reasoning.
2. **Routes the question** to the most relevant domain-expert agents (auto-detected from the question text).
3. **Queries all relevant agents in parallel** — each specialist searches the internet via the MCP client and then calls an LLM for a structured expert answer.
4. **Weighs intuition against expert consensus** — uses semantic similarity, confidence-weighted scoring, and (optionally) an LLM to produce a rich alignment analysis.
5. **Synthesizes a final answer** — blends expert knowledge with the human intuition in proportion to how well they aligned.

---

## Domain Agents

| Agent | Expertise |
|---|---|
| **Electrical Engineering** | Circuits, electromagnetics, power systems, RF |
| **Computer Science** | Algorithms, OS, compilers, networking, security |
| **Neural Networks** | Architectures (CNN/RNN/Transformer), training theory |
| **Social Science** | Sociology, psychology, behavioural economics, political science |
| **Space Science** | Astrophysics, cosmology, orbital mechanics, astrobiology |
| **Physics** | Classical & quantum mechanics, relativity, thermodynamics |
| **Deep Learning** | LLMs, fine-tuning, diffusion models, scaling laws |

---

## Architecture

```
main.py (CLI)
  └─ AgentOrchestrator  ← brain
       ├─ IntuitionCapture     ← captures human intuition
       ├─ Domain Agents (×7)   ← parallel expert queries
       │    ├─ MCPClient       ← internet search (DuckDuckGo)
       │    └─ LLM (Anthropic / OpenAI)
       └─ WeighingSystem       ← deep analysis & synthesis
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your LLM API key

```bash
# Anthropic (default)
export ANTHROPIC_API_KEY="sk-ant-..."

# or OpenAI
export OPENAI_API_KEY="sk-..."
```

> **Offline / mock mode** — if no key is set, all agents run in mock mode.
> The system still exercises the full pipeline with placeholder answers, which is
> useful for development and testing.

---

## Usage

### Interactive (fully guided)

```bash
python main.py
```

### Supply the question on the command line

```bash
python main.py --question "How does quantum tunnelling work in semiconductors?"
```

### Restrict to specific domains

```bash
python main.py -q "What is entropy?" --domains physics ee
```

Available domain shortcuts: `ee`, `cs`, `nn`, `social`, `space`, `physics`, `dl`

### Use OpenAI instead of Anthropic

```bash
python main.py --provider openai --model gpt-4o
```

### Disable internet search (MCP off)

```bash
python main.py --no-mcp
```

---

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```

All tests run fully offline — no API key required.

---

## Project structure

```
intuition-scientist/
├── main.py                          # CLI entry point
├── requirements.txt
├── src/
│   ├── models.py                    # Shared data models
│   ├── agents/
│   │   ├── base_agent.py            # Abstract base for all domain agents
│   │   ├── electrical_engineering_agent.py
│   │   ├── computer_science_agent.py
│   │   ├── neural_networks_agent.py
│   │   ├── social_science_agent.py
│   │   ├── space_science_agent.py
│   │   ├── physics_agent.py
│   │   └── deep_learning_agent.py
│   ├── orchestrator/
│   │   └── agent_orchestrator.py    # Brain: coordinates all components
│   ├── intuition/
│   │   └── human_intuition.py       # Captures & structures human intuition
│   ├── mcp/
│   │   └── mcp_client.py            # MCP internet-search client
│   └── analysis/
│       └── weighing_system.py       # Intuition vs. expert weighing & analysis
└── tests/
    ├── test_models.py
    ├── test_agents.py
    ├── test_intuition.py
    ├── test_weighing_system.py
    └── test_orchestrator.py
```
