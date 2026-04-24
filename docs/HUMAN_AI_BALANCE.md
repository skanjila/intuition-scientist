# Human-AI Balance Framework

This document explains the human-AI collaboration model used across all 18 use cases.

---

## Core Principle

No AI system should operate without appropriate human oversight, especially in high-stakes domains like healthcare, finance, and legal compliance. This platform implements a **configurable autonomy spectrum** that allows organizations to set appropriate human oversight levels for each use case.

---

## Autonomy Levels

```python
class AutonomyLevel(str, Enum):
    FULL_AUTO   = "full_auto"    # AI acts without human review
    AI_PROPOSES = "ai_proposes"  # AI drafts, human approves
    AI_ASSISTS  = "ai_assists"   # AI and human collaborate equally
    HUMAN_FIRST = "human_first"  # Human decides, AI provides analysis
```

### Weight Table

| Level | Human Weight | Typical Use Cases |
|-------|-------------|-------------------|
| `FULL_AUTO` | 0% | Internal analytics, routine reports |
| `AI_PROPOSES` | 20% | Ticket triage, incident response, code review |
| `AI_ASSISTS` | 50% | Compliance Q&A, sales outreach, RFP drafts |
| `HUMAN_FIRST` | 80% | Clinical decisions, genomics, mental health |

---

## Blending Algorithm

When a `HumanJudgment` is provided alongside an AI response:

```python
def _blend(ai_text, human_judgment, autonomy, ai_confidence):
    if human_judgment.override:
        # Human completely overrides AI
        return human_judgment.judgment, 1.0
    
    # Compute effective human weight
    hw = AUTONOMY_BASE_WEIGHT[autonomy] * human_judgment.confidence
    
    if hw >= 0.5:
        # Human judgment leads, AI is supplementary
        return f"{human_judgment.judgment}\n\n[AI analysis ({1-hw:.0%} weight): {ai_text[:400]}]"
    else:
        # AI leads, human judgment is noted
        return f"{ai_text}\n\n[Human judgment ({hw:.0%} weight): {human_judgment.judgment[:200]}]"
```

### Example: AI_ASSISTS + Human at 80% confidence

```
effective_human_weight = 0.50 × 0.80 = 0.40  (40%)
→ AI leads (60%), human noted at 40%
```

### Example: HUMAN_FIRST + Human at 80% confidence

```
effective_human_weight = 0.80 × 0.80 = 0.64  (64%)
→ Human leads (64%), AI is supplementary
```

---

## HumanJudgment Dataclass

```python
@dataclass
class HumanJudgment:
    context: str           # What the judgment is about
    judgment: str          # The human's assessment/decision
    confidence: float      # 0.0 – 1.0 (validated)
    override: bool = False # True = completely bypass AI output
    reviewer_id: str = ""  # Audit trail
    timestamp: str = ""    # When judgment was made
```

### Usage

```python
from src.models import HumanJudgment, AutonomyLevel
from src.orchestrator.business_orchestrator import BusinessOrchestrator

orch = BusinessOrchestrator(backend=MockBackend())

# Provide human expert judgment
human = HumanJudgment(
    context="Ticket from enterprise customer",
    judgment="This is a billing dispute, route to Finance",
    confidence=0.9
)

result = orch.triage(
    "I've been charged twice",
    human_judgment=human,
    autonomy=AutonomyLevel.AI_ASSISTS
)
```

---

## Escalation Framework

Every result includes an `EscalationDecision`:

```python
@dataclass
class EscalationDecision:
    needs_escalation: bool
    reason: str
    urgency: str = "review"      # "informational" | "review" | "immediate"
    checkpoint: str = ""          # Required human checkpoint description
```

### Escalation Triggers

1. **Low AI confidence** — confidence below domain threshold (default: 0.55)
2. **Keyword detection** — domain-specific escalation terms (e.g., "breach", "lawsuit")
3. **Domain rule** — always escalate (medical, genomics, mental health)
4. **Business rule** — deal value > $500K, KPI anomaly > 20%, etc.

### Domain Thresholds

| Domain | Confidence Threshold | Always Escalate |
|--------|---------------------|-----------------|
| Triage | 0.55 | No |
| Compliance | 0.60 | No |
| Incident | 0.55 | No |
| Clinical Decision | — | **Yes** |
| Drug Interactions | — | **Yes** |
| Genomics | — | **Yes** |
| Mental Health | — | **Yes** |
| Clinical Trials | — | **Yes** |

---

## Audit Trail

All result dataclasses include:
- `human_judgment` — the `HumanJudgment` used (if any)
- `autonomy_used` — the `AutonomyLevel` applied
- `ai_confidence` — raw AI confidence score (0.0–1.0)
- `escalation` — the `EscalationDecision`

This provides a complete audit trail for compliance and review purposes.

---

## Medical Safety Principle

All medical use cases enforce `HUMAN_FIRST` or always-escalate semantics:

> **AI-generated analysis only. Always requires validation by a licensed healthcare professional before any clinical decision.**

This disclaimer is enforced by the `GuardrailEngine` and embedded in all medical result outputs.

---

## Configuring Per-Use-Case Autonomy

```python
from src.models import AutonomyLevel
from src.orchestrator.business_orchestrator import BusinessOrchestrator

orch = BusinessOrchestrator(backend=MockBackend())

# Override default autonomy for any use case
result = orch.triage(
    ticket_text="Customer complaint",
    autonomy=AutonomyLevel.HUMAN_FIRST  # Override default AI_PROPOSES
)
```

---

## Testing Human-AI Balance

```python
# Test override behavior
j = HumanJudgment(context="test", judgment="This is P1", override=True)
result = orch.triage("ticket", human_judgment=j)
assert "This is P1" in result.draft_response  # Human wins completely

# Test weight constants
from src.models import AUTONOMY_BASE_WEIGHT
assert AUTONOMY_BASE_WEIGHT[AutonomyLevel.FULL_AUTO] == 0.0
assert AUTONOMY_BASE_WEIGHT[AutonomyLevel.HUMAN_FIRST] == 0.80
```
