# Use Case 01: Customer Support Triage

## Overview
Automatically triage incoming customer support tickets, determine urgency (P1–P4), route to the correct department, and draft an initial response.

## Method
```python
orch.triage(ticket_text, *, human_judgment=None, autonomy=AutonomyLevel.AI_PROPOSES)
```

## Agents Used
- `CustomerSupportAgent` — primary ticket analysis
- `OrganizationalBehaviorAgent` — routing logic

## Input
| Field | Type | Description |
|-------|------|-------------|
| `ticket_text` | `str` | Raw ticket content |
| `human_judgment` | `HumanJudgment?` | Optional human override |
| `autonomy` | `AutonomyLevel` | Default: `AI_PROPOSES` |

## Output: `TriageResult`
| Field | Type | Description |
|-------|------|-------------|
| `urgency` | `str` | P1/P2/P3/P4 |
| `routing_department` | `str` | Engineering, Support, Finance, etc. |
| `draft_response` | `str` | AI-drafted reply to customer |
| `escalation` | `EscalationDecision` | Whether human review needed |
| `ai_confidence` | `float` | 0.0–1.0 |

## P1 Detection
Automatic P1 assignment when ticket contains: `outage`, `down`, `critical`, `all users`, `revenue`.

## Example
```python
result = orch.triage("Payment API returning 500 for all enterprise users")
# result.urgency == "P1"
# result.routing_department == "Engineering"
# result.escalation.needs_escalation == True
```

## CLI
```bash
python main.py triage --ticket "Payment API is down for all users"
```

## REST
```bash
curl -X POST /triage -d '{"ticket_text": "API down"}'
```
