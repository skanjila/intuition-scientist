# Use Case 16: Mental Health Triage

## Overview
Mental health crisis triage. **Always escalates** — clinician must review before any intervention.

## Medical Safety Notice
> **⚠ AI-generated analysis only.** Always requires validation by a licensed healthcare professional before any clinical decision.

## Method
```python
orch.triage_mental_health(presenting_concerns, *, autonomy=AutonomyLevel.HUMAN_FIRST)
```

## Agents Used
- `MentalHealthTriageAgent`

## Escalation
**Always escalates.** Urgency: `immediate` for crisis keywords (suicide, self harm, overdose). Resources: 988 Lifeline, Crisis Text Line.

## Example
```python
result = orch.triage_mental_health('I want to hurt myself')
print(result.risk_level)  # 'high'
print(result.escalation.urgency)  # 'immediate'
```

## CLI
```bash
python main.py mental --concerns 'feeling hopeless and having panic attacks'
```
