# Use Case 11: Drug Interaction Check

## Overview
Check drug-drug and drug-patient interactions. **Always escalates** — pharmacist must validate before dispensing.

## Medical Safety Notice
> **⚠ AI-generated analysis only.** Always requires validation by a licensed healthcare professional before any clinical decision.

## Method
```python
orch.check_drug_interactions(medications, patient_context='', *, autonomy=AutonomyLevel.AI_PROPOSES)
```

## Agents Used
- `DrugInteractionAgent`

## Escalation
**Always escalates.** Urgency: `review`. Checkpoint: Pharmacist must validate before dispensing.

## Example
```python
result = orch.check_drug_interactions(['warfarin', 'aspirin', 'ibuprofen'])
assert result.escalation.needs_escalation  # Always True
```

## CLI
```bash
python main.py drugs --medications 'warfarin,aspirin,ibuprofen'
```
