# Use Case 10: Clinical Decision Support

## Overview
AI-assisted clinical decision support. **Always escalates** — physician review required before any clinical action.

## Medical Safety Notice
> **⚠ AI-generated analysis only.** Always requires validation by a licensed healthcare professional before any clinical decision.

## Method
```python
orch.clinical_decision(assessment, *, human_judgment=None, autonomy=AutonomyLevel.HUMAN_FIRST)
```

## Agents Used
- `ClinicalDecisionSupportAgent`\n- `PatientRiskAgent`

## Escalation
**Always escalates.** Urgency: `immediate` for red flag symptoms (chest pain, stroke, sepsis, anaphylaxis, suicidal ideation, overdose).

## Example
```python
from src.models import ClinicalAssessmentInput
result = orch.clinical_decision(
    ClinicalAssessmentInput('45F', symptoms=['chest pain', 'diaphoresis'])
)
assert result.escalation.needs_escalation  # Always True
```

## CLI
```bash
python main.py clinical --symptoms 'chest pain, shortness of breath'
```
