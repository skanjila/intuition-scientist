# Use Case 13: Patient Risk Stratification

## Overview
Stratify patient readmission and complication risk based on diagnoses, age, medications, and social determinants.

## Medical Safety Notice
> **⚠ AI-generated analysis only.** Always requires validation by a licensed healthcare professional before any clinical decision.

## Method
```python
orch.stratify_patient_risk(patient, *, autonomy=AutonomyLevel.AI_PROPOSES)
```

## Agents Used
- `PatientRiskAgent`\n- `ClinicalDecisionSupportAgent`

## Escalation
Escalates for high-risk patients (age > 65 with 3+ diagnoses). Urgency: `review`.

## Example
```python
from src.models import PatientRiskInput
p = PatientRiskInput('P001', age=75, diagnoses=['diabetes', 'hypertension', 'CKD'])
result = orch.stratify_patient_risk(p)
print(result.risk_level)  # 'high'
```

## CLI
```bash
python main.py risk --patient-id P001 --age 75 --diagnoses 'diabetes,hypertension'
```
