# Use Case 17: Clinical Trial Matching

## Overview
Match patients to eligible clinical trials based on diagnoses, age, and condition. **Always escalates** — physician must confirm eligibility.

## Medical Safety Notice
> **⚠ AI-generated analysis only.** Always requires validation by a licensed healthcare professional before any clinical decision.

## Method
```python
orch.match_clinical_trials(patient, condition='', *, autonomy=AutonomyLevel.AI_PROPOSES)
```

## Agents Used
- `ClinicalTrialsAgent`

## Escalation
**Always escalates.** Physician must confirm eligibility before enrollment.

## Example
```python
from src.models import PatientRiskInput
p = PatientRiskInput('P003', age=45, diagnoses=['breast cancer'])
result = orch.match_clinical_trials(p, 'HER2+ breast cancer')
for trial in result.matched_trials:
    print(trial.trial_id, trial.eligibility_match)
```

## CLI
```bash
python main.py trials --patient-id P003 --age 45 --diagnoses 'breast cancer'
```
