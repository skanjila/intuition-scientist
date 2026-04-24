# Use Case 14: Healthcare Gap Analysis

## Overview
Identify healthcare access gaps for a region or population and recommend interventions.

## Medical Safety Notice
> **⚠ AI-generated analysis only.** Always requires validation by a licensed healthcare professional before any clinical decision.

## Method
```python
orch.analyze_healthcare_gaps(region_or_population, *, autonomy=AutonomyLevel.AI_ASSISTS)
```

## Agents Used
- `HealthcareAccessAgent`\n- `PatientRiskAgent`

## Escalation
Escalates if AI confidence below 0.35.

## Example
```python
result = orch.analyze_healthcare_gaps('rural Appalachia')
print(result.identified_gaps)
```

## CLI
```bash
python main.py gaps --population 'rural Appalachia'
```
