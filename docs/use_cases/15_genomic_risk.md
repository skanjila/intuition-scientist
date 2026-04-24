# Use Case 15: Genomic Risk Assessment

## Overview
Assess genomic risk from variant data. **Always escalates** — genetic counselor required before disclosure.

## Medical Safety Notice
> **⚠ AI-generated analysis only.** Always requires validation by a licensed healthcare professional before any clinical decision.

## Method
```python
orch.assess_genomic_risk(sample_id, variants, patient_context='', *, autonomy=AutonomyLevel.HUMAN_FIRST)
```

## Agents Used
- `GenomicsMedicineAgent`

## Escalation
**Always escalates.** Genetic counselor consult required before results disclosed.

## Example
```python
result = orch.assess_genomic_risk('S001', ['BRCA1:c.5266dupC'])
assert result.genetic_counseling_needed  # Always True
```

## CLI
```bash
python main.py genomics --sample-id S001 --variants 'BRCA1:c.5266dupC'
```
