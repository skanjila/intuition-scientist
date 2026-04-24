# Use Case 12: Medical Literature Synthesis

## Overview
Synthesize medical literature for a clinical query, summarizing key findings and evidence quality.

## Medical Safety Notice
> **⚠ AI-generated analysis only.** Always requires validation by a licensed healthcare professional before any clinical decision.

## Method
```python
orch.synthesize_literature(query, *, autonomy=AutonomyLevel.AI_ASSISTS)
```

## Agents Used
- `MedicalLiteratureAgent`

## Escalation
Escalates if AI confidence below 0.40.

## Example
```python
result = orch.synthesize_literature('GLP-1 agonists cardiovascular outcomes')
print(result.synthesis[:200])
```

## CLI
```bash
python main.py literature --query 'GLP-1 agonists cardiovascular outcomes'
```
