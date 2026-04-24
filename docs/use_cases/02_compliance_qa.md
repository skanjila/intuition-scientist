# Use Case 02: Compliance Q&A

## Overview
Answer compliance and regulatory questions with citations, escalating when legal risk is detected.

## Method
```python
orch.compliance_qa(question, *, human_judgment=None, autonomy=AutonomyLevel.AI_ASSISTS)
```

## Agents Used
- `LegalComplianceAgent`\n- `OrganizationalBehaviorAgent`

## Escalation Conditions
- AI confidence below 0.60\n- Keywords: penalty, lawsuit, violation, criminal

## CLI
```bash
python main.py compliance --question 'Do we need consent for email marketing?'
```
