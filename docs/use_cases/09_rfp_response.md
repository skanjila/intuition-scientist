# Use Case 00: RFP Response

## Overview
Draft comprehensive RFP responses including risk flag identification and section drafts.

## Method
```python
orch.draft_rfp(rfp_text, rfp_title='', *, autonomy=AutonomyLevel.AI_ASSISTS)
```

## Agents Used
- `RFPAgent`\n- `StrategyIntelligenceAgent`\n- `LegalComplianceAgent`

## Escalation Conditions
- Keywords: unlimited liability, indemnification, liquidated damages\n- AI confidence below 0.65

## CLI
```bash
python main.py rfp --title 'Cloud RFP' --text 'Vendor must provide 99.9% uptime'
```
