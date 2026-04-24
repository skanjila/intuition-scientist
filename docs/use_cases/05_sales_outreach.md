# Use Case 05: Sales Outreach

## Overview
Research a target company and draft personalized sales outreach emails.

## Method
```python
orch.outreach(company, product='', deal_value=0.0)
```

## Agents Used
- `MarketingGrowthAgent`\n- `StrategyIntelligenceAgent`

## Escalation Conditions
- Deal value > $500K → VP Sales approval required

## CLI
```bash
python main.py outreach --company 'Acme Corp' --product 'DataPlatform'
```
