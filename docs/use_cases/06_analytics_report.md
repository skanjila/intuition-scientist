# Use Case 06: Analytics Report

## Overview
Generate analytics and KPI reports, detect anomalies, and surface insights.

## Method
```python
orch.generate_report(context, *, autonomy=AutonomyLevel.FULL_AUTO)
```

## Agents Used
- `AnalyticsAgent`\n- `FinanceEconomicsAgent`

## Escalation Conditions
- KPI anomaly > 20% vs target

## CLI
```bash
python main.py report --metrics '{"revenue": 1000000}'
```
