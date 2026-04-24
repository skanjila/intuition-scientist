# Use Case 00: Supply Chain Exception

## Overview
Handle supply chain exceptions (late delivery, substitution, stockout) with financial impact estimates.

## Method
```python
orch.handle_exception(event, *, autonomy=AutonomyLevel.AI_PROPOSES)
```

## Agents Used
- `SupplyChainAgent`\n- `FinanceEconomicsAgent`

## Escalation Conditions
- Keywords: critical, stockout, production halt\n- AI confidence below 0.55

## CLI
```bash
python main.py exception --sku SKU-1 --supplier SupplierA --days-late 10
```
