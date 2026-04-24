# Use Case 04: Finance Reconciliation

## Overview
Match ledger entries against invoices, identify variances, and generate audit narratives.

## Method
```python
orch.reconcile(ledger, invoices, *, materiality_threshold=1000.0)
```

## Agents Used
- `FinanceReconciliationAgent`\n- `FinanceEconomicsAgent`

## Escalation Conditions
- Keywords: fraud, error, variance, unmatched, discrepancy

## CLI
```bash
python main.py reconcile
```
