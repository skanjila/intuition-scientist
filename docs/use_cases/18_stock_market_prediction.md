# Use Case 18: Stock Market Prediction

## Overview
Multi-agent stock analysis combining technical, fundamental, and macro perspectives. **Not investment advice.**

## Medical Safety Notice
> **⚠ This is not investment advice.** AI analysis is speculative. Consult a licensed financial advisor before making investment decisions.

## Method
```python
orch.predict_stock(stock_input, *, human_judgment=None, autonomy=AutonomyLevel.AI_ASSISTS)
```

## Agents Used
- `StockMarketAgent`\n- `FinanceEconomicsAgent`\n- `StrategyIntelligenceAgent`

## Escalation
Escalates if confidence below 0.55 or event keywords detected (earnings, FDA, merger, delisting).

## Example
```python
from src.models import StockPredictionInput, HumanJudgment
inp = StockPredictionInput('NVDA', horizon='1m', human_thesis='Strong GPU demand')
result = orch.predict_stock(inp)
print(result.direction, result.confidence_pct)
print(result.disclaimer)
```

## CLI
```bash
python main.py stock --ticker NVDA --horizon 1m --headline 'Record GPU shipments'
```
