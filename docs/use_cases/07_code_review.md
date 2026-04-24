# Use Case 07: Code Review

## Overview
Review code diffs for bugs, security vulnerabilities, and best practice violations.

## Method
```python
orch.review_pr(diff, description='', *, autonomy=AutonomyLevel.AI_PROPOSES)
```

## Agents Used
- `CodeReviewAgent`\n- `CybersecurityAgent`

## Escalation Conditions
- Security keywords: sql injection, xss, hardcoded secret\n- Risk score > 0.6

## CLI
```bash
python main.py review --diff 'def login(u,p): return db.raw(u,p)'
```
