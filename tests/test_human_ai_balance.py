"""Tests for human-AI balance primitives."""
from __future__ import annotations
import pytest
from src.models import (
    AutonomyLevel, AUTONOMY_BASE_WEIGHT, ESCALATION_CONFIDENCE_THRESHOLD,
    HumanJudgment, EscalationDecision,
)


class TestAutonomyWeights:
    def test_full_auto_zero(self):
        assert AUTONOMY_BASE_WEIGHT[AutonomyLevel.FULL_AUTO] == 0.0

    def test_ai_proposes(self):
        assert AUTONOMY_BASE_WEIGHT[AutonomyLevel.AI_PROPOSES] == 0.20

    def test_ai_assists(self):
        assert AUTONOMY_BASE_WEIGHT[AutonomyLevel.AI_ASSISTS] == 0.50

    def test_human_first(self):
        assert AUTONOMY_BASE_WEIGHT[AutonomyLevel.HUMAN_FIRST] == 0.80


class TestHumanJudgment:
    def test_valid(self):
        j = HumanJudgment(context="test", judgment="approve", confidence=0.8)
        assert j.confidence == 0.8
        assert j.override is False

    def test_confidence_bounds(self):
        with pytest.raises(ValueError):
            HumanJudgment(context="x", judgment="y", confidence=1.5)

    def test_override_flag(self):
        j = HumanJudgment(context="x", judgment="y", override=True)
        assert j.override is True


class TestEscalationDecision:
    def test_no_escalation(self):
        e = EscalationDecision(needs_escalation=False, reason="")
        assert e.needs_escalation is False
        assert e.urgency == "review"

    def test_escalation(self):
        e = EscalationDecision(needs_escalation=True, reason="P1 incident", urgency="immediate")
        assert e.needs_escalation is True
        assert e.urgency == "immediate"


class TestEscalationThreshold:
    def test_threshold_value(self):
        assert 0.0 < ESCALATION_CONFIDENCE_THRESHOLD < 1.0
        assert ESCALATION_CONFIDENCE_THRESHOLD == 0.55
