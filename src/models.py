"""Shared data models for the Human Intuition Scientist."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Domain(str, Enum):
    """Scientific / engineering domains covered by the system."""

    ELECTRICAL_ENGINEERING = "electrical_engineering"
    COMPUTER_SCIENCE = "computer_science"
    NEURAL_NETWORKS = "neural_networks"
    SOCIAL_SCIENCE = "social_science"
    SPACE_SCIENCE = "space_science"
    PHYSICS = "physics"
    DEEP_LEARNING = "deep_learning"


@dataclass
class HumanIntuition:
    """Structured representation of a human's intuitive answer."""

    question: str
    intuitive_answer: str
    # 0.0 (wild guess) → 1.0 (highly confident)
    confidence: float
    reasoning: str = ""
    domain_guesses: list[Domain] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")


@dataclass
class AgentResponse:
    """A domain-specific agent's answer to the question."""

    domain: Domain
    answer: str
    reasoning: str
    confidence: float
    sources: list[str] = field(default_factory=list)
    mcp_context: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")


@dataclass
class AlignmentScore:
    """Semantic alignment between the human intuition and one agent response."""

    domain: Domain
    # 0.0 = completely divergent, 1.0 = perfect alignment
    semantic_similarity: float
    key_agreements: list[str] = field(default_factory=list)
    key_divergences: list[str] = field(default_factory=list)
    intuition_insight: str = ""


@dataclass
class WeighingResult:
    """Full cross-agent weighing of human intuition vs. expert answers."""

    question: str
    human_intuition: HumanIntuition
    agent_responses: list[AgentResponse]
    alignment_scores: list[AlignmentScore]
    # Weighted blend of human intuition and agent consensus
    synthesized_answer: str
    # How well the human's intuition held up overall (0-100 %)
    intuition_accuracy_pct: float
    overall_analysis: str
    recommendations: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """A single web-search result returned by the MCP client."""

    title: str
    url: str
    snippet: str
    relevance_score: Optional[float] = None
