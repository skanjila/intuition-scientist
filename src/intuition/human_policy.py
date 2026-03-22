"""Human Involvement Policy — decides how much user interaction is required.

This module provides a lightweight, testable policy that determines whether
the system should prompt the human for more detailed intuition input.

Policy levels
-------------
``HumanPolicy.AUTO``
    Prompt only when the system detects a situation that genuinely benefits
    from deeper human judgment (escalation triggers).  This is the **default**.

``HumanPolicy.ALWAYS``
    Always prompt interactively, regardless of question type or confidence.
    Equivalent to the legacy fully-interactive mode.

``HumanPolicy.NEVER``
    Never prompt; always use auto-generated intuition.
    Useful for CI, scripting, or batch runs.

Escalation triggers (used in AUTO mode)
----------------------------------------
The system escalates to interactive prompting when *any* of the following
conditions are met (unless the policy is NEVER):

1. **High-stakes domain**: the question is inferred to belong to a healthcare,
   legal, or finance domain — mistakes in these areas can have serious
   real-world consequences.
2. **Low overall confidence**: the mean agent confidence across all responses
   falls below ``LOW_CONFIDENCE_THRESHOLD`` (0.40).
3. **High disagreement**: the spread (max − min) of agent confidence scores
   exceeds ``HIGH_DISAGREEMENT_THRESHOLD`` (0.45), indicating agents strongly
   disagree.
4. **MCP expected but absent**: one or more tool-heavy domains were queried
   but MCP returned no results for *any* of them.

These thresholds are intentionally conservative (designed not to escalate on
normal questions) and can be tuned without changing the public interface.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from src.models import AgentResponse, Domain

# ---------------------------------------------------------------------------
# High-stakes domains — escalate to human review when these are present
# ---------------------------------------------------------------------------

_HIGH_STAKES_DOMAINS: frozenset[Domain] = frozenset(
    {
        Domain.HEALTHCARE,
        Domain.LEGAL_COMPLIANCE,
        Domain.FINANCE_ECONOMICS,
    }
)

# ---------------------------------------------------------------------------
# Domains where MCP/tool evidence is especially important
# ---------------------------------------------------------------------------

_TOOL_HEAVY_DOMAINS: frozenset[Domain] = frozenset(
    {
        Domain.HEALTHCARE,
        Domain.LEGAL_COMPLIANCE,
        Domain.FINANCE_ECONOMICS,
        Domain.CYBERSECURITY,
        Domain.SUPPLY_CHAIN,
        Domain.CLIMATE_ENERGY,
    }
)

# ---------------------------------------------------------------------------
# Thresholds (module-level constants for easy testing/tuning)
# ---------------------------------------------------------------------------

#: Mean agent confidence below this value triggers escalation
LOW_CONFIDENCE_THRESHOLD: float = 0.40

#: (max − min) confidence spread above this value triggers escalation
HIGH_DISAGREEMENT_THRESHOLD: float = 0.45


# ---------------------------------------------------------------------------
# Policy enum
# ---------------------------------------------------------------------------


class HumanPolicy(str, Enum):
    """Controls how much interactive human input the system requires."""

    AUTO = "auto"
    """Prompt only on escalation triggers (default)."""

    ALWAYS = "always"
    """Always prompt interactively."""

    NEVER = "never"
    """Never prompt; always use auto-generated intuition."""


# ---------------------------------------------------------------------------
# Escalation-trigger helpers
# ---------------------------------------------------------------------------


def has_high_stakes_domain(domains: list[Domain]) -> bool:
    """Return True if any domain in *domains* is considered high-stakes."""
    return any(d in _HIGH_STAKES_DOMAINS for d in domains)


def has_low_confidence(responses: list[AgentResponse]) -> bool:
    """Return True if mean agent confidence is below the low-confidence threshold."""
    if not responses:
        return False
    mean_conf = sum(r.confidence for r in responses) / len(responses)
    return mean_conf < LOW_CONFIDENCE_THRESHOLD


def has_high_disagreement(responses: list[AgentResponse]) -> bool:
    """Return True if the confidence spread across agents exceeds the threshold.

    A wide spread indicates that agents strongly disagree about the answer,
    which is a signal that human judgment may help break the tie.
    """
    if len(responses) < 2:
        return False
    confs = [r.confidence for r in responses]
    spread = max(confs) - min(confs)
    return spread > HIGH_DISAGREEMENT_THRESHOLD


def has_missing_mcp_for_tool_domains(
    responses: list[AgentResponse],
    use_mcp: bool,
) -> bool:
    """Return True when MCP was enabled but returned no results for tool-heavy domains.

    Parameters
    ----------
    responses:
        The agent responses to inspect.
    use_mcp:
        Whether MCP was enabled for this run.  If MCP is disabled, this
        trigger is never raised (it would be a false alarm).
    """
    if not use_mcp:
        return False
    tool_domain_responses = [r for r in responses if r.domain in _TOOL_HEAVY_DOMAINS]
    if not tool_domain_responses:
        return False
    # If every tool-heavy domain response has an empty mcp_context, MCP
    # returned nothing useful for tool-heavy questions.
    return all(not r.mcp_context for r in tool_domain_responses)


# ---------------------------------------------------------------------------
# Main policy decision function
# ---------------------------------------------------------------------------


def should_escalate(
    domains: list[Domain],
    responses: Optional[list[AgentResponse]] = None,
    use_mcp: bool = True,
) -> bool:
    """Return True if the system should prompt for deeper human input.

    This function evaluates all escalation triggers in order and returns
    ``True`` as soon as any one of them fires.

    Parameters
    ----------
    domains:
        The list of domain agents selected for this question.
    responses:
        Agent responses collected so far.  May be ``None`` or empty if called
        before agents have run (in that case only the domain-level trigger
        can fire).
    use_mcp:
        Whether MCP internet search is enabled for this run.
    """
    responses = responses or []

    if has_high_stakes_domain(domains):
        return True
    if has_low_confidence(responses):
        return True
    if has_high_disagreement(responses):
        return True
    if has_missing_mcp_for_tool_domains(responses, use_mcp=use_mcp):
        return True
    return False


def decide_interactive(
    policy: HumanPolicy,
    domains: list[Domain],
    responses: Optional[list[AgentResponse]] = None,
    use_mcp: bool = True,
) -> bool:
    """Return True if interactive human-intuition capture should be used.

    Parameters
    ----------
    policy:
        The user-selected (or default) ``HumanPolicy``.
    domains:
        Domains inferred for the current question.
    responses:
        Agent responses (optional; used for confidence/disagreement checks).
    use_mcp:
        Whether MCP is enabled.
    """
    if policy == HumanPolicy.ALWAYS:
        return True
    if policy == HumanPolicy.NEVER:
        return False
    # AUTO: escalate only when a trigger fires
    return should_escalate(domains, responses=responses, use_mcp=use_mcp)
