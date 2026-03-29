"""Solver policy and approach enumerations."""

from __future__ import annotations

from enum import Enum


class SolverPolicy(str, Enum):
    """Controls how the :class:`~src.solver.router.StrategyRouter` selects an approach.

    auto
        Default.  Route deterministically based on question features, but
        apply a small amount of epsilon-greedy exploration (ε=0.05 by default)
        so that non-dominant approaches are occasionally tried.  Exploration is
        disabled automatically when high-stakes signals are detected.
    baseline
        Preserve existing behaviour — equivalent to the ``direct`` approach
        with the ``--adaptive-agents`` flag honoured as-is.  The router never
        deviates from the recommended (deterministic) path.
    explore
        Increase exploration.  The router uses the caller-supplied
        ``explore_epsilon`` (or a module-level default of 0.30 if not set) so
        that a wider variety of approaches are exercised — useful for
        collecting performance data or discovering stronger strategies.
    fixed
        Always use the approach specified by ``--solver-approach``.  The
        router never explores and ignores all question features.
    """

    AUTO = "auto"
    BASELINE = "baseline"
    EXPLORE = "explore"
    FIXED = "fixed"


class SolverApproach(str, Enum):
    """The concrete execution strategy to run for a question.

    direct
        Call :meth:`AgentOrchestrator.run` with the default (non-adaptive)
        domain-selection path.
    adaptive
        Force the adaptive agent-expansion loop
        (:meth:`AgentOrchestrator._adaptive_select_and_run`) even if
        ``--adaptive-agents`` was not supplied on the CLI.
    debate
        Run :meth:`AgentOrchestrator.debate` — a structured multi-party
        debate between human intuition, MCP tool evidence, and domain agents.
    experiment
        Route through ``ExperimentRunnerAgent.classify_question``; if the
        question is experimentable (score ≥ threshold) generate an experiment
        plan and synthesis; otherwise fall back to ``direct``.
    portfolio
        Run ``direct`` **and** one additional approach (``debate`` when the
        question is conversational / analytical, ``experiment`` when it is
        experimentable) then reconcile results using the
        :class:`~src.analysis.weighing_system.WeighingSystem`.
    """

    DIRECT = "direct"
    ADAPTIVE = "adaptive"
    DEBATE = "debate"
    EXPERIMENT = "experiment"
    PORTFOLIO = "portfolio"
