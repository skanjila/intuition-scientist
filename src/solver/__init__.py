"""Solver policy package — strategy selection and routing.

Public API
----------
SolverPolicy
    Enum of selection policies: ``auto``, ``baseline``, ``explore``, ``fixed``.
SolverApproach
    Enum of runnable approaches: ``direct``, ``adaptive``, ``debate``,
    ``experiment``, ``portfolio``.
StrategyRouter
    Epsilon-greedy router that picks an approach given a question and policy.
"""

from src.solver.policy import SolverApproach, SolverPolicy
from src.solver.router import StrategyRouter

__all__ = ["SolverApproach", "SolverPolicy", "StrategyRouter"]
