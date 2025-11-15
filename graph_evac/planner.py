"""Planning interface for MineEvac graph abstraction."""
from __future__ import annotations

from configs import Config
from .greedy import SweepPlan, plan_greedy
from .problem import EvacuationProblem


def plan_sweep(problem: EvacuationProblem, config: Config, algorithm: str = "greedy") -> SweepPlan:
    algorithm = algorithm or config.algorithm
    if algorithm == "greedy":
        return plan_greedy(problem, config)
    raise NotImplementedError(f"Unknown algorithm: {algorithm}")


__all__ = ["plan_sweep"]
