"""Timeline simulator for evacuation plans."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from configs import Config
from .greedy import SweepPlan
from .problem import EvacuationProblem


@dataclass
class TimelineEntry:
    responder_id: str
    segment_type: str
    target: str
    start: float
    end: float


def simulate_sweep(problem: EvacuationProblem, plan: SweepPlan, config: Config) -> List[TimelineEntry]:
    timeline: List[TimelineEntry] = []
    for responder_plan in plan.responder_plans:
        for segment in responder_plan.segments:
            timeline.append(
                TimelineEntry(
                    responder_id=responder_plan.responder_id,
                    segment_type=segment["type"],
                    target=segment["target"],
                    start=segment["start"],
                    end=segment["end"],
                )
            )
    return timeline


__all__ = ["simulate_sweep", "TimelineEntry"]
