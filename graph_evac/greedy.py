"""Greedy sweeping heuristics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from configs import Config
from .problem import EvacuationProblem, Room, Responder


@dataclass
class ResponderPlan:
    responder_id: str
    assigned_rooms: List[str]
    visit_order: List[str]
    total_time: float
    segments: List[Dict]


@dataclass
class SweepPlan:
    responder_plans: List[ResponderPlan]
    makespan: float


def _room_assignments(problem: EvacuationProblem, config: Config) -> Dict[str, List[str]]:
    assignments: Dict[str, List[str]] = {resp.id: [] for resp in problem.responders}
    rooms = list(problem.rooms)

    if config.redundancy_mode == "per_responder_all_rooms":
        for responder in problem.responders:
            assignments[responder.id] = [room.id for room in rooms]
        return assignments

    if config.redundancy_mode == "double_check":
        for room in rooms:
            ranked = sorted(
                (
                    (problem.distance(responder.start_node, room.id), responder.id)
                    for responder in problem.responders
                ),
                key=lambda x: x[0],
            )
            for _, resp_id in ranked[: min(2, len(ranked))]:
                assignments[resp_id].append(room.id)
        return assignments

    # default assignment strategy
    for room in rooms:
        best_resp = min(
            problem.responders,
            key=lambda resp: (
                problem.distance(resp.start_node, room.id),
                len(assignments[resp.id]),
            ),
        )
        assignments[best_resp.id].append(room.id)
    return assignments


def _nearest_neighbor_order(responder: Responder, room_ids: Sequence[str], problem: EvacuationProblem) -> List[str]:
    order: List[str] = []
    remaining = set(room_ids)
    current = responder.start_node
    while remaining:
        next_room = min(remaining, key=lambda room_id: problem.distance(current, room_id))
        order.append(next_room)
        remaining.remove(next_room)
        current = next_room
    return order


def _room_duration(room: Room, config: Config) -> float:
    return config.base_check_time + config.time_per_occupant * room.occupants


def plan_greedy(problem: EvacuationProblem, config: Config) -> SweepPlan:
    assignments = _room_assignments(problem, config)
    responder_plans: List[ResponderPlan] = []
    makespan = 0.0

    for responder in problem.responders:
        assigned_rooms = assignments.get(responder.id, [])
        visit_order = _nearest_neighbor_order(responder, assigned_rooms, problem) if assigned_rooms else []

        segments: List[Dict] = []
        current_node = responder.start_node
        current_time = 0.0
        for room_id in visit_order:
            travel_distance = problem.distance(current_node, room_id)
            travel_time = travel_distance / config.walk_speed if config.walk_speed else 0.0
            if travel_time:
                segments.append(
                    {
                        "type": "move",
                        "target": room_id,
                        "start": current_time,
                        "end": current_time + travel_time,
                    }
                )
                current_time += travel_time
            room = problem.room_lookup[room_id]
            check_time = _room_duration(room, config)
            segments.append(
                {
                    "type": "room",
                    "target": room_id,
                    "start": current_time,
                    "end": current_time + check_time,
                }
            )
            current_time += check_time
            current_node = room_id

        if current_node != responder.start_node:
            exit_id, exit_distance = problem.nearest_exit(current_node)
            exit_time = exit_distance / config.walk_speed if config.walk_speed else 0.0
            segments.append(
                {
                    "type": "egress",
                    "target": exit_id,
                    "start": current_time,
                    "end": current_time + exit_time,
                }
            )
            current_time += exit_time

        makespan = max(makespan, current_time)
        responder_plans.append(
            ResponderPlan(
                responder_id=responder.id,
                assigned_rooms=assigned_rooms,
                visit_order=visit_order,
                total_time=current_time,
                segments=segments,
            )
        )

    return SweepPlan(responder_plans=responder_plans, makespan=makespan)


__all__ = ["plan_greedy", "SweepPlan", "ResponderPlan"]
