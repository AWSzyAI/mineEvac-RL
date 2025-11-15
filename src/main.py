"""Command-line entrypoint for the MineEvac graph abstraction."""
from __future__ import annotations

import argparse
from typing import Iterable, List

from graph_evac import (
    Config,
    EvacuationProblem,
    Exit,
    Responder,
    Room,
    ensure_dir,
    expand_floors,
    load_layout,
    plan_sweep,
    save_json,
    save_timeline,
    simulate_sweep,
)


def _rooms_from_dicts(room_dicts: Iterable[dict]) -> List[Room]:
    rooms: List[Room] = []
    for room in room_dicts:
        rooms.append(
            Room(
                id=room["id"],
                coord=tuple(room["coord"]),
                floor=room.get("floor", 0),
                occupants=int(room.get("occupants", 0)),
            )
        )
    return rooms


def _responders_from_dicts(responder_dicts: Iterable[dict]) -> List[Responder]:
    responders: List[Responder] = []
    for responder in responder_dicts:
        responders.append(
            Responder(
                id=responder["id"],
                start_node=responder["start_node"],
                floor=responder.get("floor", 0),
            )
        )
    return responders


def _exits_from_dicts(exit_dicts: Iterable[dict]) -> List[Exit]:
    exits: List[Exit] = []
    for exit_def in exit_dicts:
        exits.append(
            Exit(
                id=exit_def["id"],
                coord=tuple(exit_def["coord"]),
                floor=exit_def.get("floor", 0),
            )
        )
    return exits


def build_problem(config: Config) -> EvacuationProblem:
    rooms_raw, responders_raw, exits_raw, _ = load_layout(config.layout_path)
    rooms_raw, responders_raw, exits_raw = expand_floors(
        rooms_raw, responders_raw, exits_raw, config.floors, config.floor_spacing
    )
    rooms = _rooms_from_dicts(rooms_raw)
    responders = _responders_from_dicts(responders_raw)
    exits = _exits_from_dicts(exits_raw)
    return EvacuationProblem(rooms=rooms, responders=responders, exits=exits, config=config)


def _plan_to_dict(plan) -> dict:
    return {
        "makespan": plan.makespan,
        "responders": [
            {
                "responder_id": rp.responder_id,
                "assigned_rooms": rp.assigned_rooms,
                "visit_order": rp.visit_order,
                "total_time": rp.total_time,
                "segments": rp.segments,
            }
            for rp in plan.responder_plans
        ],
    }


def run_from_cli(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description="MineEvac greedy abstraction")
    parser.add_argument("--layout", default="layout/baseline.json", help="Layout JSON path")
    parser.add_argument("--floors", type=int, default=1, help="Number of floors to replicate")
    parser.add_argument("--floor-spacing", type=float, default=4.0, help="Vertical spacing between floors")
    parser.add_argument("--algorithm", default="greedy", help="Planner algorithm")
    parser.add_argument("--redundancy", default="assignment", help="Responder redundancy mode")
    parser.add_argument("--no-sim", action="store_true", help="Skip the timeline simulation")
    parser.add_argument("--output", default="outputs", help="Directory for JSON/CSV outputs")
    args = parser.parse_args(argv)

    config = Config(
        layout_path=args.layout,
        floors=args.floors,
        floor_spacing=args.floor_spacing,
        algorithm=args.algorithm,
        redundancy_mode=args.redundancy,
        simulate=not args.no_sim,
        output_dir=args.output,
    )
    config.update_from_env()

    problem = build_problem(config)
    plan = plan_sweep(problem, config, algorithm=config.algorithm)

    out_dir = ensure_dir(config.output_dir)
    save_json(_plan_to_dict(plan), str(out_dir / "plan.json"))

    if config.simulate:
        timeline = simulate_sweep(problem, plan, config)
        timeline_rows = [entry.__dict__ for entry in timeline]
        save_timeline(timeline_rows, str(out_dir / "timeline.csv"))
        save_json(timeline_rows, str(out_dir / "timeline.json"))
        return plan, timeline

    return plan, []


if __name__ == "__main__":
    # 1. Gather configuration inputs from the CLI or environment
    # 2. Load the layout description and replicate floors if requested
    # 3. Build the evacuation problem model and select an algorithm
    # 4. Generate the sweep plan, optionally simulate the execution, and persist artefacts
    pass
