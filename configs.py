"""Centralised configuration objects for MineEvac workflows."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _coerce_value(value: str, target_type: Any) -> Any:
    """Best-effort coercion used by environment overrides."""

    origin = getattr(target_type, "__origin__", None)
    if origin in {list, List}:  # comma-separated parsing for lists
        return [item.strip() for item in value.split(",") if item.strip()]
    if target_type is bool:
        return value.lower() in {"1", "true", "yes", "on"}
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return value


@dataclass
class Config:
    """Simulation-wide settings for deterministic graph sweeps."""

    redundancy_mode: str = "assignment"
    empirical_mode: bool = False
    base_check_time: float = 30.0
    time_per_occupant: float = 2.0
    walk_speed: float = 1.2
    responder_speed_search: float = 1.0
    responder_speed_carry: float = 0.8
    carry_capacity: int = 2
    comm_success: float = 1.0
    egress_distance_factor: float = 1.0
    floors: int = 1
    floor_spacing: float = 4.0
    algorithm: str = "greedy"
    layout_path: str = "layout/baseline.json"
    simulate: bool = True
    output_dir: str = "artifacts"
    plan_filename: str = "plan.json"
    timeline_json_filename: str = "timeline.json"
    timeline_csv_filename: str = "timeline.csv"
    log_filename: str = "run.log"
    gif_filename: str = "timeline.gif"
    run_name: str | None = None

    def ensure_output_dir(self) -> Path:
        directory = Path(self.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def output_path(self, filename: str) -> Path:
        return self.ensure_output_dir() / filename

    def label(self) -> str:
        return self.run_name or f"{self.algorithm}_{self.redundancy_mode}_F{self.floors}"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def update_from_env(self, prefix: str = "GRAPH_EVAC_") -> "Config":
        for field in fields(self):
            env_key = f"{prefix}{field.name.upper()}"
            if env_key in os.environ:
                raw_value = os.environ[env_key]
                coerced = _coerce_value(raw_value, field.type)
                setattr(self, field.name, coerced)
        return self


@dataclass
class BatchSettings:
    """Grid definition for `make batch`."""

    layout_path: str = "layout/baseline.json"
    redundancy_modes: List[str] = field(
        default_factory=lambda: ["assignment", "double_check", "per_responder_all_rooms"]
    )
    floors: List[int] = field(default_factory=lambda: [1, 2])
    algorithms: List[str] = field(default_factory=lambda: ["greedy"])
    floor_spacing: float = 4.0
    output_root: str = "batch_runs"

    def iter_configs(self) -> Iterable[Config]:
        for algorithm in self.algorithms:
            for floor in self.floors:
                for redundancy in self.redundancy_modes:
                    label = f"{algorithm}_{redundancy}_F{floor}"
                    yield Config(
                        layout_path=self.layout_path,
                        floors=floor,
                        floor_spacing=self.floor_spacing,
                        redundancy_mode=redundancy,
                        algorithm=algorithm,
                        output_dir=str(Path(self.output_root) / label),
                        run_name=label,
                    )


__all__ = ["Config", "BatchSettings"]
