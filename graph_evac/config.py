"""Configuration helpers for the MineEvac graph abstraction."""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
import os
from typing import Any, Dict


def _coerce_value(value: str, target_type: Any) -> Any:
    """Coerce ``value`` to ``target_type`` with a few best-effort rules."""
    origin = getattr(target_type, "__origin__", None)
    if origin is not None and origin is list:
        raise TypeError("List configuration values are not supported via environment variables")
    if target_type is bool:
        return value.lower() in {"1", "true", "yes", "on"}
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return value


@dataclass
class Config:
    """Container for the evacuation simulation configuration."""

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
    output_dir: str = "outputs"

    def update_from_env(self, prefix: str = "GRAPH_EVAC_") -> "Config":
        """Override configuration fields from environment variables."""

        for field in fields(self):
            env_key = f"{prefix}{field.name.upper()}"
            if env_key in os.environ:
                raw_value = os.environ[env_key]
                try:
                    coerced = _coerce_value(raw_value, field.type)
                except Exception as exc:  # pragma: no cover - defensive branch
                    raise ValueError(f"Failed to parse env var {env_key}: {raw_value}") from exc
                setattr(self, field.name, coerced)
        return self

    def as_dict(self) -> Dict[str, Any]:
        """Return the configuration as a serialisable dictionary."""

        return asdict(self)


__all__ = ["Config"]
