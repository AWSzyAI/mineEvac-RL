"""Problem definitions for MineEvac graph abstraction."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .config import Config

Coord = Tuple[float, float, float]


def _euclidean(a: Coord, b: Coord) -> float:
    return sum((ax - bx) ** 2 for ax, bx in zip(a, b)) ** 0.5


@dataclass
class Room:
    id: str
    coord: Coord
    floor: int
    occupants: int


@dataclass
class Responder:
    id: str
    start_node: str
    floor: int = 0


@dataclass
class Exit:
    id: str
    coord: Coord
    floor: int


@dataclass
class EvacuationProblem:
    rooms: Sequence[Room]
    responders: Sequence[Responder]
    exits: Sequence[Exit]
    config: Config
    graph: Mapping[str, Coord] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.graph:
            node_coords: Dict[str, Coord] = {}
            for room in self.rooms:
                node_coords[room.id] = room.coord
            for exit_node in self.exits:
                node_coords[exit_node.id] = exit_node.coord
            for responder in self.responders:
                node_coords.setdefault(responder.start_node, node_coords[self.exits[0].id])
            self.graph = node_coords
        self._distances: Dict[Tuple[str, str], float] = {}
        nodes = list(self.graph.items())
        for idx, (name_a, coord_a) in enumerate(nodes):
            for name_b, coord_b in nodes[idx:]:
                distance = _euclidean(coord_a, coord_b)
                self._distances[(name_a, name_b)] = distance
                self._distances[(name_b, name_a)] = distance

        self._room_lookup = {room.id: room for room in self.rooms}
        self._exit_lookup = {exit_node.id: exit_node for exit_node in self.exits}

    def distance(self, node_a: str, node_b: str) -> float:
        return self._distances[(node_a, node_b)]

    def nearest_exit(self, node_id: str) -> Tuple[str, float]:
        best = min(((exit_id, self.distance(node_id, exit_id)) for exit_id in self._exit_lookup), key=lambda x: x[1])
        return best

    def as_basic(self) -> Tuple[Sequence[Room], Sequence[Responder], Mapping[str, Coord]]:
        return self.rooms, self.responders, self.graph

    @property
    def room_lookup(self) -> Mapping[str, Room]:
        return self._room_lookup


__all__ = ["Room", "Responder", "Exit", "EvacuationProblem"]
