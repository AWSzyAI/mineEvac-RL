import sys
import pytest

# The environment module lives under src/
sys.path.append("src")

pytest.importorskip("gymnasium")

from mine_evac_env import MineEvacEnv


def test_responder_crosses_door_into_room():
    env = MineEvacEnv("layout/baseline.json", max_steps=10)
    env.reset()
    door_x = env.layout.doors_xs[0]
    start_pos = (door_x, env.layout.doors_bottom_z + 1)
    env.responder.position = start_pos
    _, _, _, _, info = env.step(4)
    assert env.responder.position == (door_x, env.layout.doors_bottom_z)
    assert info.get("door_crossed"), "Expected door_crossed flag after moving through doorway"
