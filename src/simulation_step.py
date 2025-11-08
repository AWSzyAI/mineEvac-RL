"""简易仿真驱动脚本（基于 agent 风格的 Occupant / Responder）。

这个文件提供了一个小型驱动：初始化若干 occupant 与 responder，然后按时间步推进。
"""
import math
import random
import time

from occupant import Occupant
from responder import Responder
from simulation_logger import SimulationLogger


# 初始化
exits = [(10, 0), (90, 0)]
occupants = [Occupant(i, (random.randint(20, 80), random.randint(10, 40)), exits) for i in range(10)]
responders = [Responder(0, (50.0, 25.0), role="guide")]


def smoke_field(pos):
    # 简化烟雾场，越靠近(50,25)越浓
    x, y = pos
    return min(1.0, math.exp(-((x - 50) ** 2 + (y - 25) ** 2) / 400))


def run_sim(steps=500, dt=0.2, verbose=True, log_trajectories: bool = True):
    logger = SimulationLogger() if log_trajectories else None

    for t in range(steps):
        # occupants 先移动
        for o in occupants:
            o.step(smoke_field(o.pos), responders, dt)

        # responders 更新
        for r in responders:
            r.step(occupants, smoke_field, dt)

        # 记录轨迹（同时记录 responders）
        if logger is not None:
            logger.record(t, occupants, responders)

        if verbose and (t % 50 == 0 or t == steps - 1):
            alive = sum(1 for o in occupants if o.state != "unconscious" and o.state != "rescued")
            print(f"t={t}, responder_pos={responders[0].pos}, alive_occupants={alive}")

    if logger is not None:
        logger.close()


if __name__ == "__main__":
    run_sim()
