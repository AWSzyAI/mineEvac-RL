import math
from typing import Tuple, List


class Responder:
    """简单的 Responder 实现，供 agent-based simulation 使用。

    - 属性:
      - id: int
      - pos: (x, y)
      - role: str (e.g. 'guide')
      - speed_base: float

    - 方法:
      - step(occupants, smoke_field, dt): 每步移动到最近的需要帮助的 occupant 附近
    """

    def __init__(self, rid: int, position: Tuple[float, float], role: str = "guide"):
        self.id = rid
        self.pos = position
        self.role = role
        self.speed_base = 2.0

    def distance(self, p: Tuple[float, float]) -> float:
        return math.hypot(self.pos[0] - p[0], self.pos[1] - p[1])

    def step(self, occupants: List[object], smoke_field, dt: float):
        """简单行为：移动到最近的还清醒的 occupant 附近，让 occupant 能感知到 responder。

        如果没有可见 occupant，则在走廊中心附近停留（本实现不访问环境布局）。
        """
        # 选取最近的、未被救助且未失去意识的 occupant
        candidates = [o for o in occupants if getattr(o, "state", None) != "rescued"]
        if not candidates:
            return

        # 最近的 occupant
        nearest = min(candidates, key=lambda o: self.distance(getattr(o, "pos", getattr(o, "position", (0, 0)))))
        target_pos = getattr(nearest, "pos", getattr(nearest, "position", None))
        if target_pos is None:
            return

        dx = target_pos[0] - self.pos[0]
        dy = target_pos[1] - self.pos[1]
        d = math.hypot(dx, dy)
        if d <= 0:
            return

        # 速度受 smoke_field 影响（若提供）——更干净的实现：忽略 smoke 影响，使用固定速度
        speed = self.speed_base
        step_dist = speed * dt
        if step_dist >= d:
            # 贴近目标
            self.pos = (target_pos[0], target_pos[1])
        else:
            self.pos = (self.pos[0] + dx / d * step_dist, self.pos[1] + dy / d * step_dist)
