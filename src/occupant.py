# occupant.py
import math
import random

class Occupant:
    def __init__(self, oid, position, exit_points):
        self.id = oid
        self.pos = position            # (x, y)
        self.state = "idle"            # idle, moving, unconscious, rescued
        self.hp = 100.0                # health points
        self.awareness = 0.0           # 0–1, 认知火灾程度
        self.panic = 0.0               # 0–1, 惊慌度
        self.vision = 1.0              # 可见度，受烟雾影响
        self.exit_points = exit_points # [(x,y), ...]
        self.target = None             # 当前目标点
        self.speed_base = 1.0
        self.direction = (0.0, 0.0)

    def update_awareness(self, smoke, alarm, see_others):
        # 烟雾、警报、社会线索增强觉察
        delta = 0.1 * alarm + 0.05 * smoke + 0.1 * see_others
        self.awareness = min(1.0, self.awareness + delta)

    def update_health(self, smoke_concentration, dt):
        # 烟雾损伤
        self.hp -= smoke_concentration * 2.0 * dt
        if self.hp <= 20:
            self.state = "unconscious"

    def update_panic(self, density, smoke):
        # 密度与烟雾增强panic
        base = 0.2 * density + 0.3 * smoke
        self.panic = min(1.0, base + (1 - self.awareness) * 0.3)

    def perceive_responders(self, responders):
        visible = [r for r in responders if self.distance(r.pos) < 8]
        return visible

    def follow_responder(self, responder):
        # 将目标方向设为 responder 的位置方向
        dx, dy = responder.pos[0] - self.pos[0], responder.pos[1] - self.pos[1]
        d = math.hypot(dx, dy)
        if d > 0:
            self.direction = (dx / d, dy / d)
        self.target = responder.pos

    def random_wander(self):
        theta = random.gauss(0, 3.14 * self.panic)
        self.direction = (math.cos(theta), math.sin(theta))

    def move_toward_exit(self):
        if not self.exit_points: return
        # 选择最近出口
        self.target = min(self.exit_points, key=lambda e: self.distance(e))
        dx, dy = self.target[0]-self.pos[0], self.target[1]-self.pos[1]
        d = math.hypot(dx, dy)
        if d > 0:
            self.direction = (dx / d, dy / d)

    def step(self, smoke_conc, responders, dt):
        if self.state == "unconscious":
            return

        # 更新可见度与速度
        self.vision = math.exp(-0.05 * smoke_conc)
        v = self.speed_base * self.vision * (1 - 0.5 * self.panic)

        # 感知 responder
        visible_resp = self.perceive_responders(responders)
        if visible_resp:
            self.follow_responder(visible_resp[0])
            self.panic *= 0.8
        elif smoke_conc > 0.6:
            self.random_wander()
        else:
            self.move_toward_exit()

        # 移动
        self.pos = (
            self.pos[0] + self.direction[0] * v * dt,
            self.pos[1] + self.direction[1] * v * dt,
        )

        # 更新HP
        self.update_health(smoke_conc, dt)

    def distance(self, p):
        return math.hypot(self.pos[0] - p[0], self.pos[1] - p[1])
