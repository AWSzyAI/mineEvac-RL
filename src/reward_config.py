from dataclasses import dataclass, field


@dataclass
class RewardTerm:
    enabled: bool = True
    weight: float = 0.0


@dataclass
class RewardConfig:
    # Base step penalties / rewards
    # 每步时间成本（负奖励）
    time_penalty: RewardTerm = field(default_factory=lambda: RewardTerm(True, -0.5))
    # 任意 evac 增量（自救或被救）带来的奖励
    delta_evac: RewardTerm = field(default_factory=lambda: RewardTerm(True, 0.5))
    # 被 responder 救出的个体奖励
    rescued_bonus: RewardTerm = field(default_factory=lambda: RewardTerm(True, 10.0))
    # occupants 自主撤离的奖励
    self_evac_bonus: RewardTerm = field(default_factory=lambda: RewardTerm(True, 1.0))
    # 房间完全清空奖励
    room_clear_bonus: RewardTerm = field(default_factory=lambda: RewardTerm(True, 15.0))
    # 尚未处理完的需求数量惩罚
    needs_remaining_penalty: RewardTerm = field(default_factory=lambda: RewardTerm(False, -0.5))
    # 原地不动的 responder 惩罚
    responder_still_penalty: RewardTerm = field(default_factory=lambda: RewardTerm(True, -1.5))
    # 探索新格子的奖励
    new_cell_bonus: RewardTerm = field(default_factory=lambda: RewardTerm(True, 0.5))
    # 第一次踏入房间的奖励
    room_entry_bonus: RewardTerm = field(default_factory=lambda: RewardTerm(True, 15.0))
    # 成功附着/带上需要救援者的奖励
    attach_bonus: RewardTerm = field(default_factory=lambda: RewardTerm(True, 5.0))
    # 身处未清空房间时的奖励（鼓励继续搜）
    in_uncleared_room_bonus: RewardTerm = field(default_factory=lambda: RewardTerm(True, 0.8))
    # needs 数量下降的奖励
    needs_delta_bonus: RewardTerm = field(default_factory=lambda: RewardTerm(True, 2.0))
    # 远离走廊、进入房间区域的奖励
    far_from_corridor_bonus: RewardTerm = field(default_factory=lambda: RewardTerm(True, 0.5))
    # 穿过门洞的奖励
    door_cross_bonus: RewardTerm = field(default_factory=lambda: RewardTerm(True, 10.0))
    # 在门口磨蹭的惩罚
    door_idle_penalty: RewardTerm = field(default_factory=lambda: RewardTerm(False, -0.5))

    # Episode termination terms
    # 成功撤离所有人的终局奖励
    success_bonus: RewardTerm = field(default_factory=lambda: RewardTerm(True, 200.0))
    # 时间被截断（失败）时的惩罚
    truncation_penalty: RewardTerm = field(default_factory=lambda: RewardTerm(True, -200.0))
    # 终局剩余 occupants 惩罚
    remaining_penalty: RewardTerm = field(default_factory=lambda: RewardTerm(True, -10.0))
    # 终局未清空房间数量的惩罚
    uncleared_penalty: RewardTerm = field(default_factory=lambda: RewardTerm(True, -4.0))


reward_cfg = RewardConfig()
