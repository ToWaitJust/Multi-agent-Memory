"""算法层插件：奖励信号（完整 + 桩）。

- reward.explicit ：ExplicitReward（显式点赞/点踩 → +1 / -1）
- reward.none     ：NoReward（无反馈，消融退化，恒 0 不触发在线更新）
"""
from __future__ import annotations

from ..base import RewardSignal
from ..registry import register


@register("reward.explicit", "reward")
class ExplicitReward(RewardSignal):
    """显式点赞/点踩：+1 / -1。"""

    def from_feedback(self, like: bool) -> float:
        return 1.0 if like else -1.0


@register("reward.none", "reward")
class NoReward(RewardSignal):
    """无反馈信号（消融组专用）：恒返回 0，不触发 update_with_feedback。"""

    def from_feedback(self, like: bool) -> float:
        return 0.0
