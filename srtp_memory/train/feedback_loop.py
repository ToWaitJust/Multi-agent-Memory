"""在线 RL 反馈闭环（§11.2 阶段二，train/feedback_loop.py）。

reward = 显式点赞/点踩（+1/-1）；仅对 condition 切片做一步 bandit 式更新（V1 实现为整模型一步，
condition 切片区隔留待 V1.1）；受 REQ-405 正则化约束（β 不偏离 LLM 先验）。
"""
from __future__ import annotations

import numpy as np

from ..condition import ConditionKey
from ..weights import HybridWeightCalculator


def apply_feedback(weights: HybridWeightCalculator, query_emb: np.ndarray,
                   reward: float, condition: ConditionKey | None = None) -> dict:
    """一次在线反馈更新。reward∈{+1 赞 / -1 踩}。返回更新后的 loss 与当前权重。"""
    cond = condition or ConditionKey(user_id="u_tu", scenario_id="research", business_id="srtp")
    res = weights.update_with_feedback(query_emb, reward, cond)
    cur = weights._learnable_forward(query_emb, cond)
    return {"loss": res["loss"], "updated_weights": cur}


if __name__ == "__main__":
    w = HybridWeightCalculator()
    q = np.zeros(1024, dtype=np.float32)
    q[0] = 1.0
    print("点赞反馈:", apply_feedback(w, q, +1.0))
    print("点踩反馈:", apply_feedback(w, q, -1.0))
