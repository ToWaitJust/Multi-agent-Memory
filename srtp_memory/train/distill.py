"""离线 MSE 蒸馏预训练（§11.2 阶段一，train/distill.py）。

让可学习 MLP 复现 LLM 先验权重判断。样本 = 真实交互日志 + 模拟生成。
用法：python -m srtp_memory.train.distill（示例，接真实数据后替换数据集来源）。
"""
from __future__ import annotations

import numpy as np

from ..attention import DEFAULT_WEIGHTS
from ..condition import ConditionKey
from ..weights import HybridWeightCalculator


def _simulate_dataset(n: int = 50, query_dim: int = 1024, seed: int = 42) -> list:
    """模拟蒸馏样本（真实样本 = 交互日志中 query_emb + LLM 先验权重）。"""
    rng = np.random.default_rng(seed)
    dataset = []
    for _ in range(n):
        q_emb = rng.normal(0, 1, query_dim).astype(np.float32)
        cond = ConditionKey(user_id="u_tu", scenario_id="research", business_id="srtp")
        # 模拟 LLM 先验：语义维偏高（贴近真实分布）
        w = {"time": 0.2, "semantic": 0.45, "frequency": 0.15, "task": 0.2}
        dataset.append((q_emb, cond, w))
    return dataset


def run(n: int = 50, epochs: int = 3) -> dict:
    w = HybridWeightCalculator()
    dataset = _simulate_dataset(n=n)
    for _ in range(epochs):
        w.pretrain_distill(dataset)
    # 验证：蒸馏后输出应接近默认/先验分布
    q = np.zeros(1024, dtype=np.float32)
    q[0] = 1.0
    out = w._learnable_forward(q, ConditionKey(user_id="u_tu", scenario_id="research", business_id="srtp"))
    return {"final_weights": out, "default": DEFAULT_WEIGHTS}


if __name__ == "__main__":
    print(run())
