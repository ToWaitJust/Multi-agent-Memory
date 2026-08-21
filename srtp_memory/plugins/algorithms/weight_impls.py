"""算法层插件：混合权重计算器（完整 + 桩 + 参考）。

- weight.full    ：HybridWeightImpl（= §4.3 HybridWeightCalculator，条件化完整实现）
- weight.uniform ：UniformWeight（固定 DEFAULT_WEIGHTS，消融退化）
- weight.reme    ：RemeNativeWeight（参考：ReMe vector_weight=0.7 映射到语义维，强基座对照）
"""
from __future__ import annotations

import numpy as np

from ...attention import DEFAULT_WEIGHTS
from ...condition import ConditionKey
from ...weights import HybridWeightCalculator
from ..base import WeightCalculator
from ..registry import register


@register("weight.full", "weight")
class HybridWeightImpl(HybridWeightCalculator, WeightCalculator):
    """默认完整实现（条件化 MLP + LLM 先验融合）。"""


@register("weight.uniform", "weight")
class UniformWeight(WeightCalculator):
    """桩：固定 DEFAULT_WEIGHTS，故意退化。"""

    def calculate(self, query_emb, condition) -> dict[str, float]:
        return dict(DEFAULT_WEIGHTS)

    def pretrain_distill(self, prior_dataset: list) -> None:
        pass

    def update_with_feedback(self, query_emb, reward: float, condition) -> None:
        pass


@register("weight.reme", "weight")
class RemeNativeWeight(WeightCalculator):
    """参考实现：委托 ReMe 原生 vector_weight=0.7（强基座对照），不重写。"""

    def calculate(self, query_emb, condition) -> dict[str, float]:
        return {"time": 0.0, "semantic": 0.7, "frequency": 0.0, "task": 0.3}

    def pretrain_distill(self, prior_dataset: list) -> None:
        pass

    def update_with_feedback(self, query_emb, reward: float, condition) -> None:
        pass
