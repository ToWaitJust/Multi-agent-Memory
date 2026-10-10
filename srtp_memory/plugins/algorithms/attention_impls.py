"""算法层插件：四维注意力头（完整 + 归一化 + 自适应时间尺度 + 桩）。

- attention.full          ：MultiHeadAttentionImpl（= §4.2 MultiHeadAttentionMemoryScorer，默认完整实现）
- attention.normalized    ：批级去偏 + 无效维门控 + 语义保底（S0 修复，见 attention.py 类文档）
- attention.adaptive_time ：批级自适应时间尺度（λ_eff = K / 候选时间跨度），
                            修 LoCoMo 上暴露的"固定 λ 与语料跨度不匹配"缺陷（D-050）
- attention.noop          ：NoOpAttention（均匀 0.25，消融退化，enabled_dims=set() 等价）
"""
from __future__ import annotations

import numpy as np

from ...attention import (DEFAULT_WEIGHTS, AdaptiveGatedScorer,
                          MultiHeadAttentionMemoryScorer,
                          NormalizedAttentionScorer, TimeAdaptiveScorer)
from ..base import AttentionHead
from ..registry import register


@register("attention.full", "attention")
class MultiHeadAttentionImpl(MultiHeadAttentionMemoryScorer, AttentionHead):
    """默认完整实现（四维打分 + enabled_dims 掩码）。"""

    def names(self) -> list[str]:
        return ["time", "semantic", "frequency", "task"]


@register("attention.normalized", "attention")
class NormalizedAttentionImpl(NormalizedAttentionScorer, AttentionHead):
    """批级归一化实现（S0 修复）：门控无区分度维 + 权重重分配 + 语义保底 + 批内 min-max。"""

    def names(self) -> list[str]:
        return ["time", "semantic", "frequency", "task"]


@register("attention.adaptive_time", "attention")
class AdaptiveTimeAttentionImpl(TimeAdaptiveScorer, AttentionHead):
    """自适应时间尺度实现（LoCoMo 诊断 D-050）。

    与 attention.full 的**唯一**差异：指数衰减的 λ 不再是写死的 `LAMBDA_TIME=0.01/小时`，
    而是按当前候选集的时间戳跨度自适应的 `λ_eff = TIME_DECAY_K / span_hours`。
    四维公式、权重、门控语义完全不变，因此可作为**单变量消融**验证该修复。
    """

    def names(self) -> list[str]:
        return ["time", "semantic", "frequency", "task"]


@register("attention.adaptive_gated", "attention")
class AdaptiveGatedAttentionImpl(AdaptiveGatedScorer, AttentionHead):
    """自适应时间尺度 + 常量维门控（无 min-max，D-050）。

    在 `attention.adaptive_time` 之上再叠加一层"无效维权重回收"：
    LoCoMo 上 `task` 恒 1.0 / `frequency` 首轮恒 0，合计白占 0.4 权重，
    门控后这些权重按原比例还给语义等存活维。
    """

    def names(self) -> list[str]:
        return ["time", "semantic", "frequency", "task"]


@register("attention.noop", "attention")
class NoOpAttention(AttentionHead):
    """桩：均匀 0.25，故意退化（intentional degradation，消融干净的前提）。"""

    def score(self, query_emb, memory, current_time, task_tag, enabled_dims=None) -> dict:
        return {"final": 0.25, "time": 0.25, "semantic": 0.25,
                "frequency": 0.25, "task": 0.25}

    def names(self) -> list[str]:
        return ["time", "semantic", "frequency", "task"]
