"""算法层插件：四维注意力头（完整 + 归一化 + 桩）。

- attention.full       ：MultiHeadAttentionImpl（= §4.2 MultiHeadAttentionMemoryScorer，默认完整实现）
- attention.normalized ：批级去偏 + 无效维门控 + 语义保底（S0 修复，见 attention.py 类文档）
- attention.noop       ：NoOpAttention（均匀 0.25，消融退化，enabled_dims=set() 等价）
"""
from __future__ import annotations

import numpy as np

from ...attention import (DEFAULT_WEIGHTS, MultiHeadAttentionMemoryScorer,
                          NormalizedAttentionScorer)
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


@register("attention.noop", "attention")
class NoOpAttention(AttentionHead):
    """桩：均匀 0.25，故意退化（intentional degradation，消融干净的前提）。"""

    def score(self, query_emb, memory, current_time, task_tag, enabled_dims=None) -> dict:
        return {"final": 0.25, "time": 0.25, "semantic": 0.25,
                "frequency": 0.25, "task": 0.25}

    def names(self) -> list[str]:
        return ["time", "semantic", "frequency", "task"]
