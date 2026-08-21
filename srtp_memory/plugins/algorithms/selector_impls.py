"""算法层插件：动作选择器（完整 + 桩）。

- selector.full ：ThresholdSelectorImpl（= §4.4 ActionSelector，阈值自适应）
- selector.topk  ：TopKSelector（固定 top_k，消融退化）
- selector.recency：RecencySelector（按时间降序 top_k，附加基线）
"""
from __future__ import annotations

from ...selector import ActionSelector
from ..base import ActionSelector as ActionSelectorABC
from ..registry import register


@register("selector.full", "selector")
class ThresholdSelectorImpl(ActionSelector, ActionSelectorABC):
    """默认完整实现（阈值自适应 + 动态阈值）。"""


@register("selector.topk", "selector")
class TopKSelector(ActionSelectorABC):
    """桩：固定 top_k=5，故意退化。"""

    def __init__(self, top_k: int = 5, **kw):
        self.top_k = top_k

    def select(self, scored_memories: list[dict]) -> dict:
        kept = sorted(scored_memories, key=lambda m: m["score"]["final"], reverse=True)[: self.top_k]
        kept_ids = {id(m) for m in kept}
        discarded = [m for m in scored_memories if id(m) not in kept_ids]
        return {"kept": kept, "discarded": discarded}

    def compression_ratio(self, result: dict) -> float:
        n = len(result["kept"]) + len(result["discarded"])
        return len(result["kept"]) / n if n else 0.0


@register("selector.recency", "selector")
class RecencySelector(ActionSelectorABC):
    """附加基线：按时间降序 top_k（只看新旧，不看相关性）。"""

    def __init__(self, top_k: int = 5, **kw):
        self.top_k = top_k

    def select(self, scored_memories: list[dict]) -> dict:
        kept = sorted(scored_memories, key=lambda m: m["memory"].timestamp, reverse=True)[: self.top_k]
        kept_ids = {id(m) for m in kept}
        discarded = [m for m in scored_memories if id(m) not in kept_ids]
        return {"kept": kept, "discarded": discarded}

    def compression_ratio(self, result: dict) -> float:
        n = len(result["kept"]) + len(result["discarded"])
        return len(result["kept"]) / n if n else 0.0
