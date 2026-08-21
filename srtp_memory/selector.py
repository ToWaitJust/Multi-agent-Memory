"""动作选择器（§4.4，FR-5）。

依据最终得分 + 动态阈值决定每条记忆 保留（进共享池）/ 丢弃，替代固定 top_k。
自适应阈值：max(初始阈值, mean - 0.5·std)，避免候选整体偏低时全丢。
"""
from __future__ import annotations

import statistics
from typing import Any


class ActionSelector:
    """阈值自适应选择器（默认完整实现，插件名 selector.full）。"""

    def __init__(self, threshold: float = 0.6, max_shared: int = 10):
        self.threshold = threshold
        self.max_shared = max_shared

    def select(self, scored_memories: list[dict]) -> dict[str, list[dict]]:
        """输入 [{memory, score:{final,...}, ...}]，返回 {kept, discarded}。

        kept 按 final 降序、截断 max_shared；discarded 为其余。
        """
        finals = [m["score"]["final"] for m in scored_memories]
        thr = self._dynamic_threshold(finals)
        kept = [m for m in scored_memories if m["score"]["final"] >= thr]
        kept.sort(key=lambda m: m["score"]["final"], reverse=True)
        kept = kept[: self.max_shared]
        discarded = [m for m in scored_memories if m not in kept]
        return {"kept": kept, "discarded": discarded}

    def _dynamic_threshold(self, finals: list[float]) -> float:
        if not finals:
            return self.threshold
        mean = statistics.mean(finals)
        std = statistics.stdev(finals) if len(finals) > 1 else 0.0
        return max(self.threshold, mean - 0.5 * std)

    def compression_ratio(self, result: dict[str, list[dict]]) -> float:
        """len(kept)/len(总候选)（验收 ≤0.3）。"""
        n_kept = len(result["kept"])
        n_total = n_kept + len(result["discarded"])
        return n_kept / n_total if n_total else 0.0
