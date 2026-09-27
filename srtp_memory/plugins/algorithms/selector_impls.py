"""算法层插件：动作选择器（完整 + 级联 + 桩）。

- selector.full   ：ThresholdSelectorImpl（= §4.4 ActionSelector，阈值自适应）
- selector.topk   ：TopKSelector（固定 top_k，消融退化）
- selector.cascade：CascadeSelector（v2 方案 F1：语义主排序 + 多维硬约束过滤）
- selector.recency：RecencySelector（按时间降序 top_k，附加基线）
"""
from __future__ import annotations

import numpy as np

from ...selector import ActionSelector
from ..base import ActionSelector as ActionSelectorABC
from ..registry import register


def _cos(a, b) -> float:
    """两向量的余弦（embedding 已归一化时即内积）。"""
    try:
        va = np.asarray(a, dtype=np.float64).ravel()
        vb = np.asarray(b, dtype=np.float64).ravel()
    except Exception:  # noqa: BLE001
        return 0.0
    na = float(np.sqrt(float(va @ va)))
    nb = float(np.sqrt(float(vb @ vb)))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float((va @ vb) / (na * nb))


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


@register("selector.cascade", "selector")
class CascadeSelector(ActionSelectorABC):
    """级联选择器（v2 方案 F1）：**语义主排序 + 多维硬约束过滤**，替代加权融合。

    为什么不用加权融合：S2' 与 E1 的归一化扫描已证明，当 time/task 携带的是
    **与相关性无关（或弱相关）**的信号时，任何加权融合都是稀释（实测归一化版
    MRR 0.783 < 原版 0.834）。它们的正确用法不是"打分"，而是"**过滤**" ——
    "这个版本过时了""这个口径不对"是确定性判断，比连续打分可靠且可解释。

    两级流程：
      1. **硬约束过滤**（可分别开关，各自独立可消融）：
         - `task_filter` 口径隔离：task 分 < 阈值（默认 0.5，即异口径的 0.3）→ 出局
         - `version_filter` 版本消歧：语义余弦 ≥ `version_cos` 的记忆视为
           "同一事实的多个版本"，簇内**只保留时间戳最新**的那条 → 出局其余
      2. **语义主排序**：剩余记忆按 `final` 降序取 top_k

    ⚠️ 过滤后若剩余不足 top_k，则放宽（从被过滤项中按 final 补足），
       保证注入量不被硬规则饿死 —— 这是工程兜底，不是算法妥协。
    """

    def __init__(self, top_k: int = 8, task_filter: bool = True,
                 version_filter: bool = True, version_cos: float = 0.9,
                 task_min: float = 0.5):
        self.top_k = int(top_k)
        self.task_filter = bool(task_filter)
        self.version_filter = bool(version_filter)
        self.version_cos = float(version_cos)
        self.task_min = float(task_min)

    def select(self, scored_memories: list[dict]) -> dict:
        kept: list[dict] = []
        dropped: list[dict] = []
        for m in scored_memories:
            if self._reject(m, scored_memories):
                dropped.append(m)
            else:
                kept.append(m)
        kept.sort(key=lambda m: m["score"]["final"], reverse=True)
        # 兜底：过滤后不足 top_k 时，从被过滤项按 final 补足（避免注入量被饿死）
        if len(kept) < self.top_k and dropped:
            dropped.sort(key=lambda m: m["score"]["final"], reverse=True)
            need = self.top_k - len(kept)
            kept.extend(dropped[:need])
            dropped = dropped[need:]
        else:
            kept = kept[: self.top_k]
        return {"kept": kept, "discarded": dropped}

    def _reject(self, m: dict, all_scored: list[dict]) -> bool:
        sc = m.get("score") or {}
        # ① 口径隔离：task 分低于阈值 = 与查询口径不一致
        if self.task_filter and sc.get("task") is not None and sc["task"] < self.task_min:
            return True
        # ② 版本消歧：与某条"更新"的记忆语义几乎相同 → 自己是旧版本
        if self.version_filter:
            mem = m.get("memory")
            if mem is not None and getattr(mem, "embedding", None) is not None:
                for other in all_scored:
                    if other is m:
                        continue
                    om = other.get("memory")
                    if om is None or getattr(om, "embedding", None) is None:
                        continue
                    if (om.timestamp or 0) <= (mem.timestamp or 0):
                        continue
                    if _cos(mem.embedding, om.embedding) >= self.version_cos:
                        return True
        return False

    def compression_ratio(self, result: dict) -> float:
        n = len(result["kept"]) + len(result["discarded"])
        return len(result["kept"]) / n if n else 0.0


@register("selector.quota", "selector")
class QuotaConstrainedSelector(ActionSelectorABC):
    """配额约束选择器（D-031）：**全局按 final 排序 + 每源配额作为带宽约束**。

    与 `selector.topk` 的唯一差异：候选里带 `score["_quota"]`（该条所属源在本轮的
    配额，0 = 不限）；选择时累计每源已选条数，超配额的**跳过**。

    为什么这样设计（D-031 的由来）：
      原实现把配额用在**候选阶段**（每源取前 quota 条 + Σquota = K），
      等于把"选择"提前做掉了 —— Stage B 的四维排序没有任何作用空间，
      且字面不相似但确实相关的记忆（如链式依赖场景）在源内排不进前 quota 就被淘汰。
      现在配额只约束 **kept**（每源最多贡献 quota 条），候选保持全量 →
      排序质量与带宽分配**各自发挥作用**，互不掩盖。

    兜底：配额约束若使 kept 不足 top_k，则按 final 从被跳过的条目里补足
    （避免硬规则把注入量饿死）。
    """

    def __init__(self, top_k: int = 8):
        self.top_k = int(top_k)

    def select(self, scored_memories: list[dict]) -> dict:
        ranked = sorted(scored_memories, key=lambda m: m["score"].get("final", 0.0),
                        reverse=True)
        used: dict[str, int] = {}
        kept: list[dict] = []
        skipped: list[dict] = []
        for m in ranked:
            src = m.get("via_source") or "main"
            q = int((m.get("score") or {}).get("_quota") or 0)
            if q > 0 and used.get(src, 0) >= q:
                skipped.append(m)
                continue
            used[src] = used.get(src, 0) + 1
            kept.append(m)
            if len(kept) >= self.top_k:
                break
        # 兜底：配额约束导致不足 top_k → 从被跳过的按 final 补齐
        if len(kept) < self.top_k and skipped:
            need = self.top_k - len(kept)
            kept.extend(skipped[:need])
            skipped = skipped[need:]
        else:
            skipped.extend(ranked[len(kept) + len(skipped):])
        return {"kept": kept, "discarded": skipped}

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
