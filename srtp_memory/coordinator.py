"""调度编排（§4.6，FR-5/6/7 总装配）。

MemoryCoordinator 持常驻完整池（resident_pool，per-user 真池）+ 检索插件（retriever）
+ 四维打分器 + 混合权重 + 选择器 + 共享池；按 (user_id, session_id) 隔离。
不持有 ReMeMiddleware（检索由 retriever 在常驻池上接管，D-9）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .attention import MemoryCandidate, MultiHeadAttentionMemoryScorer
from .condition import ConditionKey
from .retriever import BaseRetriever
from .selector import ActionSelector
from .shared_pool import SharedMemoryPool
from .weights import HybridWeightCalculator


@dataclass
class ScheduleResult:
    """一次调度决策的结果（供埋点与注入）。"""

    query: str
    query_emb: Any = None
    task_tag: Optional[str] = None
    condition_key: Optional[ConditionKey] = None
    weights: dict[str, float] = field(default_factory=dict)
    candidates: list[MemoryCandidate] = field(default_factory=list)
    scored: list[dict] = field(default_factory=list)
    kept: list[MemoryCandidate] = field(default_factory=list)
    discarded: list[MemoryCandidate] = field(default_factory=list)
    compression_ratio: float = 0.0
    latency_ms: dict[str, float] = field(default_factory=dict)
    enabled_dims: Optional[set[str]] = None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "task_tag": self.task_tag,
            "condition_key": (
                "|".join(x or "" for x in self.condition_key.to_tuple())
                if self.condition_key else None
            ),
            "weights": self.weights,
            "candidate_count": len(self.candidates),
            "kept_count": len(self.kept),
            "discarded_count": len(self.discarded),
            "compression_ratio": self.compression_ratio,
            "latency_ms": self.latency_ms,
            "enabled_dims": sorted(self.enabled_dims) if self.enabled_dims else None,
        }


class MemoryCoordinator:
    def __init__(self, resident_pool, retriever: BaseRetriever,
                 scorer: Optional[MultiHeadAttentionMemoryScorer] = None,
                 weights: Optional[HybridWeightCalculator] = None,
                 selector: Optional[ActionSelector] = None,
                 shared_pool: Optional[SharedMemoryPool] = None,
                 user_id: str = "", session_id: str = "",
                 logger=None):
        self.resident_pool = resident_pool
        self.retriever = retriever
        self.scorer = scorer or MultiHeadAttentionMemoryScorer()
        self.weights = weights or HybridWeightCalculator()
        self.selector = selector or ActionSelector()
        self.shared_pool = shared_pool or SharedMemoryPool()
        self.user_id = user_id
        self.session_id = session_id
        self.logger = logger  # ScheduleLogger（可选）

    def schedule(self, query: str, query_emb, task_tag: Optional[str],
                 now: float, enabled_dims: Optional[set[str]] = None,
                 condition: Optional[ConditionKey] = None,
                 top_k: int = 50) -> ScheduleResult:
        """① 常驻池召回 → ② 四维打分 → ③ 混合权重重算 final → ④ 选择 → ⑤ 共享池 → ⑥ 埋点。"""
        res = ScheduleResult(query=query, query_emb=query_emb, task_tag=task_tag,
                             condition_key=condition, enabled_dims=enabled_dims)

        # ① 候选召回（BaseRetriever 在常驻池，候选集 candidate_override）
        t0 = _now_ms()
        candidates = self.retriever.recall(query, self.resident_pool, top_k=top_k)
        res.candidates = candidates
        res.latency_ms["recall"] = _now_ms() - t0

        # ③ 混合权重（先验由中间件提前 set_prior；此处只融合）
        t0 = _now_ms()
        cond = condition or ConditionKey()
        weights = self.weights.calculate(query_emb, cond)
        res.weights = weights
        res.latency_ms["weights"] = _now_ms() - t0

        # ② 四维打分 + 重算 final（用动态权重）
        t0 = _now_ms()
        for c in candidates:
            sc = self.scorer.score(query_emb, c, now, task_tag, enabled_dims=enabled_dims)
            sc["final"] = sum(weights[d] * sc[d] for d in weights) / (sum(weights.values()) or 1.0)
            res.scored.append({"memory": c, "score": sc})
        res.latency_ms["scoring"] = _now_ms() - t0

        # ④ 动作选择
        t0 = _now_ms()
        sel = self.selector.select(res.scored)
        res.kept = [m["memory"] for m in sel["kept"]]
        res.discarded = [m["memory"] for m in sel["discarded"]]
        res.compression_ratio = self.selector.compression_ratio(sel)
        res.latency_ms["select"] = _now_ms() - t0

        # ⑤ 写入共享池（频率计数同步到常驻池）
        final_by_id = {s["memory"].memory_id: s["score"]["final"] for s in res.scored}
        for m in res.kept:
            self.shared_pool.add({
                "memory_id": m.memory_id, "text": m.text, "path": m.path,
                "start_line": m.start_line, "end_line": m.end_line,
                "score": final_by_id.get(m.memory_id, 0.0),
                "timestamp": m.timestamp,
            })
            self.resident_pool.mark_access(m.memory_id)

        res.latency_ms["total"] = sum(res.latency_ms.values())

        # ⑥ L3 埋点
        if self.logger is not None:
            self.logger.log_schedule(res.to_dict())
        return res

    def dispatch(self, sub_agent, top_k: int = 10) -> list[dict]:
        """共享池 top() 全量（≤10）格式化后交副线中间件注入 HintBlock。"""
        items = self.shared_pool.top()[:top_k]
        return items

    def bookmark(self, memory: MemoryCandidate) -> dict:
        """生成书签：{path, start_line, end_line, snapshot_text, created_at}（FR-8）。"""
        return {
            "path": memory.path,
            "start_line": memory.start_line,
            "end_line": memory.end_line,
            "snapshot_text": memory.text,
            "created_at": _now_ms(),
        }


def _now_ms() -> float:
    import time
    return time.time() * 1000.0
