"""调度编排（§4.6，FR-5/6/7 总装配）+ 图架构 Stage A（优化方案 §4）。

MemoryCoordinator 持常驻完整池（resident_pool，per-user 真池）+ 检索插件（retriever）
+ 四维打分器 + 混合权重 + 选择器 + 共享池；按 (user_id, session_id) 隔离。
不持有 ReMeMiddleware（检索由 retriever 在常驻池上接管，D-9）。

V3.0（图架构）新增：
- **Stage A 选源**：`source_selector.select(cur_node, query_emb, K)` → 源集合 + 配额/偏置；
  候选召回被限定在 **邻域源节点的记忆并集**（大池按 `owner_node` 标记过滤，零拷贝）。
- **软偏置**：`final += μ · src_score(owner_node)`（`quota_mode=soft_bias`）。
- **硬配额**：先按每源配额预筛候选（`strength`/`uniform`），再走原选择器。
- **query_emb 透传**：修正既有缺陷 —— 原实现未把 `query_emb` 传给检索器，
  导致 `retriever.vector` 退化为按 `vector_score`（恒为 0）的稳定排序 = 等价于 `retriever.full`。
- `graph_enabled=False` 时**代码路径与改造前一致**（不进入 Stage A，不查 node_index）。
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Optional

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
    # ---- 图架构 / 三角均衡 新增字段（附加，不影响既有消费方）----
    graph_enabled: bool = False
    cur_node: str = "main"
    sources: list[dict] = field(default_factory=list)     # Stage A 源明细
    src_scores: dict[str, float] = field(default_factory=dict)
    scope_size: int = 0                                   # 邻域候选池规模
    scope_ratio: float = 0.0                              # kept / scope_size
    stage_a: dict = field(default_factory=dict)
    perf_tier: str = ""
    llm_calls: int = 0
    embed_calls_miss: int = 0
    injected_tokens: int = 0

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
            # ---- 图 / 均衡埋点 ----
            "graph_enabled": self.graph_enabled,
            "cur_node": self.cur_node,
            "sources": self.sources,
            "src_scores": self.src_scores,
            "scope_size": self.scope_size,
            "scope_ratio": self.scope_ratio,
            "stage_a": self.stage_a,
            "perf_tier": self.perf_tier,
            "llm_calls": self.llm_calls,
            "embed_calls_miss": self.embed_calls_miss,
            "injected_tokens": self.injected_tokens,
        }


class MemoryCoordinator:
    def __init__(self, resident_pool, retriever: BaseRetriever,
                 scorer: Optional[MultiHeadAttentionMemoryScorer] = None,
                 weights: Optional[HybridWeightCalculator] = None,
                 selector: Optional[ActionSelector] = None,
                 shared_pool: Optional[SharedMemoryPool] = None,
                 user_id: str = "", session_id: str = "",
                 logger=None,
                 source_selector=None, graph_cfg=None, budget=None):
        self.resident_pool = resident_pool
        self.retriever = retriever
        self.scorer = scorer or MultiHeadAttentionMemoryScorer()
        self.weights = weights or HybridWeightCalculator()
        self.selector = selector or ActionSelector()
        self.shared_pool = shared_pool or SharedMemoryPool()
        self.user_id = user_id
        self.session_id = session_id
        self.logger = logger  # ScheduleLogger（可选）
        # ---- 图架构（可为 None → 完全等价于改造前）----
        self.source_selector = source_selector
        self.graph_cfg = graph_cfg
        self.budget = budget
        self._accepts_query_emb = _accepts_query_emb(self.retriever)
        # 注入上限 K：优先取选择器自己的 max_shared；`selector.topk` 等桩实现没有该属性
        # （G0/G1 会用到），回退到 10（与 config.max_shared 默认一致）。
        self.injection_cap = int(getattr(self.selector, "max_shared", 0) or 10)

    @property
    def graph_enabled(self) -> bool:
        return self.source_selector is not None and bool(
            getattr(self.graph_cfg, "enabled", True))

    # ------------------------------------------------------------------

    def schedule(self, query: str, query_emb, task_tag: Optional[str],
                 now: float, enabled_dims: Optional[set[str]] = None,
                 condition: Optional[ConditionKey] = None,
                 top_k: int = 50, cur_node: str = "main",
                 quota_mode: Optional[str] = None,
                 quota_mu: Optional[float] = None,
                 manual_sources=None,
                 lambda_override: Optional[float] = None) -> ScheduleResult:
        """① (Stage A 选源) → ② 常驻池召回 → ③ 四维打分 → ④ 选择 → ⑤ 共享池 → ⑥ 埋点。"""
        res = ScheduleResult(query=query, query_emb=query_emb, task_tag=task_tag,
                             condition_key=condition, enabled_dims=enabled_dims,
                             cur_node=cur_node, graph_enabled=self.graph_enabled)
        cfg = self.graph_cfg
        gc = getattr(cfg, "graph", cfg) if cfg is not None else None   # 兼容传整配置

        # ---------------- Stage A：选源（图架构特有）----------------
        sel = None
        bonus: dict[str, float] = {}
        mode = quota_mode or (getattr(gc, "quota_mode", "soft_bias") if gc else "soft_bias")
        mu = float(quota_mu if quota_mu is not None
                   else (getattr(gc, "quota_mu", 0.2) if gc else 0.2))
        if self.graph_enabled:
            t0 = _now_ms()
            sel = self.source_selector.select(
                cur_node, _to_list(query_emb), K=self.injection_cap,
                mode=mode, lambda_override=lambda_override,
                manual_sources=manual_sources,
            )
            res.sources = [s.to_dict() for s in sel.sources]
            res.src_scores = {s.node_id: round(s.src_score, 6) for s in sel.sources}
            res.stage_a = sel.to_dict()
            res.latency_ms["stage_a"] = _now_ms() - t0
            if mode == "soft_bias":
                bonus = self.source_selector.bonus_map(sel, mu)

        # ---------------- ② 候选召回（限定邻域源，可选）----------------
        t0 = _now_ms()
        scope = self.resident_pool
        scope_nodes: list[str] = []
        if self.graph_enabled and sel is not None and sel.sources:
            scope_nodes = list(sel.node_ids)
            fallback = bool(getattr(gc, "fallback_all", True)) if gc else True
            scope = self.resident_pool.visible_from(scope_nodes, fallback_all=fallback)
        res.scope_size = _safe_len(scope)
        scoped = self.graph_enabled and bool(scope_nodes)
        candidates = self._recall(query, scope, top_k, query_emb, scoped=scoped)
        res.candidates = candidates
        res.latency_ms["recall"] = _now_ms() - t0

        # 硬配额：按每源配额预筛（填充式；POC 已证与封顶式数学等价）
        # 无源时不预筛（否则会把候选集清空 —— 冷启动期图还没边，必须退回不限范围）
        if (self.graph_enabled and mode in ("strength", "uniform")
                and sel is not None and sel.sources):
            t0 = _now_ms()
            candidates = _apply_hard_quota(candidates, sel)
            res.latency_ms["quota"] = _now_ms() - t0

        # ---------------- ③ 混合权重（先验由中间件提前 set_prior）--------
        t0 = _now_ms()
        cond = condition or ConditionKey()
        weights = self.weights.calculate(query_emb, cond)
        res.weights = weights
        res.latency_ms["weights"] = _now_ms() - t0

        # ---------------- ② 四维打分 + 重算 final ----------------
        t0 = _now_ms()
        for c in candidates:
            sc = self.scorer.score(query_emb, c, now, task_tag, enabled_dims=enabled_dims)
            total = sum(weights[d] * sc[d] for d in weights) / (sum(weights.values()) or 1.0)
            owner = getattr(c, "owner_node", "main") or "main"
            b = bonus.get(owner, 0.0)
            if b:
                total += b
            sc["final"] = total
            sc["src_score"] = res.src_scores.get(owner, 0.0)
            res.scored.append({"memory": c, "score": sc, "via_source": owner})
        res.latency_ms["scoring"] = _now_ms() - t0

        # ---------------- ④ 动作选择 ----------------
        t0 = _now_ms()
        sel_res = self.selector.select(res.scored)
        res.kept = [m["memory"] for m in sel_res["kept"]]
        res.discarded = [m["memory"] for m in sel_res["discarded"]]
        res.compression_ratio = self.selector.compression_ratio(sel_res)
        res.scope_ratio = len(res.kept) / res.scope_size if res.scope_size else 0.0
        res.latency_ms["select"] = _now_ms() - t0

        # ---------------- ⑤ 写入共享池（频率计数同步到常驻池）------------
        final_by_id = {s["memory"].memory_id: s["score"]["final"] for s in res.scored}
        for m in res.kept:
            self.shared_pool.add({
                "memory_id": m.memory_id, "text": m.text, "path": m.path,
                "start_line": m.start_line, "end_line": m.end_line,
                "score": final_by_id.get(m.memory_id, 0.0),
                "timestamp": m.timestamp,
                "owner_node": getattr(m, "owner_node", "main"),   # 图架构附加字段
            })
            self.resident_pool.mark_access(m.memory_id)

        res.latency_ms["total"] = sum(res.latency_ms.values())

        # ---------------- 成本计数 ----------------
        if self.budget is not None:
            res.perf_tier = self.budget.tier
            c = self.budget.counters()
            res.llm_calls, res.embed_calls_miss = c["llm_calls"], c["embed_calls_miss"]
            res.injected_tokens = _estimate_tokens([m.text for m in res.kept])

        # ---------------- ⑥ L3 埋点 ----------------
        if self.logger is not None:
            self.logger.log_schedule(res.to_dict())
        return res

    # ------------------------------------------------------------------

    def _recall(self, query: str, scope, top_k: int, query_emb, scoped: bool = False):
        """召回。

        - **限定范围（图开启且有源）**：走范围内**精确余弦** —— 因为检索器绑定的
          faiss 索引是全池的，用它会静默丢掉范围外排位靠后的范围内条目（实测缺陷）。
        - **不限范围（含 graph.enabled=False 的全部路径）**：对支持 `query_emb` 的
          检索器透传（修正原实现未透传导致 `retriever.vector` 退化为写入序的问题）。
        """
        if scoped:
            from .graph.view import exact_in_scope_recall
            return exact_in_scope_recall(query_emb, scope.all(), top_k)
        if self._accepts_query_emb:
            return self.retriever.recall(query, scope, top_k=top_k, query_emb=query_emb)
        return self.retriever.recall(query, scope, top_k=top_k)

    def dispatch(self, sub_agent, top_k: int = 10) -> list[dict]:
        """共享池 top() 全量（≤10）格式化后交副线中间件注入 HintBlock。"""
        return self.shared_pool.top()[:top_k]

    def bookmark(self, memory: MemoryCandidate) -> dict:
        """生成书签：{path, start_line, end_line, snapshot_text, created_at}（FR-8）。"""
        return {
            "path": memory.path,
            "start_line": memory.start_line,
            "end_line": memory.end_line,
            "snapshot_text": memory.text,
            "created_at": _now_ms(),
        }


# ------------------------------------------------------------------ helpers


def _accepts_query_emb(retriever) -> bool:
    try:
        params = inspect.signature(retriever.recall).parameters
    except (TypeError, ValueError):
        return False
    return "query_emb" in params


def _safe_len(scope) -> int:
    try:
        return int(len(scope.all()))
    except Exception:  # noqa: BLE001 - 视图统计失败不影响主链路
        return 0


def _to_list(vec) -> Optional[list[float]]:
    if vec is None:
        return None
    if isinstance(vec, list):
        return vec
    tolist = getattr(vec, "tolist", None)
    if callable(tolist):
        try:
            return list(tolist())
        except Exception:  # noqa: BLE001
            return None
    try:
        return [float(x) for x in vec]
    except Exception:  # noqa: BLE001
        return None


def _apply_hard_quota(candidates: list[MemoryCandidate], sel) -> list[MemoryCandidate]:
    """硬配额预筛：每源取前 quota 条（按原始召回序，源内已由检索器排序）。

    返回顺序 = 源顺序（边强度/源分降序），保持"先选源再选记忆"的可解释性。
    """
    by_node: dict[str, list[MemoryCandidate]] = {}
    for c in candidates:
        by_node.setdefault(getattr(c, "owner_node", "main") or "main", []).append(c)
    out: list[MemoryCandidate] = []
    seen: set[str] = set()
    for s in sel.sources:
        picked = by_node.get(s.node_id, [])[: max(1, int(s.quota))]
        for c in picked:
            if c.memory_id not in seen:
                seen.add(c.memory_id)
                out.append(c)
    return out


def _estimate_tokens(texts: list[str]) -> int:
    """粗略 token 估算（中文 ~1.5 字/token）。仅用于成本埋点，不作计费依据。"""
    n = sum(len(t or "") for t in texts)
    return int(n / 1.5)


def _now_ms() -> float:
    import time
    return time.time() * 1000.0
