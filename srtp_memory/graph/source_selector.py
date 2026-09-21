"""Stage A：选源（§4.1）—— 图相对主副线**唯一新增的算法含量**。

管线：
    邻域展开(depth) → 版本折叠 → 环降级边排除 → 源打分 → 源裁剪(topk) → 配额/偏置

源打分（**strength 静态 + relevance 动态**，2026-09-07 定案）：
    src_score = λ · strength(edge(n_cur → s_i)) + (1 − λ) · cos(emb(topic_i), emb(q))
- `strength` 建边时一次算好，**本模块只读不写**（不 per-query 重算）。
- `emb(q)` 复用 Stage B 语义头本来要算的那一份 → 增量只有 m 次内积，亚毫秒，**零 LLM 成本**。
- 退化保护：各源 relevance 方差 < 阈值 → λ 置 1，退回纯静态。
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

from .graph_store import GraphStore
from .models import SourceRef, SourceSelection, cosine

QUOTA_MODES = ("none", "uniform", "strength", "soft_bias")


class SourceSelector:
    """Stage A 选源器。纯 CPU、可单测、零 LLM。"""

    def __init__(self, store: GraphStore, cfg=None):
        self.store = store
        self.cfg = cfg or _DefaultCfg()

    # ------------------------------------------------------------------

    def select(self, cur_node: str, query_emb: Optional[list[float]],
               K: int = 10, mode: Optional[str] = None,
               lambda_override: Optional[float] = None,
               manual_sources: Optional[Iterable[str]] = None) -> SourceSelection:
        """产出源集合与配额/权重。

        - `manual_sources` 非空 → `source_mode=manual`：用户直接指定源（上限仍是 source_topk）
        - `mode` 覆盖配置的 `quota_mode`
        - `lambda_override` 供均衡档 B3「纯静态」使用
        """
        c = self.cfg
        mode = mode or getattr(c, "quota_mode", "soft_bias")
        if mode not in QUOTA_MODES:
            raise ValueError(f"未知 quota_mode: {mode}（可用 {QUOTA_MODES}）")

        folded = self.store.fold_versions()
        cur = folded.get(cur_node, cur_node)

        sel = SourceSelection(cur_node=cur_node, mode=mode, depth=int(getattr(c, "depth", 1)))
        sel.folded = {k: v for k, v in folded.items() if k != v}

        # ① 邻域展开（沿**入边**取上游源节点；排除环降级边；strength_min 过滤弱边）
        #   邻域**不**包含 cur 自身 —— 自身记忆由 Stage B 直接可见，
        #   若把自身当源会与"多源"语义混淆（G2 单源时就成了"只看自己"，失去对照意义）。
        neigh = self.store.predecessors(
            cur, depth=int(getattr(c, "depth", 1)),
            strength_min=float(getattr(c, "strength_min", 0.0)),
            edge_types=getattr(c, "edge_types", None),
        )
        # 记录被 P0 环检测降级的边（溯源可见，不入邻域）
        sel.rejected_edges = [
            e.edge_id for e in self.store.edges()
            if not e.in_neighborhood and (e.src == cur or e.dst == cur)
        ]

        # ② 源集合确定：manual 优先
        if manual_sources:
            wanted = list(dict.fromkeys(manual_sources))
            pool = {nid: e for nid, e in neigh}
            pairs = []
            for nid in wanted:
                real = folded.get(nid, nid)
                if real in pool:
                    pairs.append((real, pool[real]))
                elif self.store.get_node(real) is not None:
                    # 手动指定的源即使没有 active 边也允许（视为 strength=0 的弱源）
                    pairs.append((real, None))
        else:
            pairs = list(neigh)

        if not pairs:
            sel.reason = "邻域为空（无 active 上游边）"
            return sel

        # ③ 源打分
        lam = float(lambda_override if lambda_override is not None
                    else getattr(c, "quota_lambda", 0.7))
        rels: list[float] = []
        for nid, _e in pairs:
            node = self.store.get_node(nid)
            r = cosine(node.emb if node else None, query_emb)
            rels.append(r if r is not None else 0.0)
        var = _variance(rels)
        degraded = False
        thr = float(getattr(c, "rel_var_threshold", 0.02))
        if var < thr:
            lam, degraded = 1.0, True          # 区分度不足 → 纯静态
        sel.lambda_ = lam
        sel.degraded_static = degraded

        refs: list[SourceRef] = []
        for (nid, e), rel in zip(pairs, rels):
            st = float(e.strength) if e is not None else 0.0
            refs.append(SourceRef(
                node_id=nid, strength=st, relevance=rel,
                src_score=lam * st + (1.0 - lam) * rel,
                via_edge=(e.edge_id if e is not None else ""),
            ))

        # ④ 源裁剪（top-k）
        refs.sort(key=lambda r: r.src_score, reverse=True)
        topk = int(getattr(c, "source_topk", 3))
        if topk > 0:
            refs = refs[:topk]
        if not refs:
            sel.reason = "源裁剪后为空"
            return sel

        # ⑤ 权重 / 配额
        tau = float(getattr(c, "quota_tau", 0.5)) or 0.5
        w = _softmax([r.src_score / tau for r in refs])
        for r, wi in zip(refs, w):
            r.weight = wi
        min_q = int(getattr(c, "min_quota", 1))
        if mode == "strength":
            _assign_hard_quota(refs, w, K, min_q)
        elif mode == "uniform":
            _assign_hard_quota(refs, [1.0 / len(refs)] * len(refs), K, min_q)
        # `none`（G0/G1：邻域全量、不做源偏向）与 `soft_bias` 均不设硬配额：
        #   - none      → 无配额、无 bonus（源集合只用于限定检索范围）
        #   - soft_bias → 无配额，改用 μ·src_score 加性偏置（由 coordinator 施加）

        nb = _hist([self.store.get_edge(r.via_edge).type for r in refs if r.via_edge])
        sel.sources = refs
        sel.reason = (
            f"邻域 {len(neigh)} 源 → top{topk} 取 {len(refs)}；"
            f"λ={lam:.2f}{'(退化纯静态)' if degraded else ''}；"
            f"rel_var={var:.4f}；边型 {nb}"
        )
        return sel

    # ------------------------------------------------------------------

    def bonus_map(self, sel: SourceSelection, mu: float) -> dict[str, float]:
        """软偏置：源分 → 加到记忆 total 上的加性 bonus（按 owner_node 查表）。

        `μ=0` 时返回空 dict（等价于不做源偏向）。
        """
        if mu <= 0.0 or not sel.sources:
            return {}
        return {s.node_id: mu * s.src_score for s in sel.sources}


# ------------------------------------------------------------------ helpers


class _DefaultCfg:
    """未传配置时的默认值（与 config/srtp.yaml 的 graph 段保持一致）。"""

    depth = 1
    source_topk = 3
    quota_mode = "soft_bias"
    quota_tau = 0.5
    quota_lambda = 0.7
    quota_mu = 0.2
    min_quota = 1
    strength_min = 0.0
    edge_types = None
    rel_var_threshold = 0.02


def _softmax(xs: list[float]) -> list[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def _variance(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def _assign_hard_quota(refs: list[SourceRef], weights: list[float], K: int,
                       min_q: int) -> None:
    """硬配额：`max(min_q, round(K·w))` 后按比例归一回 Σquota=K。

    ⚠️ 已知（POC 第一轮实测）：当 Σquota=K 且每个源池 ≥ quota 时，
    "填充式"与"封顶式"**数学等价** —— 即硬配额只有一种有效语义。
    这正是默认改用 `soft_bias` 的原因（见优化方案 §4.3）。
    """
    n = len(refs)
    if n == 0:
        return
    K = max(K, n * min_q)
    q = [max(min_q, int(round(K * w))) for w in weights]
    # 归一化到 Σ=K
    for _ in range(64):
        s = sum(q)
        if s == K:
            break
        if s > K:
            order = sorted(range(n), key=lambda i: q[i], reverse=True)
            for i in order:
                if s == K:
                    break
                if q[i] > min_q:
                    q[i] -= 1
                    s -= 1
            if all(x <= min_q for x in q):
                break
        else:
            order = sorted(range(n), key=lambda i: q[i], reverse=True)
            for i in order:
                if s == K:
                    break
                q[i] += 1
                s += 1
    for r, qi in zip(refs, q):
        r.quota = qi


def _hist(xs: list[str]) -> dict[str, int]:
    h: dict[str, int] = {}
    for x in xs:
        h[x] = h.get(x, 0) + 1
    return h
