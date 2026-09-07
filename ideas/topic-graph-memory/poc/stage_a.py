"""POC Stage A：选源（01_idea_draft.md §5 的最小实现，纯离线、零 LLM）。

- G1  图 + 均匀：邻域全部源，配额均分，不打分
- G2  图 + 单源：只取边强度最高的 1 个源（= 主副线等价物）
- G3  多源 + 四维：top-k 源，配额均匀
- G4  full-ours：top-k 源，src_score = λ·strength + (1−λ)·rel，配额按 softmax(src_score)，
      配额语义 = **填充式**（每源保底占坑）
- G4C 变体：同 G4 选源，但配额语义 = **封顶式**（全局排序，每源最多占 quota 条）
      —— 实测与 G4 完全等价（Σquota=K 且池够大时数学上必然），保留作记录
- G4S 变体：**软偏置**——不做硬配额，源分以加性 bonus 并入记忆总分后全局 top-K
      —— 第三种"边参与计算"的语义，让强源自然多贡献而弱源不占死坑位
"""
from __future__ import annotations

import math

from graph_gen import PocEdge, PocNode


def incoming_sources(cur: str, edges: list[PocEdge]) -> list[PocEdge]:
    """当前节点的信息来源 = 指向它的边（src → dst 语义：dst 依赖 src）。"""
    return [e for e in edges if e.dst == cur]


def _softmax(xs: list[float], tau: float) -> list[float]:
    m = max(xs)
    exps = [math.exp((x - m) / tau) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def _quota_from_probs(probs: list[float], K: int) -> list[int]:
    """按比例分 K 条，保底 1，再归一化回 Σ=K。"""
    raw = [max(1, round(K * p)) for p in probs]
    total = sum(raw)
    quota = [max(1, round(K * r / total)) for r in raw]
    while sum(quota) > K:
        quota[quota.index(max(quota))] -= 1
    while sum(quota) < K:
        quota[quota.index(min(quota))] += 1
    return quota


def select_sources(
    cur: str,
    nodes: dict[str, PocNode],
    edges: list[PocEdge],
    rel_by_src: dict[str, float],
    group: str,
    K: int = 10,
    topk: int = 3,
    lam: float = 0.7,
    tau: float = 0.5,
) -> dict[str, int]:
    """返回 {node_id: quota}。rel_by_src = {src_id: cos(emb(topic), emb(q))}。"""
    src_edges = incoming_sources(cur, edges)
    if not src_edges:
        return {}

    if group == "G1":                                   # 图 + 均匀（无算法下界）
        per = K // len(src_edges) or 1
        return {e.src: per for e in src_edges}

    if group == "G2":                                   # 图 + 单源（主副线等价物）
        best = max(src_edges, key=lambda e: e.strength)
        return {best.src: K}

    ranked = sorted(src_edges, key=lambda e: e.strength, reverse=True)[:topk]
    if group == "G3":                                   # 多源 + 均匀配额
        return {e.src: q for e, q in zip(ranked, _quota_from_probs([1.0] * len(ranked), K))}

    if group in ("G4", "G4C"):                          # 多源 + 静态强度×query 相关度
        scores = [lam * e.strength + (1 - lam) * rel_by_src.get(e.src, 0.0) for e in ranked]
        probs = _softmax(scores, tau)
        return {e.src: q for e, q in zip(ranked, _quota_from_probs(probs, K))}

    if group == "G4S":                                  # 软偏置：quota 字段不用于硬分配
        scores = [lam * e.strength + (1 - lam) * rel_by_src.get(e.src, 0.0) for e in ranked]
        return {e.src: round(s, 4) for e, s in zip(ranked, scores)}

    raise ValueError(f"unknown group: {group}")
