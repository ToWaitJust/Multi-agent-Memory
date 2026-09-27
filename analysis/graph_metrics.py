"""图架构新增指标（优化方案 §6.4）—— 缺失则 G3/G4 无法证明任何东西。

四个新指标 + 一个**必报项**：

| 指标 | 定义 | 为什么需要 |
|---|---|---|
| `source_hit_rate` | 人工标注的"应参考源节点"中被 Stage A 实际选中的比例 | 直接衡量选源质量（Q2 主判据） |
| `multi_source_ratio` | 注入集中来自 >=2 个不同源节点的条目占比 | 证明"多对多"真的发生，而非退化成单源 |
| `cross_source_relevance` | 注入集每条记忆与其源节点的平均相关分 | 防止为了凑覆盖率拉进无关记忆 |
| `dedup_rate` | 多路径去重掉的重复条目占比 | 衡量多路径冗余 |
| **`ceiling`（必报）** | `min(1, K / |relevant|)`，K = 注入上限 | POC 教训：不报天花板会把 `K/|rel|` 的物理上限误读为方法优势 |

注意（本轮实测发现）：`owner_node` 采用**单值标记**（决策 D-013），
每个源节点池天然互斥，因此 `dedup_rate` **结构性恒为 0**。
指标保留（多值标记或"记忆继承"方案下才有意义），但论文中不应把它当证据使用。

本模块只做纯函数计算，不依赖 srtp_memory（便于单测与复用）。
"""
from __future__ import annotations


def recall_at_kept(kept_ids: list[str], relevant: list[str]) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    return len(rel & set(kept_ids)) / len(rel)


def ceiling(relevant: list[str], injection_cap: int) -> float:
    """召回天花板：相关集大于注入上限时，recall 物理上不可能到 1。"""
    n = len(relevant)
    if n == 0:
        return 0.0
    return min(1.0, injection_cap / n)


def recall_normalized(recall: float, ceil: float) -> float:
    return round(recall / ceil, 6) if ceil > 0 else 0.0


def source_hit_rate(selected_sources: list[str], relevant_sources: list[str]) -> float:
    """应参考源节点中被选中的比例。无标注源 → NaN（不可评估，不参与平均）。"""
    rel = set(relevant_sources)
    if not rel:
        return float("nan")
    return len(rel & set(selected_sources)) / len(rel)


def multi_source_ratio(kept_owners: list[str]) -> float:
    """`1 − max_源占比`：单源时=0，来源越分散越大（POC 同口径，G2 恒为 0）。"""
    if not kept_owners:
        return 0.0
    counts: dict[str, int] = {}
    for o in kept_owners:
        counts[o] = counts.get(o, 0) + 1
    return 1.0 - max(counts.values()) / len(kept_owners)


def cross_source_relevance(kept_owners: list[str], src_scores: dict[str, float]) -> float:
    """注入记忆与其源节点 src_score 的均值（只统计能查到源分的条目）。"""
    vals = [src_scores[o] for o in kept_owners if o in src_scores]
    return round(sum(vals) / len(vals), 6) if vals else 0.0


def dedup_rate(n_after: int, n_before: int) -> float:
    """被路径去重掉的占比。`n_before` 为去重前条目总数。"""
    if n_before <= 0:
        return 0.0
    return round(max(0, n_before - n_after) / n_before, 6)


# ---------------------------------------------------------------------------
# rank-based 指标（S2' 方法学修正）
#
# **为什么需要**：recall@K 在 `|relevant| ≪ K` 时**天然不敏感** ——
# 只要相关记忆都进了候选且排名不太靠后，recall@K 就恒等于 1，
# 干扰项再强也"挤不掉"相关记忆（因为名额够多）。
# 本轮实测确认：干扰项把语义 top-4 变成 [相关, 干扰, 相关, 干扰]，
# 但 recall@8 仍 = 2/2 = 1.0 → **recall@8 对排序质量零区分度**。
#
# 补两个对**排序位置**敏感的标准指标：
#   - MRR      ：第一条相关记忆的倒数排名均值（对"最相关的排在第几"敏感）
#   - nDCG@K   ：分级增益折扣（对"相关记忆整体排多靠前"敏感，标准 IR 指标）
# ---------------------------------------------------------------------------


def cand_hit_rate(candidate_ids: list[str], relevant: list[str]) -> float:
    """候选命中率：相关记忆进候选池的比例（S-C 场景主指标）。

    这是"多对多选源"的**直接度量**：图选源的价值 = 把语义检索够不着的
    记忆（字面不相似、分散多源）送进候选池。候选都没进，后面排序再好也无用。
    """
    rel = set(relevant)
    if not rel:
        return 0.0
    return len(rel & set(candidate_ids)) / len(rel)


def mrr(kept_ids: list[str], relevant: list[str]) -> float:
    """平均倒数排名：kept 中第一条相关记忆排名的倒数；kept 内无相关 → 0。"""
    rel = set(relevant)
    for i, mid in enumerate(kept_ids, 1):
        if mid in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(kept_ids: list[str], relevant: list[str], k: int | None = None) -> float:
    """nDCG@K（二值增益）。kept 已按 final 降序 → 位置即排名。"""
    rel = set(relevant)
    if not rel:
        return 0.0
    k = len(kept_ids) if k is None else min(k, len(kept_ids))
    dcg = sum(1.0 / math_log2(i + 1) for i, mid in enumerate(kept_ids[:k], 1) if mid in rel)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math_log2(i + 1) for i in range(1, ideal_hits + 1))
    return round(dcg / idcg, 6) if idcg > 0 else 0.0


def math_log2(x: float) -> float:
    import math
    return math.log2(x)


def aggregate(rows: list[dict]) -> dict:
    """把一个 group 的所有 episode 行聚合成摘要（含天花板与多源占比）。"""
    n = len(rows)
    if n == 0:
        return {"n_episodes": 0}

    def avg(key: str) -> float:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def nanavg(key: str) -> float | None:
        vals = [r[key] for r in rows
                if isinstance(r.get(key), float) and r[key] == r[key]]   # 过滤 NaN
        return round(sum(vals) / len(vals), 4) if vals else None

    # 多源指标只在"多源样本"上算才有意义：单源样本的 multi_source_ratio 恒为 0，
    # 直接对全体求平均会被稀释（本轮实测 0.119 vs 多源子集 0.381）。
    multi_rows = [r for r in rows if len(r.get("relevant_sources") or []) >= 2]

    def avg_multi(key: str) -> float:
        vals = [r[key] for r in multi_rows if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "n_episodes": n,
        "n_multi_source_episodes": len(multi_rows),
        "avg_recall_kept": avg("recall_kept"),
        "avg_ceiling": avg("ceiling"),
        "avg_recall_norm": avg("recall_norm"),
        "avg_cand_hit_rate": avg("cand_hit_rate"),
        "avg_mrr": avg("mrr"),
        "avg_ndcg_at_k": avg("ndcg_at_k"),
        "avg_source_hit_rate": nanavg("source_hit_rate"),
        "avg_multi_source_ratio": avg("multi_source_ratio"),
        "avg_multi_source_ratio_multi": avg_multi("multi_source_ratio"),
        "avg_recall_kept_multi": avg_multi("recall_kept"),
        "avg_source_hit_rate_multi": avg_multi("source_hit_rate"),
        "avg_cross_source_relevance": avg("cross_source_relevance"),
        "avg_dedup_rate": avg("dedup_rate"),
        "avg_candidate_count": avg("candidate_count"),
        "avg_scope_size": avg("scope_size"),
        "avg_kept": avg("kept_count"),
        "avg_injected_tokens": avg("injected_tokens"),
        "avg_llm_calls": avg("llm_calls"),
        "avg_embed_calls_miss": avg("embed_calls_miss"),
        "avg_latency_ms": avg("latency_ms"),
        "p50_latency_ms": _pct([r["latency_ms"] for r in rows], 0.50),
        "p90_latency_ms": _pct([r["latency_ms"] for r in rows], 0.90),
        "n_relevant_mean": avg("n_relevant"),
        "n_sources_mean": avg("n_selected_sources"),
        "perf_tier": rows[0].get("perf_tier", ""),
    }


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return round(s[0], 2)
    idx = q * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return round(s[lo] * (1 - frac) + s[hi] * frac, 2)


def compare(a: dict, b: dict, key: str) -> float | None:
    """b − a 的差值（用于 G4−G1 / G4−G2 / G4−G3 判据）。"""
    va, vb = a.get(key), b.get(key)
    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
        return round(vb - va, 4)
    return None


def aggregate_by_scene(rows: list[dict]) -> dict:
    """按场景分组聚合（E1：S-A 考 time / S-B 考 task / fact 为事实型基线）。

    为什么要分组：总平均会把"组件在特定场景的增益"稀释掉——
    time/task 维只在对应场景有发挥空间，混在一起看等于白做。
    """
    out: dict[str, dict] = {}
    for sc in sorted({r.get("scene", "fact") for r in rows}):
        out[sc] = aggregate([r for r in rows if r.get("scene", "fact") == sc])
    return out


def judge(group_summaries: dict[str, dict]) -> dict:
    """自动判据（优化方案 §6.5）。返回 {criterion: {value, pass, note}}。"""
    g = group_summaries
    out: dict[str, dict] = {}

    def _get(name: str, key: str):
        return (g.get(name) or {}).get(key)

    # 结构主张：G4 > G2
    g4r, g2r = _get("graph_g4", "avg_recall_kept"), _get("graph_g2", "avg_recall_kept")
    g4s, g2s = _get("graph_g4", "avg_source_hit_rate"), _get("graph_g2", "avg_source_hit_rate")
    out["结构主张 G4>G2（recall）"] = {
        "value": None if None in (g4r, g2r) else round(g4r - g2r, 4),
        "pass": None if None in (g4r, g2r) else bool(g4r > g2r),
    }
    out["结构主张 G4>G2（source_hit）"] = {
        "value": None if None in (g4s, g2s) else round(g4s - g2s, 4),
        "pass": None if None in (g4s, g2s) else bool(g4s > g2s),
    }

    # 归因：G4 − G1 相对天花板 >= 5%
    g4n, g1n = _get("graph_g4", "avg_recall_norm"), _get("graph_g1", "avg_recall_norm")
    delta = None if None in (g4n, g1n) else round(g4n - g1n, 4)
    out["归因 G4−G1 >= 5%(天花板归一)"] = {
        "value": delta, "pass": None if delta is None else bool(delta >= 0.05),
        "note": "不通过则说明增益主要来自图结构而非调度算法（须在论文中如实说明）",
    }

    # 选源层：G4 − G3 在 src_hit 上 >= 8%
    g4h, g3h = _get("graph_g4", "avg_source_hit_rate"), _get("graph_g3", "avg_source_hit_rate")
    d2 = None if None in (g4h, g3h) else round(g4h - g3h, 4)
    out["选源层 G4−G3 >= 8%(source_hit)"] = {
        "value": d2, "pass": None if d2 is None else bool(d2 >= 0.08),
        "note": "不通过则边强度配额无用，Stage A 应简化为均匀",
    }

    # 需要调度：G1 > G0
    g1r, g0r = _get("graph_g1", "avg_recall_kept"), _get("graph_g0", "avg_recall_kept")
    d3 = None if None in (g1r, g0r) else round(g1r - g0r, 4)
    out["需要调度 G1>G0"] = {"value": d3, "pass": None if d3 is None else bool(d3 > 0)}

    # 速度
    p90 = _get("graph_g4", "p90_latency_ms")
    out["速度 G4 P90 < 1000ms"] = {
        "value": p90, "pass": None if p90 is None else bool(p90 < 1000)}

    # 多源确实发生（只在多源样本上判：单源样本该值恒为 0，会把均值稀释掉）
    msr = _get("graph_g4", "avg_multi_source_ratio_multi")
    out["多源生效 G4 multi_source_ratio(多源样本) > 0.3"] = {
        "value": msr, "pass": None if msr is None else bool(msr > 0.3),
        "note": "若接近 0 说明配额或去重有 bug，先修再谈结论",
    }
    return out
