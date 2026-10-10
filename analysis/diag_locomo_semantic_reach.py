"""诊断：语义选源（query × session topic）的可达性上界（D-051 方法侧修复测算）。

为什么需要这个诊断：
    Jaccard 建边下，相关源的名次分布**几乎是均匀的**（diag_locomo_edge_reach 实测），
    要到 89% 覆盖率得连 15/19 = 79% 的边 → 等于全连接 → **数据侧加宽是死路**。
    本质原因：静态图 top-k 邻域覆盖不了 **query-dependent** 的答案位置。

本脚本回答：
    **如果把选源改成「query 与全部 session topic 做语义检索」，覆盖率能到多少？**
    这是"图邻域 + 语义扩展"这一方法侧修复的理论上界。

    注意：这不等于放弃图 —— 图邻域仍可提供 typed 边强度（derives_from 等硬关系），
    语义只是把**候选源池**从"邻域内 3 个"扩展成"全图 + 语义排序"。

用法：
    python -m analysis.diag_locomo_semantic_reach --dataset data/locomo_conv0.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _cos(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class _Embedder:
    """带落盘缓存的 DashScope embedding（与生产链路同一实现，保证可比）。"""

    def __init__(self, cache_dir: str):
        # 复用生产链路的 DashScopeEmbedding（自带落盘缓存，路径/格式与线上一致）
        from srtp_memory.plugins.infra.embedding_impls import DashScopeEmbedding
        self.miss = 0
        try:
            self._inner = DashScopeEmbedding(cache_dir=str(ROOT / cache_dir))
        except Exception as exc:          # noqa: BLE001
            print(f"[warn] embedding 后端不可用: {exc}")
            self._inner = None

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self._inner is None:
            return [[] for _ in texts]
        before = len(getattr(self._inner, "_cache", {}) or {})
        vecs = [self._inner.encode(t) for t in texts]     # 单条接口，逐条编码
        after = len(getattr(self._inner, "_cache", {}) or {})
        self.miss += max(0, after - before)
        return [list(v) for v in vecs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/locomo_conv0.jsonl")
    ap.add_argument("--cache-dir", default="data/embed_cache")
    args = ap.parse_args()

    eps = [json.loads(l) for l in open(ROOT / args.dataset, encoding="utf-8")
           if l.strip()]
    print(f"episodes = {len(eps)}")

    nodes = {n["node_id"]: n["topic"] for n in eps[0]["nodes"]}
    node_ids = sorted(nodes)
    print(f"session 节点 = {len(node_ids)}  cur_node = {eps[0]['cur_node']}")

    emb = _Embedder(args.cache_dir)
    topic_vecs = emb.encode([nodes[n] for n in node_ids])
    if not any(topic_vecs):
        raise SystemExit("无法计算 topic embedding（缺 DASHSCOPE_API_KEY 或网络不通）")
    query_vecs = emb.encode([e["query"] for e in eps])
    print(f"embedding 缓存未命中 = {emb.miss}（已落盘）")

    # 逐个 episode：所有节点按 cos(query, topic) 排序，看相关源排第几
    print("\n=== 语义选源覆盖率（候选源池 = 全部 session，按 cos(query,topic) 排序）===")
    print(f"{'top-N':>6} {'可达 episode':>12} {'占比':>8}")
    for k in (1, 2, 3, 5, 8, 10, 15, len(node_ids)):
        hit = 0
        for e, qv in zip(eps, query_vecs):
            if not qv:
                continue
            order = sorted(node_ids, key=lambda n: -_cos(qv, topic_vecs[node_ids.index(n)]))
            if set(e["relevant_sources"]) & set(order[:k]):
                hit += 1
        print(f"{k:>6} {hit:>12} {hit / len(eps) * 100:>7.1f}%")

    # 相关源的名次分布（对照 Jaccard 的均匀分布）
    ranks = []
    for e, qv in zip(eps, query_vecs):
        if not qv:
            continue
        order = sorted(node_ids, key=lambda n: -_cos(qv, topic_vecs[node_ids.index(n)]))
        pos = {n: i + 1 for i, n in enumerate(order)}
        for s in e["relevant_sources"]:
            ranks.append(pos.get(s, 10**9))
    ranks.sort()
    if ranks:
        n = len(ranks)
        print(f"\n相关源的语义名次分布（共 {n} 个）:")
        buckets = [(1, 1), (2, 3), (4, 5), (6, 8), (9, 12), (13, 19), (20, 10**9)]
        for lo, hi in buckets:
            c = sum(1 for r in ranks if lo <= r <= hi)
            label = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+")
            bar = "#" * int(c / max(1, n) * 40)
            print(f"  第 {label:>4} 名: {c:>3} ({c / n * 100:>4.1f}%) {bar}")

    print("\n对照：Jaccard 建边 top-3 实测 src_hit = 0.2744；oracle = 0.9974")


if __name__ == "__main__":
    main()
