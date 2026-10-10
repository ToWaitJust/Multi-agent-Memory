"""诊断：加宽建边 top-k 到底能救回多少相关源（D-051 数据侧修复的上界测算）。

背景：
    `locomo_adapter` 每个 session 只向"更早的 session"建 Jaccard top-3 边，
    且 cur_node 取最后一个 session → **入边邻居恒为 3**，`source_topk=3` 等于全选
    → `src_score` 从未参与选择（src_hit 恒 0.2744）。
    oracle 实测 recall 0.9029 vs 实际 0.2303，说明选源可达性是压倒性瓶颈。

本脚本回答一个决定修复方案的问题：
    **相关 session 在"与 cur_node 的相似度排序"里排第几？**
    如果普遍排在 top-8 内 → 加宽 edge_topk 就能救；
    如果排在 15 名开外 → 加宽无用，必须换建边策略（语义 embedding / 双向 / 多跳）。

用法：
    python -m analysis.diag_locomo_edge_reach --dataset data/locomo_conv0.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/locomo_conv0.jsonl")
    args = ap.parse_args()

    eps = [json.loads(l) for l in open(ROOT / args.dataset, encoding="utf-8")
           if l.strip()]
    print(f"episodes = {len(eps)}")

    # 每个 episode 的 corpus / nodes 都一样（同一对话），只算一次 session 文本
    sess_text: dict[str, set[str]] = {}
    for c in eps[0]["corpus"]:
        sess_text.setdefault(c["owner_node"], set()).update(_tokens(c["text"]))

    cur = eps[0]["cur_node"]
    # 与 cur_node 的相似度排序（所有其他 session）
    ranked = sorted(
        ((_jaccard(sess_text[cur], t), n) for n, t in sess_text.items() if n != cur),
        key=lambda x: -x[0])
    rank_of = {n: i + 1 for i, (_s, n) in enumerate(ranked)}
    n_sessions = len(sess_text)
    print(f"cur_node = {cur}   session 总数 = {n_sessions}")
    print(f"与 cur_node 相似度 top-10: "
          f"{[(n, round(s, 3)) for s, n in ranked[:10]]}")

    # 相关源在相似度排序中的名次（越小越靠前）
    ranks: list[int] = []
    for e in eps:
        for s in e["relevant_sources"]:
            if s == cur:
                ranks.append(0)          # 自身节点，无需边（但设计上自身不作为源）
            elif s in rank_of:
                ranks.append(rank_of[s])
    ranks.sort()
    dist = Counter(ranks)
    print(f"\n相关源的相似度名次分布（0 = 就是 cur_node 自身，共 {len(ranks)} 个相关源）:")
    for r in sorted(dist)[:20]:
        print(f"  第 {r:>2} 名: {dist[r]} 个")

    # 不同 edge_topk 下的覆盖率（一个 episode 只要有一个相关源进邻域就算"可达"）
    print("\n=== 加宽 edge_topk 的可达性上界 ===")
    print(f"{'topk':>5} {'可达 episode':>12} {'占比':>8}   （含 cur_node 自身的情况）")
    for k in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, n_sessions):
        hit = 0
        for e in eps:
            reach = {n for n in set(e["relevant_sources"])
                     if n == cur or rank_of.get(n, 10**9) <= k}
            if reach:
                hit += 1
        print(f"{k:>5} {hit:>12} {hit / len(eps) * 100:>7.1f}%")

    # 对照：当前实际 source_hit
    print("\n对照：当前实测 source_hit = 0.2744（topk=3 且邻域恒为 3）")
    print("若上界显著高于 0.2744 → 加宽建边有效；若上界也低 → 必须换建边策略")


if __name__ == "__main__":
    main()
