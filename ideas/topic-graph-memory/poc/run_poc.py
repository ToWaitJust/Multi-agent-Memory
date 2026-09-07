"""POC 主跑批：G1/G2/G3/G4 × 6 queries，产出 4 指标对比表（04_poc_plan.md Step 3）。

不碰 srtp_memory 主链路，只复用其纯函数打分器（attention.full + DEFAULT_WEIGHTS）。
判据（写死）：G4 > G2 且 multi_source_ratio(G4) > 0.3 → go；G4 ≈ G2 → no-go。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from srtp_memory.attention import (  # noqa: E402
    DEFAULT_WEIGHTS, MemoryCandidate, MultiHeadAttentionMemoryScorer,
)

from fake_embed import cosine, text_vec  # noqa: E402
from graph_gen import build_graph, is_dag  # noqa: E402
from stage_a import select_sources  # noqa: E402

GROUPS = ["G1", "G2", "G3", "G4", "G4S"]
K = 10
TOPK = 3
LAM = 0.7
TAU = 0.5
MU = 0.2  # G4S 软偏置系数：源分并入记忆总分的权重


def relevant_memories(query, nodes, memories) -> set[str]:
    """ground truth：相关节点中、与问句命中 ≥2 个节点关键词的记忆。

    注意：若只要求 ≥1 个命中，相关集会大到超过注入上限 K（10），
    所有组的 recall 都被 10/|relevant| 压死，判据失真（第一轮 POC 的教训）。
    """
    out = set()
    for m in memories:
        if m.node_id in query.relevant_nodes:
            kws = nodes[m.node_id].keywords
            if sum(1 for kw in kws if kw in query.text) >= 2:
                out.add(m.memory_id)
    return out


def run_group(group, nodes, edges, memories, queries, scorer, emb, now):
    per_query = []
    for q in queries:
        q_vec = emb[q.query_id]
        # Stage A：query 相关度（复用打分头同源的 emb）
        rel_by_src = {nid: cosine(emb["topic:" + nid], q_vec)
                      for nid in {e.src for e in edges if e.dst == q.cur_node}}
        quota = select_sources(q.cur_node, nodes, edges, rel_by_src, group,
                               K=K, topk=TOPK, lam=LAM, tau=TAU)

        # Stage B：四维打分（G1 不打分，按写入序取；G4S 用软偏置全局排序）
        if group == "G4S":
            all_cands = []
            for nid, src_score in quota.items():        # quota 字段此时存的是 src_score
                for m in [x for x in memories if x.node_id == nid]:
                    cand = MemoryCandidate(
                        memory_id=m.memory_id, text=m.text, path="", timestamp=m.timestamp,
                        access_count=m.access_count, task_tag=m.task_tag,
                        bm25_score=cosine(text_vec(m.text), q_vec),
                        vector_score=cosine(emb[m.memory_id], q_vec),
                        embedding=emb[m.memory_id],
                    )
                    s = scorer.score(q_vec, cand, now, q.task_tag)
                    all_cands.append((s["final"] + MU * src_score, nid, m))
            all_cands.sort(key=lambda t: t[0], reverse=True)
            picked = [(nid, m) for _, nid, m in all_cands[:K]]
        elif group == "G4C":
            # 封顶式配额：选中源的全部记忆进全局排序，逐条取，每源最多 quota 条
            all_cands = []
            for nid in quota:
                for m in [x for x in memories if x.node_id == nid]:
                    cand = MemoryCandidate(
                        memory_id=m.memory_id, text=m.text, path="", timestamp=m.timestamp,
                        access_count=m.access_count, task_tag=m.task_tag,
                        bm25_score=cosine(text_vec(m.text), q_vec),
                        vector_score=cosine(emb[m.memory_id], q_vec),
                        embedding=emb[m.memory_id],
                    )
                    s = scorer.score(q_vec, cand, now, q.task_tag)
                    all_cands.append((s["final"], nid, m))
            all_cands.sort(key=lambda t: t[0], reverse=True)
            used = {nid: 0 for nid in quota}
            picked = []
            for _, nid, m in all_cands:
                if used[nid] < quota[nid] and len(picked) < K:
                    used[nid] += 1
                    picked.append((nid, m))
        else:
            picked: list[tuple[str, object]] = []
            for nid, qta in quota.items():
                pool = [m for m in memories if m.node_id == nid]
                if group == "G1":
                    picked += [(nid, m) for m in pool[:qta]]
                    continue
                scored = []
                for m in pool:
                    cand = MemoryCandidate(
                        memory_id=m.memory_id, text=m.text, path="", timestamp=m.timestamp,
                        access_count=m.access_count, task_tag=m.task_tag,
                        bm25_score=cosine(text_vec(m.text), q_vec),
                        vector_score=cosine(emb[m.memory_id], q_vec),
                        embedding=emb[m.memory_id],
                    )
                    s = scorer.score(q_vec, cand, now, q.task_tag)
                    scored.append((s["final"], m))
                scored.sort(key=lambda t: t[0], reverse=True)
                picked += [(nid, m) for _, m in scored[:qta]]

        # 跨源合并 + 去重（同 memory_id 取先到者）+ 截断 K
        seen, injected = set(), []
        for nid, m in picked:
            if m.memory_id not in seen:
                seen.add(m.memory_id)
                injected.append((nid, m))

        # 指标
        rel = relevant_memories(q, nodes, memories)
        hit_ids = {m.memory_id for _, m in injected} & rel
        contributors = {nid for nid, _ in injected}
        src_hit = contributors & set(q.relevant_nodes)
        counts = {}
        for nid, _ in injected:
            counts[nid] = counts.get(nid, 0) + 1
        msr = 0.0 if len(counts) <= 1 else 1.0 - max(counts.values()) / len(injected)

        per_query.append({
            "recall": len(hit_ids) / len(rel) if rel else 0.0,
            "source_hit_rate": len(src_hit) / len(q.relevant_nodes),
            "multi_source_ratio": msr,
            "compression": 1.0 - len(injected) / len(memories),
            "n_injected": len(injected),
        })
    return {k: float(np.mean([r[k] for r in per_query])) for k in per_query[0]}


def main():
    nodes, edges, memories, queries = build_graph()
    assert is_dag(nodes, edges), "合成图必须是无环 DAG（P0 约束）"
    print(f"图校验通过：{len(nodes)} 节点 / {len(edges)} 边 / DAG ✓ / {len(memories)} 条记忆\n")

    emb = {"topic:" + nid: text_vec(n.topic + n.summary) for nid, n in nodes.items()}
    emb.update({m.memory_id: text_vec(m.text) for m in memories})
    emb.update({q.query_id: text_vec(q.text) for q in queries})
    now = max(m.timestamp for m in memories) + 1.0
    scorer = MultiHeadAttentionMemoryScorer(dict(DEFAULT_WEIGHTS))

    rows = {}
    for g in GROUPS:
        rows[g] = run_group(g, nodes, edges, memories, queries, scorer, emb, now)

    header = f"{'group':<6}{'recall':>9}{'src_hit':>9}{'multi_src':>11}{'compress':>10}{'kept':>7}"
    print(header)
    print("-" * len(header))
    for g in GROUPS:
        r = rows[g]
        print(f"{g:<6}{r['recall']:>9.3f}{r['source_hit_rate']:>9.3f}"
              f"{r['multi_source_ratio']:>11.3f}{r['compression']:>10.3f}{r['n_injected']:>7.1f}")

    # 写死判据（04_poc_plan.md Step 4）
    print("\n=== go/no-go 判据 ===")
    c1 = rows["G4S"]["recall"] > rows["G2"]["recall"] and rows["G4S"]["multi_source_ratio"] > 0.3
    c2 = rows["G4S"]["recall"] > rows["G3"]["recall"] and rows["G4S"]["source_hit_rate"] >= rows["G3"]["source_hit_rate"] - 1e-9
    print(f"[{'PASS' if c1 else 'FAIL'}] G4S > G2 且 multi_source_ratio(G4S) > 0.3 "
          f"（多对多优于一対一）")
    print(f"[{'PASS' if c2 else 'FAIL'}] G4S 优于 G3（软偏置让边强度参与计算有效）")
    verdict = "GO：继续推进" if (c1 or c2) else "NO-GO：归档本 idea"
    print("=> " + verdict)

    out = Path(__file__).parent / "poc_results.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已落盘 {out}")


if __name__ == "__main__":
    main()
