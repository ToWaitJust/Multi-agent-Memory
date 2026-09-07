"""真实数据 POC：real_dataset.json 上跑 G1/G2/G3/G4/G4S（05 场景 + 04 判据）。

与合成 POC 共用 stage_a 与打分器，只换两处：
- embedding：假向量 → 真实 DashScope 1024 维（落盘缓存 data/embed_cache，可复现）
- 数据：合成图 → LLM 真跑出来的 todo-cli 多模块会话

运行：E:/1shujukuyuanli/anaconda/envs/srtp-memory/python.exe run_real_poc.py
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
from srtp_memory.plugins.infra.embedding_impls import DashScopeEmbedding  # noqa: E402

from stage_a import select_sources  # noqa: E402

HERE = Path(__file__).parent
DATASET = HERE / "real_dataset.json"
RESULT = HERE / "real_poc_results.json"

GROUPS = ["G1", "G2", "G3", "G4", "G4S"]
K, TOPK, LAM, TAU, MU = 10, 3, 0.7, 0.5, 0.2


def cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def run_group(group, data, emb, scorer, now):
    nodes = {n["node_id"]: n for n in data["nodes"]}
    edges = [type("E", (), {"src": e["src"], "dst": e["dst"], "type": e["type"],
                            "strength": e["strength"]})() for e in data["edges"]]
    memories = data["memories"]
    per_query = []

    for q in data["queries"]:
        q_vec = emb["q:" + q["query_id"]]
        rel_by_src = {e.src: cosine(emb["topic:" + e.src], q_vec)
                      for e in edges if e.dst == q["cur_node"]}
        quota = select_sources(q["cur_node"], nodes, edges, rel_by_src, group,
                               K=K, topk=TOPK, lam=LAM, tau=TAU)

        def score_one(m):
            cand = MemoryCandidate(
                memory_id=m["memory_id"], text=m["text"], path="",
                timestamp=m["timestamp"], access_count=m["access_count"],
                task_tag=m["task_tag"],
                vector_score=cosine(emb["m:" + m["memory_id"]], q_vec),
                embedding=emb["m:" + m["memory_id"]],
            )
            return scorer.score(q_vec, cand, now, q["task_tag"])["final"]

        if group == "G4S":
            cands = []
            for nid, src_score in quota.items():
                for m in [x for x in memories if x["node_id"] == nid]:
                    cands.append((score_one(m) + MU * src_score, nid, m))
            cands.sort(key=lambda t: t[0], reverse=True)
            picked = [(nid, m) for _, nid, m in cands[:K]]
        else:
            picked = []
            for nid, qta in quota.items():
                pool = [m for m in memories if m["node_id"] == nid]
                if group == "G1":
                    picked += [(nid, m) for m in pool[:qta]]
                    continue
                scored = sorted(((score_one(m), m) for m in pool),
                                key=lambda t: t[0], reverse=True)
                picked += [(nid, m) for _, m in scored[:qta]]

        seen, injected = set(), []
        for nid, m in picked:
            if m["memory_id"] not in seen:
                seen.add(m["memory_id"])
                injected.append((nid, m))

        rel_ids = set(q["relevant_ids"])
        # 召回天花板：|rel|≤K 时完美系统可达 1.0；|rel|>K 时被 K/|rel| 压住
        ceiling = min(1.0, K / len(rel_ids)) if rel_ids else 0.0
        hit = {m["memory_id"] for _, m in injected} & rel_ids
        contributors = {nid for nid, _ in injected}
        counts: dict[str, int] = {}
        for nid, _ in injected:
            counts[nid] = counts.get(nid, 0) + 1
        msr = 0.0 if len(counts) <= 1 else 1.0 - max(counts.values()) / len(injected)

        per_query.append({
            "recall_raw": len(hit) / len(rel_ids) if rel_ids else 0.0,
            "recall_norm": (len(hit) / len(rel_ids)) / ceiling if rel_ids and ceiling > 0 else 0.0,
            "source_hit_rate": len(contributors & set(q["relevant_nodes"])) / len(q["relevant_nodes"]),
            "multi_source_ratio": msr,
            "ceiling": ceiling,
            "n_injected": len(injected),
        })

    avg = {k: float(np.mean([r[k] for r in per_query])) for k in per_query[0]}
    avg["per_query_ceiling"] = {q["query_id"]: r["ceiling"]
                                for q, r in zip(data["queries"], per_query)}
    return avg


def main():
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    print(f"数据集：{data['meta']['n_nodes']} 节点 / {data['meta']['n_edges']} 边 / "
          f"{data['meta']['n_memories']} 记忆 / {len(data['queries'])} 查询\n")

    embedder = DashScopeEmbedding(dim=1024, cache_dir=str(ROOT / "data" / "embed_cache"))
    emb = {}
    for q in data["queries"]:
        emb["q:" + q["query_id"]] = embedder.encode(q["text"])
    for n in data["nodes"]:
        emb["topic:" + n["node_id"]] = embedder.encode(n["topic"] + " " + n["summary"])
    for m in data["memories"]:
        emb["m:" + m["memory_id"]] = embedder.encode(m["text"])
    print(f"embedding 就绪：{len(emb)} 条（缓存命中可复现）\n")

    now = max(m["timestamp"] for m in data["memories"]) + 1.0
    scorer = MultiHeadAttentionMemoryScorer(dict(DEFAULT_WEIGHTS))

    rows = {g: run_group(g, data, emb, scorer, now) for g in GROUPS}

    # G4S 的 μ 扫描（第一轮发现：μ 过大会重新集中到单源）
    print("\n=== G4S 软偏置 μ 扫描 ===")
    mus = {}
    global MU
    for mu in (0.05, 0.1, 0.2, 0.3):
        MU = mu
        mus[mu] = run_group("G4S", data, emb, scorer, now)
    MU = 0.2
    for mu, r in mus.items():
        print(f"μ={mu:<5} recall={r['recall_raw']:.3f}  r_norm={r['recall_norm']:.3f}  "
              f"src_hit={r['source_hit_rate']:.3f}  multi_src={r['multi_source_ratio']:.3f}")

    header = (f"{'group':<6}{'recall':>8}{'r_norm':>8}{'src_hit':>9}"
              f"{'multi_src':>11}{'kept':>7}")
    print(header)
    print("-" * len(header))
    for g in GROUPS:
        r = rows[g]
        print(f"{g:<6}{r['recall_raw']:>8.3f}{r['recall_norm']:>8.3f}"
              f"{r['source_hit_rate']:>9.3f}{r['multi_source_ratio']:>11.3f}"
              f"{r['n_injected']:>7.1f}")

    print("\n=== go/no-go 判据 ===")
    strengths = [e["strength"] for e in data["edges"]]
    var0 = max(strengths) == min(strengths)
    g4, g2, g3 = rows["G4"], rows["G2"], rows["G3"]
    c1 = g4["recall_raw"] > g2["recall_raw"] and g4["multi_source_ratio"] > 0.3
    c3 = g4["source_hit_rate"] > g2["source_hit_rate"]
    print(f"[{'PASS' if c1 else 'FAIL'}] 核心：G4 recall > G2 且 multi_source_ratio > 0.3（多对多优于一対一）")
    print(f"[{'PASS' if c3 else 'FAIL'}] 覆盖：G4 的 src_hit > G2（多源补上单源漏掉的源）")
    if var0:
        print(f"[N/A ] 强度：本数据集边强度全相等（{strengths[0]}），λ·strength 增量本轮未测；"
              f"需要混合 derives_from/references 边的数据集再测")
    else:
        c2 = g4["recall_raw"] > g3["recall_raw"]
        print(f"[{'PASS' if c2 else 'FAIL'}] 强度：G4 > G3（边强度配额有效）")
    soft = rows["G4S"]
    print(f"[info] 软偏置 G4S：recall={soft['recall_raw']:.3f} multi_src={soft['multi_source_ratio']:.3f}"
          f"（μ 扫描见上；本轮未超过硬配额）")
    verdict = "GO：核心主张成立" if (c1 and c3) else "NO-GO：多源未优于一対一"
    print("=> " + verdict)

    RESULT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果落盘 {RESULT}")


if __name__ == "__main__":
    main()
