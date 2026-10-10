"""最小复现：确认「语义源扩展」是否真的把节点加进了候选池（D-051）。

直接装配 middleware → 灌库 → 选源一次，打印：
  · 邻域节点（图边）
  · semantic_expanded（语义扩展补进来的）
  · 最终 selected sources 与各自 src_score
用于定位「为什么 src_hit 恒为 0.2744」。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/locomo_conv0.jsonl")
    ap.add_argument("--group", default="graph_g4_sem5")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    from srtp_memory.config import SchedulingConfig
    from srtp_memory.middleware import MemorySchedulingMiddleware

    eps = [json.loads(l) for l in open(ROOT / args.dataset, encoding="utf-8")
           if l.strip()]

    cfg = SchedulingConfig.load(ROOT / "config" / "ablation" / f"{args.group}.yaml")
    cfg.embedding_cache_dir = "data/embed_cache"
    cfg.budget.default_tier = "fast"
    cfg.graph.auto_link = False
    print(f"graph.semantic_source_topk = {cfg.graph.semantic_source_topk}")
    print(f"graph.source_topk        = {cfg.graph.source_topk}")
    print(f"graph.quota_lambda       = {cfg.graph.quota_lambda}")
    print(f"graph.rel_var_threshold  = {cfg.graph.rel_var_threshold}")

    for i in range(args.idx, min(args.idx + args.n, len(eps))):
        ep = eps[i]
        import tempfile, os
        tmp = tempfile.mkdtemp()
        mid = MemorySchedulingMiddleware(
            config=cfg, user_id=ep["user_id"], session_id="s",
            resident_pool_path=os.path.join(tmp, "rp.jsonl"),
            shared_pool_path=os.path.join(tmp, "sp.json"),
            log_path=os.path.join(tmp, "m.jsonl"),
            graph_dir=os.path.join(tmp, "graph"),
        )
        for nd in ep["nodes"]:
            node = mid.graph.create_node(topic=nd["topic"], summary=nd["summary"],
                                         node_id=nd["node_id"])
            node.emb = mid.edge_builder.embed(f"{nd['topic']} {nd['summary']}")
        for ed in ep["edges"]:
            mid.graph.add_edge(ed["src"], ed["dst"], ed["type"],
                               evidence="dataset:ground_truth_graph")
        for m in ep["corpus"]:
            mid.add_memory(text=m["text"], memory_id=m["memory_id"],
                           task_tag=m["task_tag"], node_id=m["owner_node"],
                           timestamp=m.get("timestamp"), persist=False)
        mid.resident_pool.persist()
        mid.use_node(ep["cur_node"])

        q_emb = mid.embedding.encode(ep["query"])
        sel = mid.source_selector.select(ep["cur_node"], q_emb, K=cfg.top_k,
                                         mode=cfg.graph.quota_mode)
        print(f"\n--- episode#{i}  {ep['scene']}  cur={ep['cur_node']}")
        print(f"    query: {ep['query'][:70]}")
        print(f"    relevant_sources : {ep['relevant_sources']}")
        print(f"    degraded_static  : {sel.degraded_static}   lambda={sel.lambda_}")
        print(f"    semantic_expanded: {sel.semantic_expanded}")
        print(f"    selected({len(sel.sources)}):")
        for s in sel.sources:
            mark = "★" if s.node_id in ep["relevant_sources"] else " "
            print(f"      {mark} {s.node_id:<10} score={s.src_score:.4f} "
                  f"(str={s.strength:.3f} rel={s.relevance:.3f} via={s.via_edge or '—'})")


if __name__ == "__main__":
    main()
