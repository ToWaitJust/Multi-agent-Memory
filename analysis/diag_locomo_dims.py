"""LoCoMo 四维打分诊断（D-049 异常排查）。

排查两个异常：
  ① 禁用 time 维后，结果与 G4 **逐位相同** → time 维到底有没有起作用？
  ② G4_norm（批级归一化）反而更差 → 归一化到底动了什么？

本脚本把单个 episode 的**候选四维分分布 + 实际生效权重 + kept 明细**全部打印出来，
用数据说话，不猜。

用法
----
    python -m analysis.diag_locomo_dims --idx 5                     # 默认 graph_g4
    python -m analysis.diag_locomo_dims --idx 5 --group graph_g4_notime
    python -m analysis.diag_locomo_dims --idx 5 --group graph_g4_norm
    python -m analysis.diag_locomo_dims --idx 5 --compare            # 并排对比 g4 / notime
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mean_std(xs):
    if len(xs) < 2:
        return (xs[0] if xs else 0.0), 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def run_one(ep: dict, cfg_path: Path, cache_dir: str, warmup: bool = True,
            warmup_n: int = 3) -> dict:
    """装配 middleware → 灌库 → 调度一次，返回诊断所需明细。"""
    from srtp_memory.config import SchedulingConfig
    from srtp_memory.condition import ConditionKey
    from srtp_memory.middleware import MemorySchedulingMiddleware

    cfg = SchedulingConfig.load(cfg_path)
    cfg.embedding_cache_dir = cache_dir
    cfg.budget.default_tier = "fast"
    cfg.graph.auto_link = False

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
    mid.graph.persist()

    ts_seen = []
    for m in ep["corpus"]:
        mid.add_memory(text=m["text"], memory_id=m["memory_id"],
                       task_tag=m["task_tag"], node_id=m["owner_node"],
                       timestamp=m.get("timestamp"), persist=False)
        ts_seen.append(m.get("timestamp") or 0.0)
    mid.resident_pool.persist()

    mid.use_node(ep["cur_node"])
    cond = ConditionKey.parse(f"{ep['user_id']}|{ep['scenario']}|{ep['business']}")
    ep_now = ep.get("now") or 0.0

    if warmup:
        by_id = {m["memory_id"]: m for m in ep["corpus"]}
        for k, rid in enumerate(ep["relevant"][:max(1, warmup_n)], 1):
            txt = (by_id.get(rid) or {}).get("text")
            if txt:
                mid.schedule_once(txt, task_tag=ep["task_tag"], condition=cond,
                                  now=ep_now - 3600.0 * k)

    res = mid.schedule_once(ep["query"], task_tag=ep["task_tag"], condition=cond,
                            now=ep_now)

    # ---- 收集候选四维分 ----
    dims = ["time", "semantic", "frequency", "task"]
    cols = {d: [] for d in dims}
    for item in res.scored:
        sc = item["score"]
        for d in dims:
            cols[d].append(float(sc.get(d, 0.0)))

    stats = {}
    for d in dims:
        mu, sd = _mean_std(cols[d])
        stats[d] = {"min": min(cols[d]) if cols[d] else 0.0,
                    "max": max(cols[d]) if cols[d] else 0.0,
                    "mean": mu, "std": sd,
                    "n_unique": len(set(round(x, 6) for x in cols[d]))}

    # ---- 相关记忆在最终排序里的位置 ----
    rel_set = set(ep["relevant"])
    rel_ranks = []
    for i, item in enumerate(res.scored, 1):
        if item["memory"].memory_id in rel_set:
            rel_ranks.append(i)

    # ---- kept 明细 ----
    kept_rows = []
    for item in res.scored[:10]:
        sc = item["score"]
        kept_rows.append({
            "rank": len(kept_rows) + 1,
            "mid": item["memory"].memory_id,
            "src": item.get("via_source"),
            "final": round(float(sc.get("final", 0.0)), 4),
            "time": round(float(sc.get("time", 0.0)), 4),
            "sem": round(float(sc.get("semantic", 0.0)), 4),
            "freq": round(float(sc.get("frequency", 0.0)), 4),
            "task": round(float(sc.get("task", 0.0)), 4),
            "is_rel": item["memory"].memory_id in rel_set,
        })

    # ---- 候选时间戳分布（time 维的输入）----
    cand_ts = [it["memory"].timestamp for it in res.scored]
    ts_stat = {}
    if cand_ts:
        ts_stat = {"n_unique": len(set(round(t, 1) for t in cand_ts)),
                   "min": min(cand_ts), "max": max(cand_ts),
                   "span_hours": round((max(cand_ts) - min(cand_ts)) / 3600.0, 1)}

    return {
        "config": {
            "group": cfg_path.stem,
            "attention_impl": cfg.attention_impl,
            "attention_disabled_dims": list(cfg.attention_disabled_dims),
            "semantic_floor": cfg.attention_semantic_floor,
        },
        "enabled_dims": sorted(res.enabled_dims) if res.enabled_dims else None,
        "weights": {k: round(v, 4) for k, v in res.weights.items()},
        "n_scored": len(res.scored),
        "scope_size": res.scope_size,
        "src_scores": {k: round(v, 4) for k, v in res.src_scores.items()},
        "dim_stats": {d: {k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in s.items()} for d, s in stats.items()},
        "cand_ts": ts_stat,
        "corpus_ts_span_hours": round((max(ts_seen) - min(ts_seen)) / 3600.0, 1) if ts_seen else 0,
        "relevant_ranks": rel_ranks,
        "kept_top": kept_rows,
    }


def _print(tag: str, d: dict) -> None:
    print(f"\n{'=' * 72}\n[{tag}]  group={d['config']['group']}  "
          f"attention={d['config']['attention_impl']}")
    print(f"  attention_disabled_dims = {d['config']['attention_disabled_dims'] or '（无）'}")
    print(f"  实际 enabled_dims       = {d['enabled_dims']}")
    print(f"  实际权重 w              = {d['weights']}")
    print(f"  候选数={d['n_scored']}  scope={d['scope_size']}")
    print(f"  源分 src_scores         = {d['src_scores']}")
    print(f"\n  --- 候选四维分分布 ---")
    print(f"  {'维':10s} {'min':>8s} {'max':>8s} {'mean':>8s} {'std':>9s} {'取值数':>7s}")
    for dim, s in d["dim_stats"].items():
        print(f"  {dim:10s} {s['min']:8.4f} {s['max']:8.4f} {s['mean']:8.4f} "
              f"{s['std']:9.6f} {s['n_unique']:7d}")
    print(f"\n  候选 timestamp: 取值数={d['cand_ts'].get('n_unique')} "
          f"跨度={d['cand_ts'].get('span_hours')}h   （语料总跨度 {d['corpus_ts_span_hours']}h）")
    print(f"  相关记忆在 {d['n_scored']} 条候选中的排名 = {d['relevant_ranks']}")
    print(f"\n  --- final 前 10 条 ---")
    print(f"  {'#':>3s} {'final':>7s} {'time':>7s} {'sem':>7s} {'freq':>6s} {'task':>6s} "
          f"{'相关':>5s}  src")
    for r in d["kept_top"]:
        print(f"  {r['rank']:3d} {r['final']:7.4f} {r['time']:7.4f} {r['sem']:7.4f} "
              f"{r['freq']:6.3f} {r['task']:6.3f} {'★' if r['is_rel'] else '':>5s}  {r['src']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="LoCoMo 四维打分诊断")
    ap.add_argument("--dataset", default="data/locomo_conv0.jsonl")
    ap.add_argument("--idx", type=int, default=5, help="episode 序号")
    ap.add_argument("--group", default="graph_g4")
    ap.add_argument("--compare", action="store_true", help="并排对比 g4 / g4_notime / g4_norm")
    ap.add_argument("--cache-dir", default="data/embed_cache")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    eps = [json.loads(l) for l in open(ROOT / args.dataset, encoding="utf-8") if l.strip()]
    if not (0 <= args.idx < len(eps)):
        raise SystemExit(f"idx 超出范围（共 {len(eps)} 条）")
    ep = eps[args.idx]
    print(f"episode#{args.idx}  scene={ep['scene']}  cur_node={ep['cur_node']}")
    print(f"query: {ep['query'][:100]}")
    print(f"relevant({len(ep['relevant'])}): {ep['relevant'][:5]}")

    groups = (["graph_g4", "graph_g4_notime", "graph_g4_norm"] if args.compare
              else [args.group])
    for g in groups:
        cfg_path = ROOT / "config" / "ablation" / f"{g}.yaml"
        if not cfg_path.exists():
            print(f"\n[skip] 配置不存在: {cfg_path}")
            continue
        d = run_one(ep, cfg_path, args.cache_dir, warmup=not args.no_warmup)
        _print(g, d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
