"""图结构消融运行器（G0–G4 + 均衡档 B0–B4）。

与 `run_ablation.py` 的区别
--------------------------
- 数据来自 `data/graph_dataset.jsonl`（带真实 topic 图：节点/typed 边/owner_node 标记/标注源节点）。
- **图结构固定为常量**（设计原则：只有调度组件是变量）：每 episode 由数据集**显式注入**节点与边，
  关闭 `graph.auto_link`，避免自动建边把"图结构"这一变量搅进来。
- 输出 4 个图专用新指标 + **召回天花板**（POC 教训：不报天花板会误读）。

用法
----
    python -m analysis.run_graph_ablation                    # 全组，离线占位向量（快，免网）
    python -m analysis.run_graph_ablation --real             # 真实 DashScope embedding（推荐）
    python -m analysis.run_graph_ablation --groups graph_g4 graph_g2
    python -m analysis.run_graph_ablation --real --with-oracle   # 附带 oracle 选源上界（Q7）
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STUB_PRIOR = {"time": 0.25, "semantic": 0.35, "frequency": 0.15, "task": 0.25}


def load_dotenv_simple() -> None:
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_episodes(path: Path, n: int | None) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if n and len(rows) >= n:
                break
    return rows


def _build_pre_encoder(episodes: list[dict], cfg_path: Path, cache_dir: str, offline: bool):
    """预热 query embedding（组外一次性编码），使延迟只反映**调度**本身。

    为什么必须预热：embedding 首次调用要走网络（DashScope 约 200~400ms），
    第一个跑的组会把网络耗时算进自己的延迟里 → 组间延迟不可比。
    返回 `{query: vector}`；离线占位向量同样适用。
    """
    from srtp_memory.config import SchedulingConfig
    from srtp_memory.middleware import MemorySchedulingMiddleware

    cfg = SchedulingConfig.load(cfg_path)
    cfg.embedding_cache_dir = cache_dir
    cfg.graph.enabled = False          # 预编码不需要图
    tmp = tempfile.mkdtemp()
    try:
        mid = MemorySchedulingMiddleware(
            config=cfg, user_id="warm", session_id="w",
            resident_pool_path=os.path.join(tmp, "rp.jsonl"),
            shared_pool_path=os.path.join(tmp, "sp.json"),
            log_path=os.path.join(tmp, "m.jsonl"),
            llm_prior_fn=lambda q: dict(STUB_PRIOR),
        )
        out: dict[str, object] = {}
        for ep in episodes:
            q = ep["query"]
            if q not in out:
                out[q] = mid.embedding.encode(q)
        print(f"   (warm-up: {len(out)} query embeddings cached; latency excludes network)")
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_group(group_name: str, cfg_path: Path, episodes: list[dict],
              cache_dir: str, offline: bool, injection_cap: int = 10,
              oracle: bool = False, pre_emb: dict | None = None) -> list[dict]:
    from srtp_memory.config import SchedulingConfig
    from srtp_memory.condition import ConditionKey
    from srtp_memory.middleware import MemorySchedulingMiddleware

    from .graph_metrics import (ceiling, cross_source_relevance, dedup_rate,
                               multi_source_ratio, recall_at_kept,
                               recall_normalized, source_hit_rate)

    cfg = SchedulingConfig.load(cfg_path)
    cfg.embedding_cache_dir = cache_dir
    cfg.max_shared = injection_cap        # 注入上限可覆盖（配额是否 binding 的敏感性探针）
    if not cfg.graph.enabled:
        raise ValueError(f"{cfg_path.name} 未开启 graph.enabled，不属于图消融组")
    cfg.graph.auto_link = False   # 图结构由数据集固定注入（图是常量，调度才是变量）
    stub_prior = (lambda q: dict(STUB_PRIOR)) if offline else None

    rows: list[dict] = []
    for ep in episodes:
        tmp = tempfile.mkdtemp()
        try:
            mid = MemorySchedulingMiddleware(
                config=cfg, user_id=ep["user_id"], session_id="s",
                resident_pool_path=os.path.join(tmp, "rp.jsonl"),
                shared_pool_path=os.path.join(tmp, "sp.json"),
                log_path=os.path.join(tmp, "m.jsonl"),
                llm_prior_fn=stub_prior,
                graph_dir=os.path.join(tmp, "graph"),
            )
            # (1) 固定注入图结构（节点 + typed 边）
            for nd in ep["nodes"]:
                node = mid.graph.create_node(topic=nd["topic"], summary=nd["summary"],
                                             node_id=nd["node_id"])
                node.emb = mid.edge_builder.embed(f"{nd['topic']} {nd['summary']}")
            for ed in ep["edges"]:
                mid.graph.add_edge(ed["src"], ed["dst"], ed["type"],
                                   evidence="dataset:ground_truth_graph")
            mid.graph.persist()

            # (2) 灌库（每条记忆打 owner_node 标记 —— 大池 + 标记）
            #     timestamp 用数据集给定的固定基准时间 → 完全确定性（不依赖 wall-clock）
            for m in ep["corpus"]:
                mid.add_memory(text=m["text"], memory_id=m["memory_id"],
                               task_tag=m["task_tag"], node_id=m["owner_node"],
                               timestamp=m.get("timestamp"))

            # (3) 调度
            mid.use_node(ep["cur_node"])
            cond = ConditionKey.parse(f"{ep['user_id']}|{ep['scenario']}|{ep['business']}")
            man = ep["relevant_sources"] if oracle else None
            q_emb = (pre_emb or {}).get(ep["query"])
            t0 = time.time()
            res = mid.schedule_once(ep["query"], query_emb=q_emb,
                                    task_tag=ep["task_tag"],
                                    condition=cond, manual_sources=man,
                                    now=ep.get("now"))
            lat = (time.time() - t0) * 1000

            kept_ids = [m.memory_id for m in res.kept]
            kept_owners = [getattr(m, "owner_node", "main") for m in res.kept]
            selected = [s["node_id"] for s in res.sources]
            rel = ep["relevant"]
            r = recall_at_kept(kept_ids, rel)
            ceil = ceiling(rel, injection_cap)
            rows.append({
                "group": group_name,
                "episode_id": ep["episode_id"],
                "cur_node": ep["cur_node"],
                "query": ep["query"],
                "n_relevant": len(rel),
                "recall_kept": round(r, 6),
                "ceiling": round(ceil, 6),
                "recall_norm": recall_normalized(r, ceil),
                "source_hit_rate": source_hit_rate(selected, ep["relevant_sources"]),
                "multi_source_ratio": round(multi_source_ratio(kept_owners), 6),
                "cross_source_relevance": cross_source_relevance(kept_owners, res.src_scores),
                "dedup_rate": dedup_rate(len(kept_ids), len(kept_ids)),  # 单值标记 → 恒 0
                "candidate_count": len(res.candidates),
                "scope_size": res.scope_size,
                "kept_count": len(res.kept),
                "n_selected_sources": len(selected),
                "selected_sources": selected,
                "relevant_sources": list(ep["relevant_sources"]),
                "injected_tokens": res.injected_tokens,
                "llm_calls": res.llm_calls,
                "embed_calls_miss": res.embed_calls_miss,
                "perf_tier": res.perf_tier,
                "latency_ms": round(lat, 2),
                "internal_latency_ms": res.latency_ms.get("total", 0.0),
            })
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/graph_dataset.jsonl")
    ap.add_argument("--ablation-dir", default="config/ablation")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--real", action="store_true", help="真实 DashScope embedding（需 key）")
    ap.add_argument("--cache-dir", default="data/embed_cache")
    ap.add_argument("--with-oracle", action="store_true",
                    help="为 G4 额外跑一次 oracle 选源（人工指定正确源）= Q7 上界")
    ap.add_argument("--injection-cap", type=int, default=10,
                    help="注入上限 K（默认 10，与设计常量一致）；调小可探测配额是否 binding")
    ap.add_argument("--out", default="data/runs/graph_ablation_summary.json")
    args = ap.parse_args()

    from .graph_metrics import aggregate, judge

    load_dotenv_simple()
    offline = not args.real
    if args.real and not os.environ.get("DASHSCOPE_API_KEY"):
        print("WARN: --real but no DASHSCOPE_API_KEY -> fallback to offline placeholder")
        offline = True

    episodes = load_episodes(ROOT / args.dataset, args.n)
    if not episodes:
        raise SystemExit(f"数据集为空：{ROOT / args.dataset}（先跑 python -m analysis.gen_graph_dataset）")
    adir = ROOT / args.ablation_dir
    group_files = sorted(list(adir.glob("graph_g*.yaml")) + list(adir.glob("graph_b*.yaml")))
    if args.groups:
        wanted = set(args.groups)
        group_files = [g for g in group_files if g.stem in wanted]
    if not group_files:
        raise SystemExit("没有匹配的消融组 YAML")

    cache_dir = str(ROOT / args.cache_dir)
    cap = int(args.injection_cap)
    pre_emb = _build_pre_encoder(episodes, group_files[0], cache_dir, offline)
    all_rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for gf in group_files:
        tag = f"{gf.stem}{'+oracle' if (args.with_oracle and gf.stem == 'graph_g4') else ''}"
        print(f">> {tag}  ({len(episodes)} episodes, K={cap}, "
              f"{'offline placeholder' if offline else 'real embedding'})")
        try:
            rows = run_group(gf.stem, gf, episodes, cache_dir, offline,
                             injection_cap=cap, pre_emb=pre_emb)
        except Exception as e:  # noqa: BLE001 - 单组失败不中断整体
            print(f"   FAIL -> unavailable: {repr(e)[:140]}")
            summaries[gf.stem] = {"n_episodes": 0, "unavailable": True,
                                  "reason": repr(e)[:200]}
            continue
        if args.with_oracle and gf.stem == "graph_g4":
            try:
                rows_o = run_group("graph_g4_oracle", gf, episodes, cache_dir, offline,
                                   injection_cap=cap, oracle=True, pre_emb=pre_emb)
                all_rows.extend(rows_o)
                summaries["graph_g4_oracle"] = aggregate(rows_o)
                print("    oracle:", json.dumps(summaries["graph_g4_oracle"], ensure_ascii=False))
            except Exception as e:  # noqa: BLE001
                print(f"   oracle FAIL: {repr(e)[:120]}")
        all_rows.extend(rows)
        summaries[gf.stem] = aggregate(rows)
        s = summaries[gf.stem]
        print(f"    recall={s['avg_recall_kept']} ceil={s['avg_ceiling']} "
              f"norm={s['avg_recall_norm']} src_hit={s['avg_source_hit_rate']} "
              f"multi_src={s['avg_multi_source_ratio']}"
              f"(multi-subset={s['avg_multi_source_ratio_multi']}) "
              f"kept={s['avg_kept']} p90={s['p90_latency_ms']}ms llm={s['avg_llm_calls']}")

    # ---- 落盘 ----
    runs_dir = ROOT / "data/runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = ROOT / "data/metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "graph_schedule.jsonl", "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")

    main_chain = {k: v for k, v in summaries.items() if k.startswith("graph_g")}
    verdict = judge(main_chain)
    payload = {
        "mode": "offline" if offline else "real",
        "n_episodes": len(episodes),
        "dataset": str(ROOT / args.dataset),
        "injection_cap": cap,
        "groups": summaries,
        "judgement": verdict,
        "notes": [
            "图结构由数据集固定注入（auto_link=false）：图是常量，只有调度组件是消融变量。",
            "ceiling = min(1, K/|relevant|)，K=注入上限；recall_norm = recall/ceiling（POC 教训）。",
            "延迟口径：query embedding 已组外预热，latency_ms 只反映调度本身（不含网络）。",
            "multi_source_ratio 同时给出全体均值与**多源样本子集**均值（单源样本该值恒为 0）。",
            "dedup_rate 在 owner_node 单值标记下结构性恒为 0（见 analysis/graph_metrics.py 说明）。",
        ],
    }
    (runs_dir / Path(args.out).name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 判据（主链 G0-G4）===")
    for k, v in verdict.items():
        flag = {True: "PASS", False: "FAIL", None: "N/A"}[v.get("pass")]
        print(f"  [{flag}] {k} = {v.get('value')}")

    # 登记 run_manifest
    from srtp_memory.monitor.run_registry import RunRegistry, gen_ablation_manifest
    reg = RunRegistry(runs_dir / "run_manifest.csv")
    manifest = gen_ablation_manifest(adir, runs_dir / "ablation_manifest.json")
    for name, s in summaries.items():
        base = name.replace("_oracle", "")
        gc = manifest["groups"].get(base, {})
        reg.register({
            "group": name,
            "mode": "offline" if offline else "real",
            "retriever": gc.get("retriever"),
            "attention": gc.get("attention"),
            "weight": gc.get("weight"),
            "n_episodes": s.get("n_episodes"),
            "avg_recall_kept": s.get("avg_recall_kept"),
            "avg_precision_kept": s.get("avg_recall_norm"),
            "avg_latency_ms": s.get("avg_latency_ms"),
        })
    print(f"\nOK: data/metrics/graph_schedule.jsonl ({len(all_rows)} rows), "
          f"{args.out}, data/runs/run_manifest.csv")


def _json_default(o):
    import math
    if isinstance(o, float) and math.isnan(o):
        return None
    return str(o)


if __name__ == "__main__":
    main()
