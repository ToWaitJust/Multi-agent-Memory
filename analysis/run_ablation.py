"""5 组结构化消融运行器（V2.1：对齐 D5 验收）。

流程
----
对 config/ablation/ 下每组 YAML：
  1. 加载 SchedulingConfig
  2. 逐 episode：独立 middleware 实例（隔离语料）+ 共享 embedding 缓存目录
     -> 播种 corpus -> 调度 query -> 取 kept / candidates 记忆 id
  3. 对照 episode.relevant 算 recall@kept / precision@kept / relevant_rank
  4. 落盘 data/metrics/schedule.jsonl（每条 (group,episode) 一行）
  5. 汇总每组指标 -> data/runs/ablation_summary.json + run_manifest.csv 登记

embedding 真实性
---------------
  - 默认离线：无 DASHSCOPE_API_KEY -> embed.dashscope 自动降级确定性占位向量（免费、可复现，但无语义）
  - 设了 key（或 .env 加载）-> 真实 DashScope 语义向量，首次计算写入 data/embed_cache，后续复用
  - LLM 先验同理：离线用默认权重，真实用 DeepSeek 实时先验

用法
----
  python -m analysis.run_ablation                      # 默认全量 200 条, 离线占位
  python -m analysis.run_ablation --n 20                # 先小批冒烟
  python -m analysis.run_ablation --real                # 真实 DashScope + DeepSeek 先验
  python -m analysis.run_ablation --groups full_ours baseline_naive
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_dotenv_simple() -> None:
    """极简 .env 加载（不设第三方依赖），仅 setdefault，不覆盖已存在环境变量。"""
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
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if n and len(rows) >= n:
                break
    return rows


def run_group(group_name: str, cfg_path: Path, episodes: list[dict],
              cache_dir: str, offline: bool) -> list[dict]:
    from srtp_memory.config import SchedulingConfig
    from srtp_memory.middleware import MemorySchedulingMiddleware
    from srtp_memory.condition import ConditionKey

    cfg = SchedulingConfig.load(cfg_path)
    cfg.embedding_cache_dir = cache_dir
    # 离线：注入常量 LLM 先验，避免触网/计费
    stub_prior = (lambda q: {"time": 0.25, "semantic": 0.35,
                             "frequency": 0.15, "task": 0.25}) if offline else None

    rows: list[dict] = []
    for ep in episodes:
        tmp = tempfile.mkdtemp()
        mid = MemorySchedulingMiddleware(
            config=cfg, user_id=ep["user_id"], session_id="s",
            resident_pool_path=os.path.join(tmp, "rp.jsonl"),
            shared_pool_path=os.path.join(tmp, "sp.json"),
            log_path=os.path.join(tmp, "m.jsonl"),
            llm_prior_fn=stub_prior,
        )
        for m in ep["corpus"]:
            mid.add_memory(text=m["text"], memory_id=m["memory_id"], task_tag=m["task_tag"])
        cond = ConditionKey.parse(f"{ep['user_id']}|{ep['scenario']}|{ep['business']}")
        t0 = time.time()
        res = mid.schedule_once(ep["query"], task_tag=ep["task_tag"], condition=cond)
        lat = (time.time() - t0) * 1000

        cand_ids = [c.memory_id for c in res.candidates]
        kept_ids = [c.memory_id for c in res.kept]
        rel = set(ep["relevant"])
        recall_kept = 1.0 if rel.issubset(set(kept_ids)) else 0.0
        prec_kept = len(rel & set(kept_ids)) / max(1, len(kept_ids))
        rank = None
        for i, c in enumerate(res.candidates, 1):
            if c.memory_id in rel:
                rank = i
                break
        rows.append({
            "group": group_name,
            "episode_id": ep["episode_id"],
            "query": ep["query"],
            "candidate_count": len(res.candidates),
            "kept_count": len(res.kept),
            "relevant": ep["relevant"],
            "kept_ids": kept_ids,
            "recall_kept": recall_kept,
            "precision_kept": round(prec_kept, 4),
            "relevant_rank": rank,
            "latency_ms": round(lat, 2),
        })
    return rows


def summarize(group_name: str, rows: list[dict]) -> dict:
    n = len(rows)
    def avg(key): return sum(r[key] for r in rows) / n if n else 0.0
    ranks = [r["relevant_rank"] for r in rows if r["relevant_rank"]]
    return {
        "group": group_name,
        "n_episodes": n,
        "avg_recall_kept": round(avg("recall_kept"), 4),
        "avg_precision_kept": round(avg("precision_kept"), 4),
        "avg_relevant_rank": round(sum(ranks) / len(ranks), 2) if ranks else None,
        "avg_kept": round(avg("kept_count"), 2),
        "avg_latency_ms": round(avg("latency_ms"), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/ablation_dataset.jsonl")
    ap.add_argument("--ablation-dir", default="config/ablation")
    ap.add_argument("--n", type=int, default=None, help="只跑前 N 条 episode")
    ap.add_argument("--groups", nargs="*", default=None, help="只跑指定组（默认全跑）")
    ap.add_argument("--real", action="store_true", help="用真实 DashScope + DeepSeek 先验（需 key）")
    ap.add_argument("--cache-dir", default="data/embed_cache")
    args = ap.parse_args()

    load_dotenv_simple()
    offline = not args.real
    if args.real and not os.environ.get("DASHSCOPE_API_KEY"):
        print("⚠️ --real 但无 DASHSCOPE_API_KEY，回退离线占位向量")
        offline = True

    episodes = load_episodes(ROOT / args.dataset, args.n)
    adir = ROOT / args.ablation_dir
    group_files = sorted(adir.glob("*.yaml"))
    if args.groups:
        wanted = set(args.groups)
        group_files = [g for g in group_files if g.stem in wanted]

    cache_dir = str(ROOT / args.cache_dir)
    all_rows: list[dict] = []
    summaries: list[dict] = []
    for gf in group_files:
        print(f"▶ 运行组 {gf.stem}  ({len(episodes)} episodes, {'离线' if offline else '真实'})")
        try:
            rows = run_group(gf.stem, gf, episodes, cache_dir, offline)
        except Exception as e:  # noqa: BLE001 - 单组失败不应中断整体（如 baseline_reme 缺 ReMe 环境）
            print(f"   ⚠️ 组 {gf.stem} 运行失败，标记 unavailable: {repr(e)[:120]}")
            summaries.append({"group": gf.stem, "n_episodes": 0,
                              "avg_recall_kept": None, "avg_precision_kept": None,
                              "avg_relevant_rank": None, "avg_kept": None,
                              "avg_latency_ms": None, "unavailable": True,
                              "reason": repr(e)[:160]})
            continue
        all_rows.extend(rows)
        s = summarize(gf.stem, rows)
        summaries.append(s)
        print("   ", json.dumps(s, ensure_ascii=False))

    # 落盘
    metrics_dir = ROOT / "data/metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "schedule.jsonl", "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    runs_dir = ROOT / "data/runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / "ablation_summary.json").write_text(
        json.dumps({"mode": "offline" if offline else "real",
                    "n_episodes": len(episodes),
                    "groups": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    # 登记 run_manifest.csv
    from srtp_memory.monitor.run_registry import RunRegistry, gen_ablation_manifest
    reg = RunRegistry(runs_dir / "run_manifest.csv")
    manifest = gen_ablation_manifest(adir, runs_dir / "ablation_manifest.json")
    for s in summaries:
        grp_cfg = manifest["groups"].get(s["group"], {})
        reg.register({
            "group": s["group"],
            "mode": "offline" if offline else "real",
            "retriever": grp_cfg.get("retriever"),
            "attention": grp_cfg.get("attention"),
            "weight": grp_cfg.get("weight"),
            "n_episodes": s["n_episodes"],
            "avg_recall_kept": s["avg_recall_kept"],
            "avg_precision_kept": s["avg_precision_kept"],
            "avg_latency_ms": s["avg_latency_ms"],
        })
    print(f"\n✅ 落盘: data/metrics/schedule.jsonl ({len(all_rows)} 行), "
          f"data/runs/ablation_summary.json, data/runs/run_manifest.csv")


if __name__ == "__main__":
    main()
