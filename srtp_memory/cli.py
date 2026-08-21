"""CLI 入口（§12.5 / §13.4）：python -m srtp_memory.cli --config <yaml>。

- 单组跑：--config config/ablation/full_ours.yaml 跑一次调度管线
- 批量消融：--ablation config/ablation 扫目录 5 组全跑，汇总 metrics
- 种子注入：--seed 随机种子（可复现，NFR-6）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import SchedulingConfig
from .condition import ConditionKey
from .plugins import register_all


def run_one(config_path: str | Path, query: str = "多智能体记忆共享调度方法研究",
            seed: int = 42) -> dict:
    """加载配置 → 装配 middleware → 跑一次调度，返回结果 dict。"""
    from .middleware import MemorySchedulingMiddleware

    register_all()
    cfg = SchedulingConfig.load(config_path)
    # LLM 先验 fn：无真实小模型时用确定性的默认权重（接真实环境时替换）
    def _prior(q: str) -> dict:
        return {"time": 0.25, "semantic": 0.35, "frequency": 0.15, "task": 0.25}

    mid = MemorySchedulingMiddleware(
        config=cfg,
        user_id="u_tu", session_id=f"sess_{seed}",
        log_path="data/metrics/schedule.jsonl",
        llm_prior_fn=_prior,
    )
    # 种入一条种子记忆（真实场景由 ReMe auto_memory 写入）
    from .attention import MemoryCandidate
    mid.resident_pool.upsert(MemoryCandidate(
        memory_id=f"seed_{seed}", text="多智能体记忆共享调度方法研究：四维注意力+LLM可学习权重",
        path="daily/seed.md", timestamp=time.time(), task_tag="main_task",
        user_id="u_tu", session_id=f"sess_{seed}"))

    cond = ConditionKey.parse(cfg.condition_key)
    res = mid.schedule_once(query, task_tag=cfg.task_tag, condition=cond)
    mid.flush()
    return {
        "group": Path(config_path).stem,
        "retriever": cfg.retriever_impl,
        "attention": cfg.attention_impl,
        "weight": cfg.weight_impl,
        "selector": cfg.selector_impl,
        **res.to_dict(),
    }


def run_ablation(ablation_dir: str | Path) -> list[dict]:
    """扫 config/ablation/*.yaml 批量跑 5 组，返回各组结果。"""
    results = []
    for yf in sorted(Path(ablation_dir).glob("*.yaml")):
        print(f"[run] {yf.name} ...", flush=True)
        results.append(run_one(yf))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="srtp_memory 调度/消融 CLI")
    ap.add_argument("--config", help="单组配置文件（如 config/ablation/full_ours.yaml）")
    ap.add_argument("--ablation", help="消融目录（扫描 *.yaml 批量跑）")
    ap.add_argument("--query", default="多智能体记忆共享调度方法研究")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/runs/ablation_results.json")
    args = ap.parse_args()

    if args.config:
        r = run_one(args.config, args.query, args.seed)
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif args.ablation:
        results = run_ablation(args.ablation)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n完成 {len(results)} 组，结果写入 {out}")
        for r in results:
            print(f"  {r['group']}: kept={r['kept_count']}/{r['candidate_count']} "
                  f"ratio={r['compression_ratio']:.2f} latency={r['latency_ms'].get('total',0):.0f}ms")
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
