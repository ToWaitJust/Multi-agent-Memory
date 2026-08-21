"""统计脚本骨架：读取 schedule.jsonl / metrics.jsonl 计算验收指标（§9.3）。

用法: python -m analysis.compute_metrics --manifest data/runs/ablation_manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_schedule_metrics(schedule_rows: list[dict]) -> dict:
    """从 schedule 日志汇总 6 项验收指标（§2.4）。"""
    if not schedule_rows:
        return {}
    n = len(schedule_rows)
    compression = [r.get("compression_ratio", 0.0) for r in schedule_rows]
    latency = [r.get("latency_ms", {}).get("total", 0.0) for r in schedule_rows]
    kept = [r.get("kept_count", 0) for r in schedule_rows]
    cands = [r.get("candidate_count", 0) for r in schedule_rows]
    return {
        "n_samples": n,
        "avg_compression_ratio": sum(compression) / n,
        "avg_latency_ms": sum(latency) / n,
        "avg_kept": sum(kept) / n,
        "avg_candidates": sum(cands) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/runs/ablation_manifest.json")
    ap.add_argument("--schedule", default="data/metrics/schedule.jsonl")
    args = ap.parse_args()

    rows = load_jsonl(Path(args.schedule))
    metrics = compute_schedule_metrics(rows)
    print("schedule.jsonl 汇总指标:", json.dumps(metrics, ensure_ascii=False, indent=2))

    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        print("ablation 组:", list(m.get("groups", {}).keys()))


if __name__ == "__main__":
    main()
