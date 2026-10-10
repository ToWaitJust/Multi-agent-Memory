"""配对 bootstrap 显著性检验（论文硬需求之一：所有结论必须带置信区间）。

用法：
    python -m analysis.stat_significance                     # 默认对比 G4 vs G2
    python -m analysis.stat_significance --a graph_g4 --b graph_g3
    python -m analysis.stat_significance --pairs g4:g2 g4:g3 g4:g1 --metric mrr
    python -m analysis.stat_significance --scene             # 按 scene 分层各出一份

为什么用**配对** bootstrap：
    两组跑的是**同一批 episode**（同一个 episode_id 在两组各有一行），
    样本不独立。独立两样本检验会把"题目难度差异"当成噪声，严重低估显著性。
    配对检验对同一 episode 取差，只保留"方法造成的差异"。

输出：均值差 Δ、95% CI、bootstrap 双尾 p、Cohen's d（配对版）。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "data" / "metrics"
METRICS = METRICS_DIR / "graph_schedule.jsonl"

#: ⚠️ runner 每次跑都会**覆盖** graph_schedule.jsonl，所以历史组会丢。
#: 约定：跑完一组立刻 `cp graph_schedule.jsonl locomo_<group>.jsonl` 留档，
#: 本脚本默认扫描所有 `*.jsonl`（除 graph_schedule.jsonl 本身）合并成一张大表。
def _metrics_files(extra: list[str] | None = None) -> list[Path]:
    if extra:
        return [Path(p) for p in extra]
    files = sorted(METRICS_DIR.glob("*.jsonl"))
    # 同名的 group 若出现在多个文件里，load() 按"最后一行胜出"处理
    return files

METRICS_FIELDS = {
    "recall": "recall_kept",
    "recall_norm": "recall_norm",
    "mrr": "mrr",
    "ndcg": "ndcg_at_k",
    "source_hit": "source_hit_rate",
    "cand_hit": "cand_hit_rate",
}


def load(group: str, metric_field: str, scene: str | None = None,
         files: list[Path] | None = None) -> dict[str, float]:
    """取某组每 episode 的指标值 {episode_id: value}。同 episode 多行时取最后一行。"""
    out: dict[str, float] = {}
    for f in (files or _metrics_files()):
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("group") != group:
                continue
            if scene and r.get("scene") != scene:
                continue
            out[r["episode_id"]] = float(r.get(metric_field, 0.0) or 0.0)
    return out


def paired_bootstrap(a: dict[str, float], b: dict[str, float],
                     n_boot: int = 10000, seed: int = 20261008):
    """a - b 的配对 bootstrap：返回 (delta, ci_lo, ci_hi, p_two_sided, cohens_d, n)。"""
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    diffs = [a[k] - b[k] for k in keys]
    n = len(diffs)
    delta = sum(diffs) / n
    if n < 2:
        return dict(delta=delta, ci_lo=delta, ci_hi=delta, p=1.0, d=0.0, n=n)
    var = sum((x - delta) ** 2 for x in diffs) / (n - 1)
    sd = var ** 0.5
    d = delta / sd if sd > 1e-12 else 0.0

    rng = random.Random(seed)
    boot = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        boot.append(s)
    boot.sort()
    lo = boot[int(0.025 * n_boot)]
    hi = boot[min(n_boot - 1, int(0.975 * n_boot))]
    # 双尾 p：bootstrap 分布跨过 0 的比例
    le = sum(1 for x in boot if x <= 0)
    ge = sum(1 for x in boot if x >= 0)
    p = min(1.0, 2 * min(le, ge) / n_boot)
    return dict(delta=delta, ci_lo=lo, ci_hi=hi, p=p, d=d, n=n)


def fmt(r) -> str:
    if r is None:
        return "  （无共同 episode，无法配对）"
    sign = "显著" if r["p"] < 0.05 else "不显著"
    star = "***" if r["p"] < 0.001 else ("**" if r["p"] < 0.01 else ("*" if r["p"] < 0.05 else ""))
    return (f"  Δ = {r['delta']:+.4f}  95%CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  "
            f"p = {r['p']:.4f} {star:<3} ({sign})  Cohen's d = {r['d']:+.3f}  n = {r['n']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="graph_g4")
    ap.add_argument("--b", default="graph_g2")
    ap.add_argument("--pairs", nargs="*", default=None,
                    help='多组对比，格式 "A:B"，如 g4:g2 g4:g3')
    ap.add_argument("--metric", default="recall", choices=list(METRICS_FIELDS))
    ap.add_argument("--scene", default=None, help="只统计某个 scene（分层报告）")
    ap.add_argument("--scenes", action="store_true", help="按 scene 各出一份")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--metrics", nargs="*", default=None,
                    help="显式指定 metrics jsonl（默认扫描 data/metrics/*.jsonl）")
    ap.add_argument("--list-groups", action="store_true", help="列出当前可用的组")
    args = ap.parse_args()

    files = _metrics_files(args.metrics)
    if args.list_groups:
        seen: dict[str, int] = {}
        for f in files:
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    g = json.loads(line).get("group")
                    seen[g] = seen.get(g, 0) + 1
        print(f"扫描 {len(files)} 个文件，可用组：")
        for g, c in sorted(seen.items(), key=lambda x: str(x[0])):
            print(f"  {g:<28} {c} episodes")
        return

    field = METRICS_FIELDS[args.metric]
    pairs = []
    if args.pairs:
        for p in args.pairs:
            a, b = p.split(":")
            pairs.append((a, b))
    else:
        pairs.append((args.a, args.b))

    scenes = [None]
    if args.scenes:
        # 从文件里自动发现 scene 列表
        seen = []
        for f in files:
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    s = json.loads(line).get("scene")
                    if s and s not in seen:
                        seen.append(s)
        scenes = sorted(seen)

    for sc in scenes:
        tag = f"[{sc}]" if sc else "[全量]"
        print(f"\n=== {tag}  指标 = {args.metric}（配对 bootstrap，B={args.n_boot}）===")
        for a, b in pairs:
            ra = load(a, field, sc, files)
            rb = load(b, field, sc, files)
            if not ra or not rb:
                print(f"  {a} − {b}: 数据缺失（{a}={len(ra)} 条, {b}={len(rb)} 条）")
                continue
            ma = sum(ra.values()) / len(ra)
            mb = sum(rb.values()) / len(rb)
            print(f"  {a} − {b}   （{a}={ma:.4f}  {b}={mb:.4f}）")
            print(fmt(paired_bootstrap(ra, rb, args.n_boot)))


if __name__ == "__main__":
    main()
