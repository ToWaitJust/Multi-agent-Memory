"""单 episode 诊断工具：打印某组在一条 episode 上的选源、四维分、final 与阈值判定。

用途：当某组 recall 异常（例如全组 kept=0）时，用本工具一眼看出卡在哪一环：
  选源为空？→ 邻域/边的问题
  候选数为 0？→ 范围限定或配额预筛的问题
  final 全低于阈值？→ 打分器几何/阈值的问题（本轮就靠它定位了 threshold=0.6 不可达）

用法
----
    python -m analysis.diag_graph_episode graph_g4        # 默认第 0 条 episode
    python -m analysis.diag_graph_episode graph_g2 5      # 第 5 条
"""
import json
import os
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from analysis.run_graph_ablation import load_dotenv_simple  # noqa: E402
from srtp_memory.condition import ConditionKey  # noqa: E402
from srtp_memory.config import SchedulingConfig  # noqa: E402
from srtp_memory.middleware import MemorySchedulingMiddleware  # noqa: E402

load_dotenv_simple()
group = sys.argv[1] if len(sys.argv) > 1 else "graph_g4"
ep_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

with open(os.path.join(ROOT, "data/graph_dataset.jsonl"), encoding="utf-8") as f:
    episodes = [json.loads(ln) for ln in f if ln.strip()]
ep = episodes[ep_idx]

cfg = SchedulingConfig.load(os.path.join(ROOT, f"config/ablation/{group}.yaml"))
cfg.embedding_cache_dir = os.path.join(ROOT, "data/embed_cache")
cfg.graph.auto_link = False
tmp = tempfile.mkdtemp()
mid = MemorySchedulingMiddleware(
    config=cfg, user_id="u_tu", session_id="s",
    resident_pool_path=os.path.join(tmp, "rp.jsonl"),
    shared_pool_path=os.path.join(tmp, "sp.json"),
    log_path=os.path.join(tmp, "m.jsonl"),
    graph_dir=os.path.join(tmp, "graph"),
    llm_prior_fn=lambda q: {"time": 0.25, "semantic": 0.35, "frequency": 0.15, "task": 0.25},
)
for nd in ep["nodes"]:
    n = mid.graph.create_node(topic=nd["topic"], summary=nd["summary"], node_id=nd["node_id"])
    n.emb = mid.edge_builder.embed(f"{nd['topic']} {nd['summary']}")
for ed in ep["edges"]:
    mid.graph.add_edge(ed["src"], ed["dst"], ed["type"])
for m in ep["corpus"]:
    mid.add_memory(text=m["text"], memory_id=m["memory_id"], task_tag=m["task_tag"],
                   node_id=m["owner_node"], timestamp=m["timestamp"])
mid.use_node(ep["cur_node"])
cond = ConditionKey.parse(f"{ep['user_id']}|{ep['scenario']}|{ep['business']}")
res = mid.schedule_once(ep["query"], task_tag=ep["task_tag"], condition=cond, now=ep["now"])

print(f"[{group}] cur={ep['cur_node']} query={ep['query']}")
print("relevant    :", ep["relevant"])
print("rel_sources :", ep["relevant_sources"])
print("sources     :", [(s["node_id"], round(s["src_score"], 3), s["quota"]) for s in res.sources])
print("scope/cand  :", res.scope_size, "/", len(res.candidates))
print("weights     :", {k: round(v, 3) for k, v in res.weights.items()})
print("enabled_dims:", res.enabled_dims)
finals = []
for s in res.scored:
    fin = s["score"]["final"]
    finals.append(fin)
    mark = "RELEVANT" if s["memory"].memory_id in ep["relevant"] else ""
    print(f'  {s["memory"].memory_id:12s} owner={s["via_source"]:12s} '
          f't={s["score"]["time"]:.3f} sem={s["score"]["semantic"]:.3f} '
          f'f={s["score"]["frequency"]:.3f} task={s["score"]["task"]:.3f} '
          f'final={fin:.4f} {mark}')
if finals:
    floor = float(getattr(cfg, "threshold", 0.0))
    std = statistics.stdev(finals) if len(finals) > 1 else 0.0
    thr = max(floor, statistics.mean(finals) - 0.5 * std)
    print(f"finals: min={min(finals):.4f} max={max(finals):.4f} "
          f"mean={statistics.mean(finals):.4f} floor={floor} -> 实际阈值={thr:.4f}")
print("kept:", len(res.kept), "| latency_total_ms:", round(res.latency_ms.get("total", 0), 1))
shutil.rmtree(tmp, ignore_errors=True)
