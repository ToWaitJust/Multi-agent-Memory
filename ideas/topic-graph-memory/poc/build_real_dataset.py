"""Q4 数据集方案 A：用 LLM 真跑「todo-cli 多模块项目」会话，产出带 topic 结构的数据集。

设计见 ../05_scenario_design.md。产出 real_dataset.json：
{ nodes, edges, memories, queries, meta }

运行：E:/1shujukuyuanli/anaconda/envs/srtp-memory/python.exe build_real_dataset.py
- LLM：model.dashscope（qwen-plus，走 .env 的 DASHSCOPE_API_KEY，不落任何密钥到数据）
- 结果落盘后可重复评测，不需要重跑生成（评测侧零 LLM 依赖）
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from srtp_memory.plugins.infra.model_adapters import DashScopeModelAdapter  # noqa: E402

OUT = Path(__file__).parent / "real_dataset.json"

BACKGROUND = (
    "团队正在用 Python 开发一个命令行待办工具 todo-cli，数据存本地 JSON 文件。"
    "项目按子模块分工，每个子模块由一名工程师负责并输出设计决策。"
)

# (节点, 标题, 关键词域, task_tag)
MODULES = [
    ("M1", "配置模块", "配置文件加载、默认值、路径解析", "config"),
    ("M2", "存储模块", "JSON 持久化、schema 字段、读写", "storage"),
    ("M3", "核心逻辑", "增删改查、优先级、截止日期", "core"),
    ("M4", "命令行解析", "子命令、参数、帮助文本", "cli"),
    ("M5", "统计报表", "完成率、逾期统计、周报输出", "stats"),
    ("M6", "异常与日志", "异常分类、日志格式", "exlog"),
]

# src → dst：dst 依赖 src（设计图 = 真实依赖，关系识别质量另测 Q6）
EDGES = [
    ("M1", "M3", "depends_on"), ("M2", "M3", "depends_on"),
    ("M3", "M4", "depends_on"), ("M3", "M5", "depends_on"),
    ("M2", "M5", "depends_on"),
    ("M1", "M6", "depends_on"), ("M2", "M6", "depends_on"),
]
EDGE_STRENGTH = {"derives_from": 0.9, "depends_on": 0.8, "references": 0.5, "similar_to": 0.3}

# 查询：relevant ⊆ cur 的入边邻居（POC 第一轮教训），anchor 见 05 文档 §4
QUERIES = [
    ("RQ1", "M3", "新增任务时配置加载和存储 schema 字段要怎么衔接", ["M1", "M2"], ["配置", "schema", "字段"]),
    ("RQ2", "M4", "命令行子命令的参数怎么对接增删改查接口", ["M3"], ["增删改查", "接口", "优先级"]),
    ("RQ3", "M5", "完成率统计要读哪些存储字段和优先级逻辑", ["M2", "M3"], ["字段", "优先级", "完成"]),
    ("RQ4", "M5", "统计报表读取存储层读写结果时要注意什么", ["M2"], ["读写", "JSON", "存储"]),
    ("RQ5", "M6", "路径解析和读写失败怎么归到异常分类", ["M1", "M2"], ["路径", "读写", "存储"]),
    ("RQ6", "M6", "日志格式里怎么记录配置加载的问题", ["M1"], ["配置", "日志", "路径"]),
]

K = 10


def topological_order() -> list[str]:
    indeg = {m[0]: 0 for m in MODULES}
    adj: dict[str, list[str]] = {m[0]: [] for m in MODULES}
    for s, d, _ in EDGES:
        adj[s].append(d)
        indeg[d] += 1
    queue = [m[0] for m in MODULES if indeg[m[0]] == 0]
    order = []
    while queue:
        n = queue.pop(0)
        order.append(n)
        for t in adj[n]:
            indeg[t] -= 1
            if indeg[t] == 0:
                queue.append(t)
    assert len(order) == len(MODULES), "模块依赖必须是 DAG"
    return order


def parse_memories(text: str) -> list[str]:
    """把 LLM 输出按行解析成记忆条目（宽容版：剥装饰符、长句按分号切、限长 200）。"""
    out = []
    for line in text.splitlines():
        line = re.sub(r"^[\s\-\*\d\.•·>]+", "", line).strip().strip("`*# ")
        if len(line) > 200:
            line = line[:200]
        for seg in re.split(r"[；;。]+", line):
            seg = seg.strip()
            if 8 <= len(seg) <= 160 and not seg.startswith("#"):
                out.append(seg)
    return out


def ask_with_retry(llm, prompt, want, tries=3, **kw) -> list[str]:
    """解析结果少于 want 条时重试（M3 第一轮 0 条的教训）。"""
    best: list[str] = []
    for i in range(tries):
        out = parse_memories(llm.complete(prompt, **kw))
        best = out if len(out) > len(best) else best
        if len(best) >= want:
            return best
        print(f"    retry {i + 1}: 只解析出 {len(best)} 条（要求 ≥{want}）")
    return best


def main():
    llm = DashScopeModelAdapter()
    nodes, edges, memories = [], [], []
    mid = 0
    base_ts = 1_780_000_000.0
    order = topological_order()
    info = {nid: {"title": t, "kws": k, "tag": g} for nid, t, k, g in MODULES}
    produced: dict[str, list[str]] = {m[0]: [] for m in MODULES}

    for nid in order:
        title, kws, tag = info[nid]["title"], info[nid]["kws"], info[nid]["tag"]
        deps = [s for s, d, _ in EDGES if d == nid]
        ctx = "\n".join(
            f"- [{d} {info[d]['title']}] {m}" for d in deps for m in produced[d][:3]
        ) or "（无，本模块是起始模块）"

        prompt1 = (
            f"{BACKGROUND}\n你负责「{title}」子模块（涉及：{kws}）。\n"
            f"已知的依赖模块设计决策：\n{ctx}\n\n"
            f"请列出本模块 5 条关键设计决策，每条一行、以短横线开头，"
            f"内容要具体到字段名/函数名/文件名级别，不要空话。"
        )
        t0 = time.time()
        ms1 = ask_with_retry(llm, prompt1, want=5, max_tokens=700, temperature=0.6)[:6]
        print(f"[{nid} {title}] round1 -> {len(ms1)} 条 ({time.time() - t0:.1f}s)")

        anchor = ms1[0] if ms1 else title
        prompt2 = (
            f"{BACKGROUND}\n你负责「{title}」子模块（涉及：{kws}）。依赖模块决策：\n{ctx}\n\n"
            f"针对这条决策展开实现细节：{anchor}\n"
            f"给出 4 行具体细节（字段名、函数签名、文件布局），每行以短横线开头。"
        )
        t0 = time.time()
        ms2 = ask_with_retry(llm, prompt2, want=4, max_tokens=500, temperature=0.6)[:5]
        print(f"[{nid} {title}] round2 -> {len(ms2)} 条 ({time.time() - t0:.1f}s)")

        for k, text in enumerate(ms1 + ms2):
            memories.append({
                "memory_id": f"rm{mid:04d}", "node_id": nid, "text": text,
                "timestamp": base_ts + mid * 3600.0,
                "access_count": (mid * 7) % 6, "task_tag": tag,
            })
            produced[nid].append(text)
            mid += 1
        nodes.append({"node_id": nid, "topic": f"todo-cli·{title}",
                      "summary": f"{title}：围绕{kws}的设计决策与实现细节。",
                      "task_tag": tag})

    edges = [{"src": s, "dst": d, "type": t,
              "strength": EDGE_STRENGTH[t]} for s, d, t in EDGES]
    queries = [{"query_id": q, "cur_node": c, "text": txt, "task_tag": info[c]["tag"],
                "relevant_nodes": rel, "anchors": anc}
               for q, c, txt, rel, anc in QUERIES]

    # 天花板体检（POC 第一轮教训 2）：|relevant| 必须显著 ≤ K
    print("\n=== ground truth 天花板体检 ===")
    ok = True
    for q in queries:
        rel_ids = [m["memory_id"] for m in memories
                   if m["node_id"] in q["relevant_nodes"]
                   and any(a in m["text"] for a in q["anchors"])]
        q["relevant_ids"] = rel_ids
        ceiling = len(rel_ids) / K
        flag = "OK" if len(rel_ids) <= K else "超标!"
        if len(rel_ids) > K:
            ok = False
        print(f"{q['query_id']}: |relevant|={len(rel_ids)} (≤{K} {flag}) recall天花板={ceiling:.2f}")
    if not ok:
        print("!! 有查询的相关集超过 K，请收紧 anchors 后重跑（先改 QUERIES，不必重跑会话生成）")
        print("!! 本次仍落盘，但评测前必须修 anchors")

    data = {"nodes": nodes, "edges": edges, "memories": memories,
            "queries": queries,
            "meta": {"scenario": "todo-cli 多模块项目", "llm": "qwen-plus",
                     "n_nodes": len(nodes), "n_edges": len(edges),
                     "n_memories": len(memories), "generated_at": time.strftime("%F %T")}}
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n落盘 {OUT}：{len(nodes)} 节点 / {len(edges)} 边 / {len(memories)} 记忆 / {len(queries)} 查询")


if __name__ == "__main__":
    main()
