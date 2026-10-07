"""LoCoMo → 本系统 episode 格式适配器（真实公开数据集接入，D-049）。

为什么选 LoCoMo（ACL 2024, Snap Research）
-----------------------------------------
与本项目架构**结构同构**：

  LoCoMo 概念                  →  本系统概念
  ---------------------------------------------------------------
  session_k（一次会话）         →  topic 图节点（TopicNode）
  observation 中的每条事实      →  记忆条目（MemoryCandidate）
  observation 的 dia_id 溯源    →  owner_node 标记（事实属于哪个 session）
  qa.evidence（dia_id 列表）    →  relevant（**memory 级 ground truth**）
  qa.category（1~5）           →  scene（分层分析维度）

最关键的一点：**qa.evidence 给的是 dia_id，而每条 observation 事实自带 dia_id**，
因此可以直接推导出 **memory 级**的 ground truth，不需要人工标注、不需要猜。

数据规模（locomo10.json，公开版 10 个对话）
------------------------------------------
  session 272 个 ｜ observation 事实 2541 条 ｜ QA 1986 条
  跨源（evidence 命中 ≥2 个 session）QA 330 条，其中 multi-hop 类 269 条（占该类 95.4%）

  对比现有合成数据集（22 节点 / 490 记忆 / 50 episodes）：**规模提升一个数量级**。

category 映射（由跨源率实测确认）
---------------------------------
  1 = multi-hop     （282 条，95.4% 跨 session）★ 本课题主战场（多源选源）
  2 = temporal      （321 条）
  3 = open-domain   （96 条）
  4 = single-hop    （841 条，仅 0.2% 跨 session）
  5 = adversarial   （446 条，无 answer 字段 = 不可回答）

用法
----
    python -m analysis.locomo_adapter --conv 0                    # 只转第 0 个对话（跑通用）
    python -m analysis.locomo_adapter --conv all --out data/locomo_all.jsonl
    python -m analysis.locomo_adapter --conv 0 --stats-only       # 只看统计不落盘
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "data" / "datasets" / "locomo10.json"

#: category → scene 名（跨源率实测确认映射，见模块文档）
CATEGORY_SCENE = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}

#: 建边：每个 session 与「更早的 session」按文本相似度取 top-k 建边（方向：早 → 晚）
EDGE_TOPK = 3
#: 相似度 → 边类型分档（与 graph/models.py 的 SIMILARITY_BANDS 保持一致）
SIM_BANDS = ((0.85, "derives_from"), (0.72, "depends_on"), (0.58, "references"))


# ------------------------------------------------------------------ 工具

def _parse_date(s: str, fallback_idx: int) -> float:
    """解析 LoCoMo 的 session 日期（形如 "7 May 2023"）；失败则回退递增时间戳。

    回退策略**保持相对时序**（用 session 序号 × 1 天），保证 time 维相对新旧正确。
    """
    if not s:
        return float(fallback_idx * 86400)
    s = s.strip()
    for fmt in ("%d %B %Y", "%d %b %Y", "%B %d, %Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    # 回退：只取年份/月份等数字不可靠，直接用序号 —— 时序正确即可
    return float(fallback_idx * 86400)


def _tokens(text: str) -> set[str]:
    """轻量英文分词（只为建边算相似度，不进 BM25 主链路）。"""
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _infer_edge_type(sim: float) -> str:
    for thr, t in SIM_BANDS:
        if sim >= thr:
            return t
    return "similar_to"


# ------------------------------------------------------------------ 主体

def build_world(conv: dict, conv_idx: int, edge_topk: int = EDGE_TOPK) -> dict:
    """把一个 LoCoMo 对话转成本系统的「世界」：nodes / edges / corpus / 索引。

    返回 {nodes, edges, corpus, dia2mem, sess_of_mem, session_ts, session_ids}
    """
    raw_conv = conv["conversation"]
    obs = conv.get("observation", {})
    summ = conv.get("session_summary", {})

    # ① session 列表（按序号排序）
    sess_keys = sorted(
        [k for k in raw_conv
         if k.startswith("session_") and not k.endswith(("_date_time", "_observation", "_summary"))],
        key=lambda x: int(x.split("_")[1]))
    session_ids = [f"c{conv_idx}_s{int(k.split('_')[1])}" for k in sess_keys]

    # ② session 时间戳
    session_ts: list[float] = []
    for i, k in enumerate(sess_keys):
        dt_key = f"{k}_date_time"
        session_ts.append(_parse_date(raw_conv.get(dt_key, ""), i))

    # ③ 记忆条目（observation 事实）+ dia_id → memory_id 映射
    corpus: list[dict] = []
    dia2mem: dict[str, str] = {}
    sess_of_mem: dict[str, str] = {}
    for si, k in enumerate(sess_keys):
        nid = session_ids[si]
        ts = session_ts[si]
        ob = obs.get(f"{k}_observation", {}) or {}
        n = 0
        for speaker, facts in ob.items():
            for item in facts:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                text, dia_id = str(item[0]).strip(), str(item[1])
                if not text:
                    continue
                mid = f"{nid}_m{n}"
                n += 1
                corpus.append({
                    "memory_id": mid,
                    "text": text,
                    "task_tag": "locomo",
                    "owner_node": nid,
                    "timestamp": ts,
                    "_dia_id": dia_id,          # 内部字段，runner 不消费
                    "_speaker": speaker,
                })
                dia2mem.setdefault(dia_id, mid)
                sess_of_mem[mid] = nid

    # ④ 图节点：topic = session 摘要（无摘要则用该 session 首条事实）
    nodes: list[dict] = []
    for si, k in enumerate(sess_keys):
        nid = session_ids[si]
        topic = str(summ.get(f"{k}_summary", "") or "").strip()
        if not topic:
            first = next((c["text"] for c in corpus if c["owner_node"] == nid), "")
            topic = first[:80] or f"session {si + 1}"
        nodes.append({
            "node_id": nid,
            "topic": topic,
            "summary": topic,
            "session_id": f"conv{conv_idx}",
            "created_at": session_ts[si],
        })

    # ⑤ 边：每个 session 向「更早的 session」建 top-k 边（方向：早 → 晚 = 被依赖方 → 依赖方）
    sess_text: dict[int, set[str]] = {}
    for si, nid in enumerate(session_ids):
        toks: set[str] = set()
        for c in corpus:
            if c["owner_node"] == nid:
                toks |= _tokens(c["text"])
        sess_text[si] = toks

    edges: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for si in range(1, len(session_ids)):
        cands = []
        for sj in range(si):
            sim = _jaccard(sess_text[si], sess_text[sj])
            cands.append((sim, sj))
        cands.sort(key=lambda x: (-x[0], -x[1]))     # 相似优先；同分时取更近的 session
        for sim, sj in cands[:max(1, edge_topk)]:
            src, dst = session_ids[sj], session_ids[si]
            if (src, dst) in seen:
                continue
            seen.add((src, dst))
            edges.append({"src": src, "dst": dst, "type": _infer_edge_type(sim),
                          "similarity": round(float(sim), 4)})

    return {
        "nodes": nodes, "edges": edges, "corpus": corpus,
        "dia2mem": dia2mem, "sess_of_mem": sess_of_mem,
        "session_ts": session_ts, "session_ids": session_ids,
    }


def build_episodes(conv: dict, conv_idx: int, edge_topk: int = EDGE_TOPK,
                   with_evidence_only: bool = True) -> list[dict]:
    """把一个 LoCoMo 对话转成若干 episode（每个 QA 一个）。"""
    w = build_world(conv, conv_idx, edge_topk)
    corpus = w["corpus"]
    now = max(w["session_ts"]) if w["session_ts"] else 0.0
    # cur_node = 最新 session（模拟"对话进行到现在，问这个问题"）
    cur_node = w["session_ids"][-1] if w["session_ids"] else "main"

    episodes: list[dict] = []
    for qi, qa in enumerate(conv.get("qa", [])):
        cat = int(qa.get("category", 0))
        scene = CATEGORY_SCENE.get(cat, f"cat{cat}")
        evidence = qa.get("evidence") or []
        # memory 级 ground truth：evidence 是 dia_id，直接映射到记忆条目
        relevant = sorted({w["dia2mem"][d] for d in evidence if d in w["dia2mem"]})
        if with_evidence_only and not relevant:
            continue                       # adversarial（不可回答）无 ground truth，跳过
        rel_sources = sorted({w["sess_of_mem"][m] for m in relevant})
        episodes.append({
            "episode_id": f"lc{conv_idx}_{qi:04d}",
            "task_tag": "locomo",
            "user_id": f"conv{conv_idx}",
            "scenario": "long_term_conversation",
            "business": "locomo",
            "query": qa.get("question", ""),
            "cur_node": cur_node,
            "now": now,
            "nodes": w["nodes"],
            "edges": w["edges"],
            # 去掉内部字段，保持与既有 dataset 格式一致
            "corpus": [{k: v for k, v in c.items() if not k.startswith("_")} for c in corpus],
            "relevant": relevant,
            "relevant_sources": rel_sources,
            "scene": scene,
            # 附加：答案文本（供 LLM-judge 端到端评估，当前指标不用）
            "answer": qa.get("answer"),
            "n_evidence_sessions": len(rel_sources),
        })
    return episodes


def load_locomo(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ CLI

def main() -> int:
    ap = argparse.ArgumentParser(description="LoCoMo → 本系统 episode 适配器")
    ap.add_argument("--src", default=str(DEFAULT_SRC), help="locomo10.json 路径")
    ap.add_argument("--conv", default="0", help="对话序号（0~9）或 all")
    ap.add_argument("--out", default="", help="输出 jsonl（默认 data/locomo_conv<N>.jsonl）")
    ap.add_argument("--edge-topk", type=int, default=EDGE_TOPK, help="每 session 建边 top-k")
    ap.add_argument("--stats-only", action="store_true", help="只打印统计，不落盘")
    args = ap.parse_args()

    data = load_locomo(Path(args.src))
    if args.conv == "all":
        idxs = list(range(len(data)))
        out_path = Path(args.out or ROOT / "data" / "locomo_all.jsonl")
    else:
        idxs = [int(args.conv)]
        out_path = Path(args.out or ROOT / "data" / f"locomo_conv{idxs[0]}.jsonl")

    all_eps: list[dict] = []
    for ci in idxs:
        eps = build_episodes(data[ci], ci, edge_topk=args.edge_topk)
        all_eps.extend(eps)
        w = build_world(data[ci], ci, args.edge_topk)
        print(f"[conv {ci}] sessions={len(w['session_ids'])} 记忆={len(w['corpus'])} "
              f"边={len(w['edges'])} episodes={len(eps)}")

    if all_eps:
        sizes = {}
        for e in all_eps:
            sizes.setdefault(e["scene"], 0)
            sizes[e["scene"]] += 1
        print(f"\n合计 episodes={len(all_eps)}  分层={sizes}")
        cross = sum(1 for e in all_eps if e["n_evidence_sessions"] >= 2)
        print(f"跨源(evidence 命中 ≥2 session) = {cross} ({cross / len(all_eps) * 100:.1f}%)")
        avg_rel = sum(len(e["relevant"]) for e in all_eps) / len(all_eps)
        print(f"平均 |relevant| = {avg_rel:.2f}  （评估 K 应参考此值）")

    if not args.stats_only and all_eps:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for e in all_eps:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"\n已写出: {out_path}  ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
