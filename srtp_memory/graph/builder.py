"""两阶段建边（§3.3）—— 解决"节点初建时还没有 summary"的冷启动问题。

| 阶段 | 时机 | 依据 | 特点 |
|---|---|---|---|
| **粗识别** | 建节点时**立即** | `emb(topic 标题 + 首轮 query)` | 快，保证节点一诞生就有边可用 |
| **精识别** | `summary` 生成后 | `emb(summary)`（+ 可选 LLM 判定边类型） | 准，**可覆盖**粗识别，须写 `evidence` 溯源 |

边方向约定：**被依赖方 → 依赖方**。新节点在语义上"依赖"既有节点，
故 `src = 既有节点`，`dst = 新节点`。

边密度双阀：① 只与已有节点比（O(n)，n=节点数）；② 只保留 top-k 建边。
判据（Q6c）：精识别覆盖率应 < 30%，否则说明粗识别太差。

⚠️ `summary` 硬边界：**只用于建边与源打分，绝不用于记忆召回**（召回走实体池原文）。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from typing import Callable, Iterable, Optional

from .graph_store import GraphStore
from .models import (
    EDGE_DEFAULT_STRENGTH,
    MAIN_NODE,
    STAGE_COARSE,
    STAGE_REFINED,
    TopicEdge,
    TopicNode,
    cosine,
    infer_edge_type,
)

#: 无外部 embedding 时的本地确定性向量（字符 bigram 哈希）——保证离线/单测可跑。
HASH_DIM = 128


def hash_embed(text: str, dim: int = HASH_DIM) -> list[float]:
    """字符 bigram 哈希向量（同主题 → 余弦偏高）。POC `fake_embed.py` 同源做法。"""
    v = [0.0] * dim
    t = (text or "").strip()
    if not t:
        return v
    grams = [t[i:i + 2] for i in range(max(len(t) - 1, 1))]
    for g in grams:
        h = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16)
        v[h % dim] += 1.0
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else v


class EdgeBuilder:
    """建边器（粗/精两阶段 + 可选 LLM 判型 + summary 失效重算）。"""

    def __init__(self, store: GraphStore, cfg=None,
                 embed_fn: Optional[Callable[[str], list[float]]] = None,
                 llm_fn: Optional[Callable[[str], str]] = None):
        self.store = store
        self.cfg = cfg
        self.embed_fn = embed_fn
        self.llm_fn = llm_fn
        self.refined_count = 0
        self.coarse_count = 0

    # ------------------------------------------------------------ embedding

    def embed(self, text: str) -> list[float]:
        """优先用外部 embed_fn（真实 DashScope）；不可用/失败 → 本地哈希向量兜底。"""
        if self.embed_fn is not None:
            try:
                v = self.embed_fn(text)
                if v:
                    return list(v)
            except Exception:  # noqa: BLE001 - 兜底不阻断建边
                pass
        return hash_embed(text)

    # ------------------------------------------------------------ 邻域参数

    @property
    def topk(self) -> int:
        return int(_cfg(self.cfg, "edge_topk", 4))

    @property
    def edge_types(self) -> Optional[Iterable[str]]:
        return _cfg(self.cfg, "edge_types", None)

    # ------------------------------------------------------------ 粗识别

    def on_node_created(self, node: TopicNode, first_query: str = "",
                        auto_link: bool = True) -> list[TopicEdge]:
        """节点创建时立即建边（粗识别）。

        - 缓存 `node.emb`（供源打分复用，**零额外成本**）
        - 与既有节点比对，取 top-k 建边；类型由相似度分档确定性推断（或 LLM 判型）
        """
        basis = f"{node.topic} {first_query}".strip()
        if node.emb is None:
            node.emb = self.embed(basis)
        created: list[TopicEdge] = []
        if auto_link:
            created = self._link(node, basis, stage=STAGE_COARSE)
            self.coarse_count += len(created)
        self.store.persist()
        return created

    def _link(self, node: TopicNode, basis: str, stage: str,
              llm_judge: bool = False) -> list[TopicEdge]:
        """与既有节点比对并建边（粗/精共用）。"""
        cands: list[tuple[str, float]] = []
        # 排除自身 + 版本链同根 + main（main 作为全局上下文源，不参与 topic 建边）
        for other in self.store.nodes():
            if other.node_id in (node.node_id, MAIN_NODE):
                continue
            if node.root_id and other.root_id == node.root_id:
                continue
            sim = cosine(node.emb, other.emb)
            if sim is None:
                continue
            cands.append((other.node_id, sim))
        if not cands:
            return []
        cands.sort(key=lambda kv: kv[1], reverse=True)
        cands = cands[:self.topk]

        types = self._judge_types(node, cands) if llm_judge else None
        out: list[TopicEdge] = []
        for nid, sim in cands:
            t = (types or {}).get(nid) or infer_edge_type(sim)
            ev = json.dumps(
                {"stage": stage, "basis": basis[:120], "similarity": round(sim, 4),
                 "direction": "existing->new"},
                ensure_ascii=False,
            )
            # 方向：既有节点（信息源）→ 新节点（依赖方）
            edge, reason = self.store.add_edge(
                src=nid, dst=node.node_id, type=t,
                evidence=ev, stage=stage, similarity=sim,
            )
            if reason in ("ok", "updated"):
                out.append(edge)
        return out

    # ------------------------------------------------------------ 精识别

    def refine(self, node: TopicNode, summary: str = "",
               llm_judge: bool = True) -> dict:
        """summary 生成后精识别：可覆盖粗识别结果，**必须写 evidence 溯源**。

        返回 {"updated": n, "unchanged": n, "coverage": bool}。
        """
        text = summary or node.summary
        if not text:
            return {"updated": 0, "unchanged": 0, "coverage": False}
        if summary:
            node.summary = summary
        node.summary_at = time.time()
        node.emb = self.embed(f"{node.topic} {text}")
        before = {e.src for e in self.store.edges_of(node.node_id, "in") if e.in_neighborhood}
        new_edges = self._link(node, f"{node.topic} {text}", stage=STAGE_REFINED,
                               llm_judge=llm_judge and self.llm_fn is not None)
        after = {e.src for e in self.store.edges_of(node.node_id, "in") if e.in_neighborhood}
        changed = before.symmetric_difference(after)
        for e in new_edges:
            self.store.mark_refined(e.edge_id, f"精识别依据 summary({len(text)} 字)")
            self.refined_count += 1
        self.store.persist()
        return {"updated": len(changed), "unchanged": len(before & after),
                "coverage": len(changed) == 0}

    def refine_coverage(self) -> float:
        """精识别覆盖率 = refined 边 / active 边（判据 <30%）。"""
        act = self.store.active_edges()
        if not act:
            return 0.0
        return round(sum(1 for e in act if e.stage == STAGE_REFINED) / len(act), 4)

    # ------------------------------------------------------------ summary 失效

    def needs_summary_refresh(self, node: TopicNode,
                              texts: Optional[list[str]] = None) -> bool:
        """summary 重算触发条件：新增记忆 ≥ `summary_refresh_n` 条。"""
        _ = texts
        thr = int(_cfg(self.cfg, "summary_refresh_n", 20))
        if node.summary_at <= 0:
            return node.n_memories >= max(1, min(thr, 3))   # 首次 summary 早点生成
        return node.n_memories >= thr

    def affected_edges(self, node: TopicNode) -> list[TopicEdge]:
        """summary 重算后需要重新判定的边（该节点所有 active 邻接边）。"""
        return [e for e in self.store.edges_of(node.node_id) if e.in_neighborhood]

    # ------------------------------------------------------------ LLM 判型

    def _judge_types(self, node: TopicNode,
                     cands: list[tuple[str, float]]) -> Optional[dict[str, str]]:
        """可选：一次 LLM 调用批量判定边类型（结构化 JSON + 宽容解析 + 失败降级）。"""
        if self.llm_fn is None or not cands:
            return None
        lines = []
        for i, (nid, _sim) in enumerate(cands, 1):
            other = self.store.get_node(nid)
            lines.append(f"{i}. 新节点主题：{node.topic} | 候选节点主题：{other.topic if other else nid}")
        prompt = (
            "下面是若干组「新节点 / 候选节点」，请判断它们的关系类型，只输出 JSON。\n"
            "可选类型：derives_from(派生) / depends_on(依赖) / references(引用) / similar_to(相近)。\n"
            + "\n".join(lines)
            + '\n输出格式：{"1":"depends_on","2":"references"}'
        )
        try:
            raw = self.llm_fn(prompt) or ""
            m = re.search(r"\{[^{}]*\}", raw)
            if not m:
                return None
            data = json.loads(m.group(0))
        except Exception:  # noqa: BLE001 - 判型失败退回相似度分档
            return None
        out: dict[str, str] = {}
        for i, (nid, _sim) in enumerate(cands, 1):
            t = data.get(str(i))
            if t in EDGE_DEFAULT_STRENGTH:
                out[nid] = t
        return out or None


# ------------------------------------------------------------------ helpers


def _cfg(c, name: str, default):
    if c is None:
        return default
    if isinstance(c, dict):
        return c.get(name, default)
    return getattr(c, name, default)
