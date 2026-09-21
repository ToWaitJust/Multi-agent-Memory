"""Topic 图存储（§3）—— 强制 DAG、typed 边、邻域展开、版本折叠、持久化。

关键约束：
1. **强制 DAG**：加边前做可达性检查，若成环则**拒绝建边**并降级为 `references`
   （`status=cycle_rejected`，**不参与调度邻域**，仅在溯源视图可见）—— P0 手段。
2. **深度封顶**：邻域展开默认 `depth=1` —— P1 手段。
3. **strength 静态**：本模块只做"建边时定 / 反馈微调 / summary 重算"三种写入，
   **绝不**由 query 计算 —— 由 `tests/test_graph.py::test_edge_strength_is_static` 守护。
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Iterable, Optional

from .models import (
    EDGE_DEFAULT_STRENGTH,
    EDGE_STATUS_ACTIVE,
    EDGE_STATUS_CYCLE_REJECTED,
    MAIN_NODE,
    STAGE_COARSE,
    STAGE_REFINED,
    TopicEdge,
    TopicNode,
    new_id,
)

#: 反馈微调步长（bandit 式一小步；与 weights.update_with_feedback 的 0.1 同哲学）
FEEDBACK_STEP = 0.1
#: 反馈微调后向默认强度回归的强度（REQ-405 同源的正则思想，防漂移）
REGULARIZE_TO_DEFAULT = 0.15


class CycleError(ValueError):
    """加边会形成环（仅在 allow_downgrade=False 时抛出）。"""


class GraphStore:
    """topic 图（内存为主 + JSONL 持久化，原子写）。"""

    def __init__(self, user_id: str = "", root_dir: str | Path | None = None):
        self.user_id = user_id
        self.root_dir = Path(root_dir) if root_dir else None
        self._nodes: dict[str, TopicNode] = {}
        self._edges: dict[str, TopicEdge] = {}
        self._out: dict[str, list[str]] = {}   # src -> [edge_id]
        self._in: dict[str, list[str]] = {}    # dst -> [edge_id]
        self._ensure_main()

    # ------------------------------------------------------------------ 节点

    def _ensure_main(self) -> TopicNode:
        if MAIN_NODE not in self._nodes:
            self._nodes[MAIN_NODE] = TopicNode(
                node_id=MAIN_NODE, topic="主线", created_at=time.time(), summary_at=time.time()
            )
        return self._nodes[MAIN_NODE]

    def upsert_node(self, node: TopicNode) -> TopicNode:
        if not node.node_id:
            node.node_id = new_id("n")
        if not node.created_at:
            node.created_at = time.time()
        self._nodes[node.node_id] = node
        self._out.setdefault(node.node_id, [])
        self._in.setdefault(node.node_id, [])
        return node

    def create_node(self, topic: str, emb: Optional[list[float]] = None,
                    session_id: str = "", summary: str = "",
                    node_id: str | None = None, root_id: str | None = None) -> TopicNode:
        node = TopicNode(
            node_id=node_id or new_id("n"),
            topic=topic, summary=summary, emb=list(emb) if emb is not None else None,
            owner_user=self.user_id, session_id=session_id,
            created_at=time.time(), root_id=root_id,
            summary_at=time.time() if summary else 0.0,
        )
        return self.upsert_node(node)

    def get_node(self, node_id: str) -> Optional[TopicNode]:
        return self._nodes.get(node_id)

    def nodes(self) -> list[TopicNode]:
        return list(self._nodes.values())

    def node_ids(self) -> list[str]:
        return list(self._nodes.keys())

    def bump_memory_count(self, node_id: str, n: int = 1) -> int:
        node = self._nodes.get(node_id)
        if node is None:
            return 0
        node.n_memories += n
        return node.n_memories

    # ------------------------------------------------------------------ 边

    def add_edge(self, src: str, dst: str, type: str, strength: float | None = None,
                 evidence: str = "", stage: str = STAGE_COARSE,
                 similarity: float = 0.0, allow_downgrade: bool = True
                 ) -> tuple[TopicEdge, str]:
        """加边。返回 (edge, reason)。

        P0 环检测：若 `dst` 已可达 `src`（即新边会闭合环），则：
          - `allow_downgrade=True`（默认）→ 降级为 `references` 且 `status=cycle_rejected`；
          - `allow_downgrade=False` → 抛 CycleError。
        自环一律拒绝（同样降级/抛错）。
        """
        if src not in self._nodes:
            raise KeyError(f"源节点不存在: {src}")
        if dst not in self._nodes:
            raise KeyError(f"目标节点不存在: {dst}")
        if type not in EDGE_DEFAULT_STRENGTH:
            raise ValueError(f"未知边类型: {type}")

        reason = "ok"
        status = EDGE_STATUS_ACTIVE
        is_cycle = (src == dst) or self.reachable(dst, src)
        if is_cycle:
            reason = "cycle_rejected"
            status = EDGE_STATUS_CYCLE_REJECTED
            type = "references"                       # 降级为弱引用，不入邻域
            strength = 0.0
            evidence = (evidence + f" | P0环检测拒绝：{dst} 已可达 {src}，降级 references 不入邻域").strip(" |")
            if not allow_downgrade:
                raise CycleError(f"加边 {src}->{dst} 会形成环")

        # 同向同类型去重：已存在则更新证据（不重复建边）
        for eid in self._out.get(src, []):
            e = self._edges[eid]
            if e.dst == dst and e.type == type:
                if evidence:
                    e.evidence = evidence
                e.updated_at = time.time()
                self._reindex()
                return e, "updated"

        if strength is None:
            strength = EDGE_DEFAULT_STRENGTH[type]
        edge = TopicEdge(
            edge_id=new_id("e"), src=src, dst=dst, type=type, strength=strength,
            created_at=time.time(), updated_at=time.time(),
            evidence=evidence, stage=stage, status=status, similarity=similarity,
        )
        self._edges[edge.edge_id] = edge
        self._reindex()
        return edge, reason

    def get_edge(self, edge_id: str) -> Optional[TopicEdge]:
        return self._edges.get(edge_id)

    def edges(self) -> list[TopicEdge]:
        return list(self._edges.values())

    def active_edges(self) -> list[TopicEdge]:
        return [e for e in self._edges.values() if e.in_neighborhood]

    def edges_of(self, node_id: str, direction: str = "both") -> list[TopicEdge]:
        out: list[TopicEdge] = []
        if direction in ("out", "both"):
            out += [self._edges[i] for i in self._out.get(node_id, [])]
        if direction in ("in", "both"):
            out += [self._edges[i] for i in self._in.get(node_id, [])]
        return out

    def _reindex(self) -> None:
        self._out = {n: [] for n in self._nodes}
        self._in = {n: [] for n in self._nodes}
        for e in self._edges.values():
            self._out.setdefault(e.src, []).append(e.edge_id)
            self._in.setdefault(e.dst, []).append(e.edge_id)

    # -------------------------------------------------------- 图算法（DAG）

    def reachable(self, src: str, dst: str, max_hops: int = 64) -> bool:
        """src 是否能沿 active 边到达 dst（BFS，深度封顶防病态图）。"""
        if src == dst:
            return True
        seen = {src}
        q = deque([(src, 0)])
        while q:
            cur, d = q.popleft()
            if d >= max_hops:
                continue
            for eid in self._out.get(cur, []):
                e = self._edges[eid]
                if not e.in_neighborhood:
                    continue
                if e.dst == dst:
                    return True
                if e.dst not in seen:
                    seen.add(e.dst)
                    q.append((e.dst, d + 1))
        return False

    def topo_order(self) -> list[str]:
        """Kahn 拓扑排序（仅 active 边）。非 DAG 时抛 CycleError。"""
        indeg = {n: 0 for n in self._nodes}
        for e in self.active_edges():
            indeg[e.dst] = indeg.get(e.dst, 0) + 1
        q = deque(sorted([n for n, d in indeg.items() if d == 0]))
        order: list[str] = []
        while q:
            cur = q.popleft()
            order.append(cur)
            for eid in self._out.get(cur, []):
                e = self._edges[eid]
                if not e.in_neighborhood:
                    continue
                indeg[e.dst] -= 1
                if indeg[e.dst] == 0:
                    q.append(e.dst)
        if len(order) != len(self._nodes):
            raise CycleError("图中存在环（active 边子图非 DAG）")
        return order

    def assert_dag(self) -> None:
        self.topo_order()

    def neighbors(self, node_id: str, depth: int = 1, strength_min: float = 0.0,
                  edge_types: Optional[Iterable[str]] = None,
                  stronger_first: bool = True,
                  direction: str = "out") -> list[tuple[str, TopicEdge]]:
        """邻域展开（默认 depth=1）。返回 [(neighbor_node_id, edge)]，按边强度降序。

        `direction`：
        - `"out"`（默认）= 后继（本节点是信息源，供下游节点取用）
        - `"in"`  = **前驱**（本节点依赖的上游 = **Stage A 的源节点**，见 `predecessors()`）

        边语义统一为「被依赖方 → 依赖方」，故当前节点要取的源 = 它的**入边起点**。
        depth>1 时**首次到达某节点的路径保留最强一条**（P2 去重）。
        """
        if direction not in ("in", "out"):
            raise ValueError("direction 必须是 'in' 或 'out'")
        adj = self._in if direction == "in" else self._out
        other = (lambda e: e.src) if direction == "in" else (lambda e: e.dst)
        types = set(edge_types) if edge_types else None
        best_edge: dict[str, TopicEdge] = {}
        frontier = [(node_id, 0)]
        while frontier:
            cur, d = frontier.pop(0)
            if d >= max(1, depth):
                continue
            for eid in adj.get(cur, []):
                e = self._edges[eid]
                if not e.in_neighborhood:
                    continue
                if e.strength < strength_min:
                    continue
                if types is not None and e.type not in types:
                    continue
                nid = other(e)
                prev = best_edge.get(nid)
                if prev is None or e.strength > prev.strength:
                    best_edge[nid] = e
                    frontier.append((nid, d + 1))
        best_edge.pop(node_id, None)
        items = list(best_edge.items())
        if stronger_first:
            items.sort(key=lambda kv: kv[1].strength, reverse=True)
        return items

    def predecessors(self, node_id: str, depth: int = 1, strength_min: float = 0.0,
                     edge_types: Optional[Iterable[str]] = None
                     ) -> list[tuple[str, TopicEdge]]:
        """**Stage A 的邻域**：本节点依赖的上游源节点（沿入边展开，强边优先）。"""
        return self.neighbors(node_id, depth=depth, strength_min=strength_min,
                              edge_types=edge_types, stronger_first=True, direction="in")

    # ------------------------------------------------------------ 版本折叠

    def fold_versions(self) -> dict[str, str]:
        """P3 兜底：同 `root_id` 的节点只留最新版本为主版本，其余折叠。

        返回 {被折叠节点: 主版本节点}；无 root_id 的节点恒为主版本。
        """
        groups: dict[str, list[TopicNode]] = {}
        for n in self._nodes.values():
            key = n.root_id or n.node_id
            groups.setdefault(key, []).append(n)
        folded: dict[str, str] = {}
        for key, members in groups.items():
            if len(members) < 2:
                continue
            members.sort(key=lambda n: (n.version, n.created_at))
            primary = members[-1]
            for n in members[:-1]:
                folded[n.node_id] = primary.node_id
        return folded

    # ------------------------------------------------------- 强度更新（静态）

    def adjust_edge_strength(self, edge_id: str, reward: float) -> Optional[float]:
        """用户反馈微调边强度（唯一允许改 strength 的入口之一）。

        与 `weights.update_with_feedback` 同哲学：一小步 + 向默认强度回归（REQ-405）。
        **不接受 query 参数** —— 从签名上杜绝"按 query 改边强度"。
        """
        e = self._edges.get(edge_id)
        if e is None or e.status != EDGE_STATUS_ACTIVE:
            return None
        base = EDGE_DEFAULT_STRENGTH.get(e.type, 0.5)
        new = e.strength + FEEDBACK_STEP * reward
        new = new + REGULARIZE_TO_DEFAULT * (base - new)   # 向默认回归，防漂移
        e.strength = 0.0 if new < 0.0 else (1.0 if new > 1.0 else float(new))
        e.updated_at = time.time()
        e.evidence = (e.evidence + f" | 反馈微调 reward={reward:+.0f}").strip(" |")
        return e.strength

    def recompute_strength_from_summary(self, edge_id: str, similarity: float,
                                        type: Optional[str] = None) -> Optional[float]:
        """summary 重算后刷新边强度（唯一允许改 strength 的入口之二）。"""
        e = self._edges.get(edge_id)
        if e is None:
            return None
        if type:
            e.type = type
        base = EDGE_DEFAULT_STRENGTH.get(e.type, 0.5)
        # 用相似度在"默认强度"附近做温和校正（保留 typed 先验的主导地位）
        e.strength = max(0.0, min(1.0, base * (0.6 + 0.4 * max(0.0, similarity))))
        e.similarity = similarity
        e.updated_at = time.time()
        e.stage = STAGE_REFINED
        e.evidence = (e.evidence + f" | summary 重算 sim={similarity:.3f}").strip(" |")
        return e.strength

    def mark_refined(self, edge_id: str, evidence: str) -> None:
        e = self._edges.get(edge_id)
        if e is None:
            return
        e.stage = STAGE_REFINED
        e.updated_at = time.time()
        if evidence:
            e.evidence = (e.evidence + " | " + evidence).strip(" |")

    # ------------------------------------------------------------ 持久化

    def persist(self) -> None:
        if self.root_dir is None:
            return
        d = Path(self.root_dir)
        d.mkdir(parents=True, exist_ok=True)
        _atomic_write(d / "nodes.jsonl",
                      "\n".join(json.dumps(n.to_dict(), ensure_ascii=False)
                                for n in self._nodes.values()) + "\n")
        _atomic_write(d / "edges.jsonl",
                      "\n".join(json.dumps(e.to_dict(), ensure_ascii=False)
                                for e in self._edges.values()) + "\n")

    def load(self) -> "GraphStore":
        if self.root_dir is None:
            return self
        d = Path(self.root_dir)
        nf, ef = d / "nodes.jsonl", d / "edges.jsonl"
        if nf.exists():
            for line in _read_jsonl(nf):
                node = TopicNode.from_dict(line)
                self._nodes[node.node_id] = node
        if ef.exists():
            for line in _read_jsonl(ef):
                edge = TopicEdge.from_dict(line)
                self._edges[edge.edge_id] = edge
        self._ensure_main()
        self._reindex()
        return self

    # ------------------------------------------------------------ 统计

    def stats(self) -> dict:
        act = self.active_edges()
        n = max(len(self._nodes), 1)
        return {
            "n_nodes": len(self._nodes),
            "n_edges": len(self._edges),
            "n_active_edges": len(act),
            "n_cycle_rejected": len(self._edges) - len(act),
            "avg_out_degree": round(len(act) / n, 3),
            "is_dag": self._is_dag_safe(),
            "edge_strength_var": _variance([e.strength for e in act]),
            "edge_type_hist": _hist([e.type for e in act]),
        }

    def _is_dag_safe(self) -> bool:
        try:
            self.topo_order()
            return True
        except CycleError:
            return False


# ------------------------------------------------------------------ helpers


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        pass
    return out


def _variance(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return round(sum((x - m) ** 2 for x in xs) / (len(xs) - 1), 6)


def _hist(xs: list[str]) -> dict[str, int]:
    h: dict[str, int] = {}
    for x in xs:
        h[x] = h.get(x, 0) + 1
    return h
