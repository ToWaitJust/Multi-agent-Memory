"""Topic 图数据模型（§2.1 / §2.2）。

节点 = **一次主题会话**（不是记忆条目 —— 这是与知识图谱的根本区别）；
边 = typed 有向关系，其 `strength` 是**静态**的结构先验（建边时定，不 per-query 重算）。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# ---- 常量 ----

MAIN_NODE = "main"

EDGE_TYPES: tuple[str, ...] = ("derives_from", "depends_on", "references", "similar_to")

#: typed 边的**唯一作用** = 默认强度初始化（不是分类学装饰）。
EDGE_DEFAULT_STRENGTH: dict[str, float] = {
    "derives_from": 0.9,
    "depends_on": 0.8,
    "references": 0.5,
    "similar_to": 0.3,
}

#: 相似度分档 → 边类型（无 LLM 时的确定性判定，见决策 D-016）。
SIMILARITY_BANDS: tuple[tuple[float, str], ...] = (
    (0.85, "derives_from"),
    (0.72, "depends_on"),
    (0.58, "references"),
)

#: 边状态：active 参与调度邻域；cycle_rejected 为 P0 环检测降级产物，不参与邻域。
EDGE_STATUS_ACTIVE = "active"
EDGE_STATUS_CYCLE_REJECTED = "cycle_rejected"

#: 建边阶段
STAGE_COARSE = "coarse"
STAGE_REFINED = "refined"


def infer_edge_type(similarity: float) -> str:
    """按相似度分档推断边类型（确定性，可复现；无 LLM 成本）。"""
    for thr, t in SIMILARITY_BANDS:
        if similarity >= thr:
            return t
    return "similar_to"


@dataclass
class TopicNode:
    """topic 会话节点。`emb` 仅用于建边与源打分，**绝不进 retriever**。"""

    node_id: str
    topic: str = ""
    summary: str = ""
    emb: Optional[list[float]] = None      # topic(+summary) 的 embedding，建节点时缓存
    owner_user: str = ""
    session_id: str = ""
    created_at: float = 0.0
    summary_at: float = 0.0                # 0 = 尚未生成 summary
    version: int = 1
    root_id: Optional[str] = None          # 同 root_id 视为同一 topic 的不同版本
    status: str = "active"                 # active / archived
    n_memories: int = 0                    # 该节点已产出记忆条数（summary 重算触发用）

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TopicNode":
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in allowed})


@dataclass
class TopicEdge:
    """typed 有向边。`strength` 是静态量：只在建边 / 反馈 / summary 重算三时机变化。"""

    edge_id: str
    src: str
    dst: str
    type: str
    strength: float
    created_at: float = 0.0
    updated_at: float = 0.0
    evidence: str = ""
    stage: str = STAGE_COARSE
    status: str = EDGE_STATUS_ACTIVE
    similarity: float = 0.0                # 建边时的相似度（溯源/复算用）

    def __post_init__(self) -> None:
        if self.type not in EDGE_DEFAULT_STRENGTH:
            raise ValueError(f"未知边类型: {self.type}（可用 {EDGE_TYPES}）")
        self.strength = _clamp01(self.strength)

    @property
    def in_neighborhood(self) -> bool:
        """是否参与 Stage A 邻域召回（环检测降级的边不参与，仅溯源可见）。"""
        return self.status == EDGE_STATUS_ACTIVE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TopicEdge":
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in allowed})


@dataclass
class SourceRef:
    """Stage A 输出的一个源节点（附分项分，便于埋点与论文表格）。"""

    node_id: str
    strength: float = 0.0        # 静态边强度
    relevance: float = 0.0       # 动态 query 相关度
    src_score: float = 0.0       # λ·strength + (1−λ)·relevance
    quota: int = 0               # 硬配额（quota_mode=strength 时有意义）
    weight: float = 0.0          # 软偏置用的归一化权重
    via_edge: str = ""           # 经由哪条边（溯源）

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceSelection:
    """Stage A 的完整产物（ScopeSpec 的可解释版本）。"""

    cur_node: str
    sources: list[SourceRef] = field(default_factory=list)
    mode: str = "soft_bias"
    lambda_: float = 0.7
    depth: int = 1
    degraded_static: bool = False     # 是否因 rel 方差过低退化为纯静态
    rejected_edges: list[str] = field(default_factory=list)   # 环检测拒绝的 edge_id
    folded: dict[str, str] = field(default_factory=dict)      # 版本折叠映射
    reason: str = ""

    @property
    def node_ids(self) -> list[str]:
        return [s.node_id for s in self.sources]

    def score_of(self, node_id: str) -> float:
        for s in self.sources:
            if s.node_id == node_id:
                return s.src_score
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cur_node": self.cur_node,
            "mode": self.mode,
            "lambda": self.lambda_,
            "depth": self.depth,
            "degraded_static": self.degraded_static,
            "n_sources": len(self.sources),
            "sources": [s.to_dict() for s in self.sources],
            "rejected_edges": list(self.rejected_edges),
            "folded": dict(self.folded),
            "reason": self.reason,
        }


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def cosine(a: Optional[list[float]], b: Optional[list[float]]) -> Optional[float]:
    """纯 Python 余弦相似度（不依赖 numpy，便于单测；节点数少，性能足够）。"""
    if a is None or b is None or len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return None
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return None
    return dot / ((na ** 0.5) * (nb ** 0.5))
