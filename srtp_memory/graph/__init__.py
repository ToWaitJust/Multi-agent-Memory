"""Topic 图记忆调度子包（优化方案 §7.1）。

隔离原则：本子包是**纯新增**，不修改 `attention.py` / `weights.py` / `selector.py` /
`retriever.py` 的既有逻辑；既有模块只做"加字段 + 插桩"。

模块职责：
- `models`          数据模型（TopicNode / TopicEdge / SourceRef / 边默认强度）
- `graph_store`     强制 DAG 的图存储（环检测、邻域展开、拓扑排序、版本折叠、持久化）
- `view`            大记忆池 + 节点标记倒排索引（节点专属记忆的零拷贝视图）
- `source_selector` Stage A 选源（多对多筛选的具体形态）
- `builder`         两阶段建边（粗/精识别，冷启动）
- `budget`          成本 / 效果 / 速度三角均衡控制器
"""
from __future__ import annotations

from .budget import BudgetController, PerfTier, PriorCache, TIER_SPEC
from .builder import EdgeBuilder, hash_embed
from .graph_store import CycleError, GraphStore
from .models import (
    EDGE_DEFAULT_STRENGTH,
    EDGE_STATUS_ACTIVE,
    EDGE_STATUS_CYCLE_REJECTED,
    EDGE_TYPES,
    MAIN_NODE,
    STAGE_COARSE,
    STAGE_REFINED,
    SourceRef,
    SourceSelection,
    TopicEdge,
    TopicNode,
    cosine,
    infer_edge_type,
)
from .source_selector import QUOTA_MODES, SourceSelector
from .view import DEFAULT_OWNER, NodeIndex, ScopedPool, exact_in_scope_recall

__all__ = [
    "BudgetController", "PerfTier", "PriorCache", "TIER_SPEC",
    "EdgeBuilder", "hash_embed",
    "CycleError", "GraphStore",
    "EDGE_DEFAULT_STRENGTH", "EDGE_STATUS_ACTIVE", "EDGE_STATUS_CYCLE_REJECTED",
    "EDGE_TYPES", "MAIN_NODE", "STAGE_COARSE", "STAGE_REFINED",
    "SourceRef", "SourceSelection", "TopicEdge", "TopicNode",
    "cosine", "infer_edge_type",
    "QUOTA_MODES", "SourceSelector",
    "DEFAULT_OWNER", "NodeIndex", "ScopedPool", "exact_in_scope_recall",
]
