"""大记忆池 + 节点标记（§2.3）—— 低成本节点专属记忆视图。

核心思想（用户方案）：**不复制记忆**。整池只有一份正文与 embedding，
每条记录打一个 `owner_node` 标记；"节点专属记忆" = 按标记过滤出来的视图。

收益：
- 零拷贝（正文/向量全项目一份）
- 零同步（更新/作废只改一处，不存在两份正文不一致）
- 零额外 embedding 调用（借来的记忆复用其入库时的向量）
- 倒排索引**可从池重建**，不落盘、不新增侧表（保持 D-10）
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

from .models import MAIN_NODE

#: 未标标记的历史记录默认归属（迁移策略 §7.4）
DEFAULT_OWNER = MAIN_NODE


class NodeIndex:
    """`owner_node → [memory_id]` 倒排索引（插入 O(1)，可重建）。"""

    def __init__(self) -> None:
        self._by_node: dict[str, list[str]] = defaultdict(list)
        self._owner_of: dict[str, str] = {}

    # ---- 写 ----

    def add(self, memory_id: str, owner_node: Optional[str]) -> None:
        owner = owner_node or DEFAULT_OWNER
        prev = self._owner_of.get(memory_id)
        if prev == owner:
            return
        if prev is not None:
            self._remove_from(prev, memory_id)
        self._owner_of[memory_id] = owner
        self._by_node[owner].append(memory_id)

    def remove(self, memory_id: str) -> None:
        owner = self._owner_of.pop(memory_id, None)
        if owner is not None:
            self._remove_from(owner, memory_id)

    def _remove_from(self, owner: str, memory_id: str) -> None:
        lst = self._by_node.get(owner)
        if lst and memory_id in lst:
            lst.remove(memory_id)

    # ---- 读 ----

    def owner_of(self, memory_id: str) -> Optional[str]:
        return self._owner_of.get(memory_id)

    def ids_of(self, node_id: str) -> list[str]:
        return list(self._by_node.get(node_id, ()))

    def ids_of_any(self, node_ids: Iterable[str]) -> list[str]:
        """多源合并（**跨源候选超集**），按 node_ids 顺序去重。"""
        seen: set[str] = set()
        out: list[str] = []
        for nid in node_ids:
            for mid in self._by_node.get(nid, ()):   # noqa: SIM118
                if mid not in seen:
                    seen.add(mid)
                    out.append(mid)
        return out

    def node_ids(self) -> list[str]:
        return [n for n, ids in self._by_node.items() if ids]

    def counts(self) -> dict[str, int]:
        return {n: len(ids) for n, ids in self._by_node.items() if ids}

    def size(self) -> int:
        return len(self._owner_of)

    # ---- 重建 ----

    def rebuild(self, records: Iterable) -> None:
        """从池重建（池已加载完 / 数据被外部改动时调用）。"""
        self._by_node = defaultdict(list)
        self._owner_of = {}
        for rec in records:
            self.add(getattr(rec, "memory_id", ""), getattr(rec, "owner_node", None))


class ScopedPool:
    """**只读视图代理**：把常驻池收窄到给定节点集合，接口与 `ResidentMemoryPool` 兼容。

    设计目的：`BaseRetriever.recall(query, pool, top_k)` 只依赖 `pool.all()`，
    因此传入本代理即可实现"限定检索范围"而**完全不用改 retriever.py**（保护既有算法模块）。

    ⚠️ `vector_index` 暴露为 None 是**刻意**的：faiss 索引是全池索引，
    在"限定范围"下会返回范围外的 id；置 None 让 `VectorRetriever` 走
    `linear_cosine_rank` 精确路径（范围小，精确检索反而更快更准）。
    """

    def __init__(self, pool, node_ids: Iterable[str], fallback_all: bool = False):
        self._pool = pool
        self.node_ids = list(dict.fromkeys(node_ids))
        self._fallback_all = fallback_all
        self._cands: Optional[list] = None

    # 与 ResidentMemoryPool 同名接口
    @property
    def user_id(self) -> str:
        return self._pool.user_id

    @property
    def session_id(self) -> str:
        return self._pool.session_id

    @property
    def vector_index(self):          # noqa: D401 - 见类文档说明
        return None

    def all(self) -> list:
        if self._cands is None:
            idx = getattr(self._pool, "node_index", None)
            if idx is None:
                self._cands = self._pool.all()
            else:
                ids = idx.ids_of_any(self.node_ids)
                self._cands = [r for r in (self._pool.get(i) for i in ids) if r is not None]
                if self._fallback_all and not self._cands:
                    self._cands = self._pool.all()
        return list(self._cands)

    def size(self) -> int:
        return len(self.all())

    def get(self, memory_id: str):
        return self._pool.get(memory_id)


def exact_in_scope_recall(query_emb, candidates: list, top_k: int) -> list:
    """**限定范围内**的精确召回（不做 ANN 近似）。

    为什么需要它（实测缺陷 D2）：检索器绑定的 faiss 索引是**全池**索引，
    在限定范围时它会先在全池取 top-k 再与范围内 id 求交 →
    **范围内的条目会被静默丢弃**（实测范围内 4 条只回来 2 条）。
    范围内条目数小（数十条），线性精确余弦更快、更准、且完全确定性（保 NFR-6）。
    """
    if not candidates:
        return []
    if query_emb is None:
        return list(candidates)[:top_k]
    try:
        from ..vector_index import linear_cosine_rank
        return linear_cosine_rank(query_emb, candidates, top_k)
    except Exception:  # noqa: BLE001 - 兜底：保持原序
        return list(candidates)[:top_k]
