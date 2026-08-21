"""向量索引（faiss 封装）—— 常驻池的 ANN 加速索引。

用途：VectorRetriever 从"线性遍历算余弦"升级为"faiss 近似最近邻查询"（O(log n)，达标 NFR-2 <100ms）。
- 精度可调：IndexFlatIP（精确内积）+ nprobe/ef 无关（flat 无近似）；如需近似可换 IVF/HNSW。
- 归因安全：索引只影响"召回候选"的速度/集合，不参与排序/选择（消融变量）；消融时候选集固定。
- 兜底：faiss 不可用或向量缺失时自动退化为线性余弦（原逻辑），保证可用性（D-11 精神）。
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

try:
    import faiss
    _FAISS_OK = True
except ImportError:  # pragma: no cover - 环境缺 faiss 时退化
    _FAISS_OK = False


class VectorIndex:
    """基于 faiss 的余弦 ANN 索引（内积 + L2 归一化 = 余弦）。

    add() / remove() / query() 三个原语；embedding 缺失的记录自动跳过。
    """

    def __init__(self, dim: int = 1024, metric: str = "ip"):
        self.dim = dim
        self.metric = metric
        self._ids: list[int] = []          # faiss 内部 id → 业务 memory_id 的映射序
        self._id_to_mem: dict[int, str] = {}
        self._mem_to_id: dict[str, int] = {}
        self._next_id = 0
        self._index = None
        self._ensure_index(dim)

    def _ensure_index(self, dim: int) -> None:
        """按维度创建索引；已存在且维度一致则复用。"""
        if not _FAISS_OK:
            return
        if self._index is not None and self._index.d == dim:
            return
        if self._index is not None:
            self._index.reset()
        self.dim = dim
        if self.metric == "ip":
            self._index = faiss.IndexFlatIP(dim)   # 精确内积（配 L2 归一化 = 余弦）
        else:
            self._index = faiss.IndexFlatL2(dim)

    @property
    def available(self) -> bool:
        return _FAISS_OK and self._index is not None

    def add(self, memory_id: str, embedding: np.ndarray) -> None:
        """插入一条记录。embedding 非 L2 归一化时自动归一化（保证余弦语义）。"""
        if not self.available or embedding is None:
            return
        if memory_id in self._mem_to_id:
            self.remove(memory_id)
        v = embedding.ravel().astype(np.float32)
        n = float(np.linalg.norm(v))
        if n <= 0:
            return
        self._ensure_index(v.shape[0])   # 自适应维度（首次 add 按实际维度建索引）
        v = v / n
        fid = self._next_id
        self._next_id += 1
        self._index.add(v.reshape(1, -1))
        self._ids.append(fid)
        self._id_to_mem[fid] = memory_id
        self._mem_to_id[memory_id] = fid

    def remove(self, memory_id: str) -> None:
        """按业务 memory_id 移除（faiss flat 索引不支持删除，采用全量重建策略；小规模可接受）。"""
        if memory_id not in self._mem_to_id:
            return
        fid = self._mem_to_id.pop(memory_id)
        del self._id_to_mem[fid]
        # 收集剩余向量
        remaining = []
        for i in range(self._index.ntotal):
            if self._ids[i] != fid:
                remaining.append((self._id_to_mem[self._ids[i]],
                                  self._index.reconstruct(i)))
        # 清空并全量重放（id 重排，映射同步重建）
        self._index.reset()
        self._ids.clear()
        self._id_to_mem.clear()
        self._mem_to_id.clear()
        self._next_id = 0
        for mid, vec in remaining:
            self.add(mid, vec)

    def query(self, query_emb: np.ndarray, top_k: int) -> tuple[list[str], list[float]]:
        """返回 (memory_id 列表, 余弦相似度列表)，按相似度降序。"""
        if not self.available or self._index.ntotal == 0:
            return [], []
        q = query_emb.ravel().astype(np.float32)
        nq = float(np.linalg.norm(q))
        if nq <= 0:
            return [], []
        q = q / nq
        k = min(top_k, self._index.ntotal)
        scores, idxs = self._index.search(q.reshape(1, -1), k)
        mem_ids = [self._id_to_mem[int(i)] for i in idxs[0] if int(i) != -1]
        sims = [float(s) for s in scores[0][: len(mem_ids)]]
        return mem_ids, sims

    def size(self) -> int:
        return self._index.ntotal if self._index is not None else 0

    def clear(self) -> None:
        if self._index is not None:
            self._index.reset()
        self._ids.clear()
        self._id_to_mem.clear()
        self._mem_to_id.clear()
        self._next_id = 0


def linear_cosine_rank(query_emb: np.ndarray, cands: list, top_k: int) -> list:
    """线性余弦排序兜底（faiss 不可用或退化路径），返回按 vector_score 降序的候选。"""
    q = query_emb.ravel().astype(np.float64)
    qn = math.sqrt(float(q @ q)) or 1.0
    for c in cands:
        if c.embedding is None:
            c.vector_score = 0.0
        else:
            m = c.embedding.ravel().astype(np.float64)
            mn = math.sqrt(float(m @ m)) or 1.0
            c.vector_score = float(q @ m) / (qn * mn)
    scored = sorted(cands, key=lambda c: c.vector_score, reverse=True)
    return scored[:top_k]
