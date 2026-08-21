"""四维注意力打分器（§4.2，FR-3）。

纯函数、可单测：时间（指数衰减）/ 语义（余弦+向量/BM25 原始分）/ 频率（访问次数）/ 任务（标签匹配）。
enabled_dims 掩码：关掉的维权重强制 0，其余归一化保持 Σ=1（REQ-610 四维分别验证）。
V2.1：语义头内部系数用 w_cos/w_vec/w_bm25，与全局 α/β（先验/可学习融合系数）区分。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

ALL_DIMS = ["time", "semantic", "frequency", "task"]
DEFAULT_WEIGHTS: dict[str, float] = {
    "time": 0.25,
    "semantic": 0.35,
    "frequency": 0.15,
    "task": 0.25,
}
# 语义头内部融合系数（与全局 α/β 无关）
W_COS, W_VEC, W_BM25 = 0.5, 0.3, 0.2
# 时间衰减参数（小时）
LAMBDA_TIME = 0.01


@dataclass
class MemoryCandidate:
    """常驻池中的一条记忆（§4.2）。由 BaseRetriever 在常驻池上构建。"""

    memory_id: str            # 唯一主键（uuid，引用/更新用）
    text: str                 # 记忆文本
    path: str                 # 源文件路径（书签辅助定位，FR-8）
    start_line: int = 0       # 行号起点
    end_line: int = 0         # 行号终点
    timestamp: float = 0.0    # 记忆时间戳（unix 秒）
    access_count: int = 0     # 访问次数（L3 维护）
    task_tag: Optional[str] = None  # 任务标签
    user_id: str = ""         # 归属用户（隔离过滤维度）
    session_id: str = ""      # 归属会话（隔离过滤维度）
    bm25_score: float = 0.0   # 常驻池 BM25 原始分（中性基座，D-12）
    vector_score: float = 0.0 # 常驻池向量原始分（中性基座，D-12）
    embedding: Optional[np.ndarray] = None  # 1024 维向量；缺失时 semantic 仅用 bm25（D-11）


class MultiHeadAttentionMemoryScorer:
    """四维注意力打分器（默认完整实现，插件名 attention.full）。"""

    def __init__(self, weights: Optional[dict[str, float]] = None):
        """默认权重 DEFAULT_WEIGHTS（实际运行时被 HybridWeightCalculator 动态覆盖）。"""
        self.weights = dict(weights or DEFAULT_WEIGHTS)

    def score(
        self,
        query_emb: Optional[np.ndarray],
        memory: MemoryCandidate,
        current_time: float,
        task_tag: Optional[str],
        enabled_dims: Optional[set[str]] = None,
    ) -> dict[str, float]:
        """四维打分 + 加权融合；enabled_dims=None 表示四维全开。

        返回 {final, time, semantic, frequency, task}。
        final = Σ eff_w_i·s_i / Σ eff_w_i（关闭维权重为 0 后归一化）。
        """
        dims = set(enabled_dims) if enabled_dims else set(ALL_DIMS)
        s = {
            "time": self._time_score(memory, current_time),
            "semantic": self._semantic_score(query_emb, memory),
            "frequency": self._frequency_score(memory),
            "task": self._task_score(memory, task_tag),
        }
        eff_w = {d: (self.weights[d] if d in dims else 0.0) for d in ALL_DIMS}
        norm = sum(eff_w.values()) or 1.0
        final = sum(eff_w[d] * s[d] for d in ALL_DIMS) / norm
        return {"final": final, **s}

    def _time_score(self, memory: MemoryCandidate, current_time: float) -> float:
        """指数衰减：t=(now-ts)/3600；s=exp(-λ·t)。"""
        t = max(0.0, (current_time - memory.timestamp) / 3600.0)
        return float(math.exp(-LAMBDA_TIME * t))

    def _semantic_score(self, query_emb, memory: MemoryCandidate) -> float:
        """独立融合：w_cos·cosine + w_vec·vector + w_bm25·bm25（与 ReMe RRF 解耦，D-12）。

        embedding 缺失（D-11 兜底）时 cosine 项记 0；query 或 memory 无向量时同样退化。
        """
        cos = 0.0
        if query_emb is not None and memory.embedding is not None:
            q = query_emb.ravel().astype(np.float64)
            m = memory.embedding.ravel().astype(np.float64)
            qn, mn = np.linalg.norm(q), np.linalg.norm(m)
            if qn > 0 and mn > 0:
                cos = float(np.dot(q, m) / (qn * mn))
        return W_COS * cos + W_VEC * memory.vector_score + W_BM25 * memory.bm25_score

    def _frequency_score(self, memory: MemoryCandidate) -> float:
        """频率编码：s = 1 - exp(-0.1·access_count)。"""
        return 1.0 - math.exp(-0.1 * memory.access_count)

    def _task_score(self, memory: MemoryCandidate, task_tag: Optional[str]) -> float:
        """任务标签匹配：1.0 匹配 / 0.3 不匹配 / 双方空则 0.5 中性。"""
        if task_tag is None or memory.task_tag is None:
            return 0.5
        return 1.0 if memory.task_tag == task_tag else 0.3
