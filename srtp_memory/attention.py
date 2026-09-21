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
    # ---- 图架构新增（优化方案 §2.1；带默认值，保证旧数据 / 旧调用完全兼容）----
    owner_node: str = "main"  # ★ 生产者节点：该记忆归哪个会话节点的"专属记忆"（大池 + 标记）
    producer_run: str = ""    # 产出它的调度轮次 id（溯源用，可选）


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


# ======================================================================
# S0 修复：批级去偏 + 无效维门控 + 语义保底（新增，不改上面既有打分器）
# ======================================================================

class NormalizedAttentionScorer(MultiHeadAttentionMemoryScorer):
    """批级归一化的四维注意力打分器（插件名 `attention.normalized`）。

    **动机（真实数据上的实测诊断）**：逐条四维打分后直接加权融合，会出现三类系统性退化：

    1. **无效维浪费权重**：`frequency = 1-exp(-0.1·access_count)`，首轮调度时
       `access_count ≡ 0` → 该维恒为 0，权重被完全浪费（默认 0.15）。
    2. **常数维退化为偏置**：`task` 在同一任务标签下恒为 1.0（或双方空时恒 0.5）
       → 该项对**排序**零贡献，却占用 0.25 权重。
    3. **量纲/偏置不可比**：`time = exp(-0.01·t_h)` 反映的是记忆的**绝对年龄**
       （跨模块差异可达 0.43~1.0），与"该不该被召回"无直接关系；而 `semantic` 受余弦
       上下界约束，动态范围小。等权混合会让**非相关性信号稀释语义排序**。

    实测后果：G3/G4（四维融合）的候选排序反而不如 G0/G1（不做融合、直接用检索器语义序），
    `G4 − G1 = −1.92%`。本类即针对该缺陷的修复。

    **修复四步**（全部在**批内**完成，不改变单条 `score()` 的语义）：
      A. 逐维求批内标准差，`std < std_eps` 的维判为"无区分度" → 权重置 0（门控）；
      B. 存活维按原权重比例**再分配**回收的权重；
      C. 语义维设**保底权重** `semantic_floor`（候选筛选任务中语义是主导信号）；
      D. 存活维做批内 **min-max 归一化**，消除量纲差异后再加权融合。

    ⚠️ 本类是**新增**实现，`attention.full` 原样保留作为对照 —— 这样"修复是否有效"
    本身就可消融（`G4` 用 full vs `G4_norm` 用 normalized）。
    """

    def __init__(self, weights: Optional[dict[str, float]] = None,
                 semantic_floor: float = 0.5, std_eps: float = 1e-9):
        super().__init__(weights)
        self.semantic_floor = float(max(0.0, min(1.0, semantic_floor)))
        self.std_eps = float(std_eps)

    # ------------------------------------------------------------------

    def score_batch(self, query_emb, memories: list[MemoryCandidate],
                    current_time: float, task_tag: Optional[str],
                    weights: Optional[dict[str, float]] = None,
                    enabled_dims: Optional[set[str]] = None) -> list[dict[str, float]]:
        """批级打分：返回与 `memories` 等长的 [{final, time, semantic, frequency, task, ...}]。

        额外附带 `_gated`（哪些维被判无效）与 `_eff_w`（实际生效权重）便于埋点与诊断。
        """
        w = dict(weights or self.weights)
        dims = set(enabled_dims) if enabled_dims else set(ALL_DIMS)

        # A0. 逐条取四维原始分（复用单条 score，保持公式唯一真源）
        raw: list[dict[str, float]] = []
        for m in memories:
            raw.append(self.score(query_emb, m, current_time, task_tag,
                                  enabled_dims=enabled_dims))
        if not raw:
            return []

        # A. 逐维门控
        alive: dict[str, bool] = {}
        spread: dict[str, float] = {}
        for d in ALL_DIMS:
            if d not in dims:
                alive[d], spread[d] = False, 0.0
                continue
            vals = [r[d] for r in raw]
            sd = _std(vals)
            spread[d] = sd
            alive[d] = sd > self.std_eps

        # B. 权重再分配（+ C. 语义保底）
        eff = {d: (float(w.get(d, 0.0)) if alive[d] else 0.0) for d in ALL_DIMS}
        tot = sum(eff.values())
        if tot <= 0.0:
            # 全部维无区分度（例如候选全同、或全部 dims 被掩码关掉）
            # → 退化为存活维等权；若一个存活维都没有，则均匀兜底（避免除零）
            survivors = [d for d in ALL_DIMS if alive[d]] or list(ALL_DIMS)
            eff = {d: (1.0 / len(survivors) if d in survivors else 0.0) for d in ALL_DIMS}
            tot = sum(eff.values())
        eff = {d: v / tot for d, v in eff.items()}

        if eff.get("semantic", 0.0) > 0 and self.semantic_floor > 0:
            cur = eff["semantic"]
            if cur < self.semantic_floor:
                need = self.semantic_floor - cur
                others = [d for d in ALL_DIMS if d != "semantic" and eff[d] > 0]
                other_sum = sum(eff[d] for d in others)
                if other_sum > 0:
                    for d in others:
                        eff[d] = max(0.0, eff[d] - need * eff[d] / other_sum)
                    eff["semantic"] = self.semantic_floor

        # D. 存活维批内 min-max 归一化 + 融合
        ranges = {d: (min(x[d] for x in raw), max(x[d] for x in raw)) for d in ALL_DIMS}
        out: list[dict[str, float]] = []
        for r in raw:
            s: dict[str, float] = {}
            for d in ALL_DIMS:
                if not alive[d]:
                    s[d] = r[d]                      # 保留原值（埋点/诊断用）
                    continue
                lo, hi = ranges[d]
                s[d] = (r[d] - lo) / (hi - lo) if hi > lo else 0.0
            s["final"] = sum(eff[d] * s[d] for d in ALL_DIMS)
            s["_gated"] = {d: (not alive[d]) for d in ALL_DIMS}
            s["_spread"] = {d: round(spread[d], 6) for d in ALL_DIMS}
            s["_eff_w"] = {d: round(eff[d], 4) for d in ALL_DIMS}
            out.append(s)
        return out

    def names(self) -> list[str]:
        return list(ALL_DIMS)


def _std(xs: list[float]) -> float:
    """批内标准差（n<2 视为 0，即"无法判断区分度"→ 门控掉）。"""
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return float((sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5)
