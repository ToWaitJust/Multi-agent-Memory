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

def gate_and_redistribute(raw: list[dict[str, float]], dims: set[str],
                          w: dict[str, float], semantic_floor: float,
                          std_eps: float):
    """门控三步曲（A 判无区分度 → B 权重再分配 → C 语义保底），返回 (alive, spread, eff)。

    抽成公共函数是 D-050：`AdaptiveGatedScorer` 需要**同样的门控**但**不做** min-max
    归一化 —— 逻辑必须唯一真源，否则两个实现会悄悄分叉，消融就失去意义。
    """
    # A. 逐维门控：批内标准差过小 → 该维对排序无区分度 → 权重置 0
    alive: dict[str, bool] = {}
    spread: dict[str, float] = {}
    for d in ALL_DIMS:
        if d not in dims:
            alive[d], spread[d] = False, 0.0
            continue
        sd = _std([r[d] for r in raw])
        spread[d] = sd
        alive[d] = sd > std_eps

    # B. 权重再分配：回收的权重按存活维的原权重比例重新分配（Σ=1）
    eff = {d: (float(w.get(d, 0.0)) if alive[d] else 0.0) for d in ALL_DIMS}
    tot = sum(eff.values())
    if tot <= 0.0:
        # 全部维无区分度（候选全同 / 全被掩码）→ 存活维等权；一个都没有则均匀兜底
        survivors = [d for d in ALL_DIMS if alive[d]] or list(ALL_DIMS)
        eff = {d: (1.0 / len(survivors) if d in survivors else 0.0) for d in ALL_DIMS}
        tot = sum(eff.values())
    eff = {d: v / tot for d, v in eff.items()}

    # C. 语义保底：候选筛选里语义是主导信号，权重低于 floor 时从其他维扣回来
    if eff.get("semantic", 0.0) > 0 and semantic_floor > 0:
        cur = eff["semantic"]
        if cur < semantic_floor:
            need = semantic_floor - cur
            others = [d for d in ALL_DIMS if d != "semantic" and eff[d] > 0]
            other_sum = sum(eff[d] for d in others)
            if other_sum > 0:
                for d in others:
                    eff[d] = max(0.0, eff[d] - need * eff[d] / other_sum)
                eff["semantic"] = semantic_floor
    return alive, spread, eff


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

        # A. 逐维门控 → B. 权重再分配 → C. 语义保底（抽成公共函数，供门控类复用）
        alive, spread, eff = gate_and_redistribute(
            raw, dims, w, self.semantic_floor, self.std_eps)

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


# ======================================================================
# D-049 修复：自适应时间尺度（真实长跨度语料上的 time 维失效问题）
# ======================================================================

class TimeAdaptiveScorer(MultiHeadAttentionMemoryScorer):
    """自适应时间尺度的四维打分器（插件名 `attention.adaptive_time`）。

    **问题（LoCoMo 真实数据集实测发现）**
    固定 `LAMBDA_TIME = 0.01 /小时` 的半衰期约 **69 小时（≈2.9 天）**。
    而真实长期会话语料的时间跨度是**月级**的（LoCoMo conv0 语料跨度 432 小时 ≈ 18 天），
    于是候选记忆的 time 分被整体压进一个**极窄区间**：

        实测 conv0 候选：time ∈ [0.0215, 0.0561]，std = 0.0168，只有 3 个取值

    这导致两个**方向相反但同源**的失败：
      1. **不归一化时**：time 对排序几乎零影响 —— 白占约 0.25 的权重
         （实测：禁用 time 维后结果与不禁用**逐位相同**）。
      2. **做批内 min-max 归一化时**：这个"噪声级"信号被拉伸到满量程 [0,1]，
         **放大约 28 倍**，反客为主主导排序（实测 `attention.normalized` 的
         MRR 0.1303 < `attention.full` 的 0.2028）—— 因为归一化后 time 偏向
         "时间上更新"的记忆，而**长期会话里答案往往在早期的 session**。

    **修复**：让衰减尺度跟随**当前候选集的实际时间跨度**，而不是写死一个绝对小时数：

        λ_eff = TIME_DECAY_K / span_hours      （span_hours = 候选时间戳跨度）

    这样无论语料跨小时、跨天还是跨月，time 分在候选集内始终跨越
    `exp(-K) → 1.0` 的完整动态范围（K=2 时约 [0.135, 1.0]），
    **彻底消除"时间尺度与语料跨度不匹配"这一结构性缺陷**。

    ⚠️ 本类是**新增**实现，`attention.full` / `attention.normalized` 均原样保留作对照，
    因此"自适应时间尺度是否有效"本身就可消融（单变量）。
    """

    #: 候选集时间跨度内衰减到的指数（exp(-K)），K=2 → 最旧记忆约为最新的 0.135
    TIME_DECAY_K = 2.0

    def __init__(self, weights: Optional[dict[str, float]] = None,
                 decay_k: float = TIME_DECAY_K):
        super().__init__(weights)
        self.decay_k = float(decay_k)
        # 未进入 score_batch（单条调用）时退回原固定尺度，保持向后兼容
        self._lambda_eff: float = LAMBDA_TIME
        self.last_span_hours: Optional[float] = None

    def _time_score(self, memory: MemoryCandidate, current_time: float) -> float:
        """指数衰减（尺度由 `score_batch` 自适应设定）。"""
        t = max(0.0, (current_time - memory.timestamp) / 3600.0)
        return float(math.exp(-self._lambda_eff * t))

    def _set_lambda(self, memories: list[MemoryCandidate]) -> None:
        """按候选集时间戳跨度自适应设定 λ_eff = decay_k / span_hours。

        抽成独立方法：子类 `AdaptiveGatedScorer` 要在取原始分**之前**先定尺度。
        """
        ts = [float(getattr(m, "timestamp", 0.0) or 0.0) for m in memories]
        ts = [t for t in ts if t > 0.0]
        if len(ts) >= 2:
            span_h = (max(ts) - min(ts)) / 3600.0
            self.last_span_hours = round(span_h, 2)
            self._lambda_eff = self.decay_k / max(span_h, 1.0)
        else:
            self.last_span_hours = None
            self._lambda_eff = LAMBDA_TIME

    def score_batch(self, query_emb, memories: list[MemoryCandidate],
                    current_time: float, task_tag: Optional[str],
                    weights: Optional[dict[str, float]] = None,
                    enabled_dims: Optional[set[str]] = None) -> list[dict[str, float]]:
        """批级打分：先按候选时间戳跨度自适应设定 λ，再逐条打分并融合。"""
        w = dict(weights or self.weights)

        # ① 自适应时间尺度
        self._set_lambda(memories)

        # ② 逐条打分（复用单条 score，保持公式唯一真源）+ 用传入权重重算 final
        dims = set(enabled_dims) if enabled_dims else set(ALL_DIMS)
        eff = {d: (float(w.get(d, 0.0)) if d in dims else 0.0) for d in ALL_DIMS}
        norm = sum(eff.values()) or 1.0
        out: list[dict[str, float]] = []
        for m in memories:
            sc = self.score(query_emb, m, current_time, task_tag,
                            enabled_dims=enabled_dims)
            sc["final"] = sum(eff[d] * sc[d] for d in ALL_DIMS) / norm
            sc["_lambda_eff"] = round(self._lambda_eff, 8)
            out.append(sc)
        return out

    def names(self) -> list[str]:
        return list(ALL_DIMS)


# ======================================================================
# D-050 修复二：自适应时间尺度 + 常量维门控（不做 min-max）
# ======================================================================

class AdaptiveGatedScorer(TimeAdaptiveScorer):
    """自适应时间尺度 + 常量维门控（插件名 `attention.adaptive_gated`）。

    **为什么还需要这一类**（LoCoMo 诊断的第二步）：

    `attention.adaptive_time` 只修了"尺度"，但没修"权重浪费"。实测 LoCoMo 上：

      · `task` 维**恒为 1.0**（语料没有"口径"概念，所有候选与 query 同标签）
        → 对排序零贡献却占 0.25 权重；
      · `frequency` 维首轮恒为 0（access_count=0）→ 白占 0.15。

    也就是说**一半的权重预算被无效维吃掉**，语义维实际只拿到 0.35/1.0。

    **为什么不用 `attention.normalized`**：它把 min-max 归一化也做了，
    而在长跨度语料上 min-max 会把 time 的**噪声级**信号拉伸到满量程（放大 ~28 倍），
    反客为主（实测 MRR 0.1303 < 0.2028）。本类**只做门控 + 权重再分配 + 语义保底**，
    **不做** min-max —— 各维保留原始量纲，谁也别被人为放大。

    三类打分器的差异用一行话概括：
      full          ：固定 λ，无门控，无归一化（原始基线）
      normalized    ：固定 λ，有门控，**有** min-max（放大噪声）
      adaptive_gated：自适应 λ，有门控，**无** min-max（本类）
    """

    def __init__(self, weights: Optional[dict[str, float]] = None,
                 decay_k: float = TimeAdaptiveScorer.TIME_DECAY_K,
                 semantic_floor: float = 0.5, std_eps: float = 1e-9):
        super().__init__(weights, decay_k=decay_k)
        self.semantic_floor = float(max(0.0, min(1.0, semantic_floor)))
        self.std_eps = float(std_eps)

    def score_batch(self, query_emb, memories: list[MemoryCandidate],
                    current_time: float, task_tag: Optional[str],
                    weights: Optional[dict[str, float]] = None,
                    enabled_dims: Optional[set[str]] = None) -> list[dict[str, float]]:
        w = dict(weights or self.weights)
        dims = set(enabled_dims) if enabled_dims else set(ALL_DIMS)

        # ① 自适应时间尺度（复用父类逻辑：λ_eff = K / 候选时间跨度）
        self._set_lambda(memories)

        # ② 逐条取原始四维分（time 已用 λ_eff，其余与 full 完全一致）
        raw: list[dict[str, float]] = []
        for m in memories:
            raw.append(self.score(query_emb, m, current_time, task_tag,
                                  enabled_dims=enabled_dims))
        if not raw:
            return []

        # ③ 门控 + 权重再分配 + 语义保底（与 normalized 共用同一份实现）
        alive, spread, eff = gate_and_redistribute(
            raw, dims, w, self.semantic_floor, self.std_eps)

        # ④ 融合（★ 不做 min-max：保留各维原始量纲）
        out: list[dict[str, float]] = []
        for r in raw:
            s = {d: r[d] for d in ALL_DIMS}
            s["final"] = sum(eff[d] * r[d] for d in ALL_DIMS)
            s["_gated"] = {d: (not alive[d]) for d in ALL_DIMS}
            s["_spread"] = {d: round(spread[d], 6) for d in ALL_DIMS}
            s["_eff_w"] = {d: round(eff[d], 4) for d in ALL_DIMS}
            s["_lambda_eff"] = round(self._lambda_eff, 8)
            out.append(s)
        return out

    def names(self) -> list[str]:
        return list(ALL_DIMS)
