"""成本 / 效果 / 速度三角均衡控制器（优化方案 §5）。

三个量（都可测、都可埋点）：
- **成本 C**：`llm_calls`（LLM 调用次数）/ `embed_calls_miss`（**只计未命中缓存**的 embedding 调用）
  / `injected_tokens`（注入上下文长度的粗估）。
- **效果 E**：由 `analysis/graph_metrics.py` 从 run 结果算（本模块不加成效果）。
- **速度 T**：`latency_ms.total` 的 P50 / P90。

三层机制：
1. **成本侧**：LLM 先验**缓存**（命中 = 0 次调用）+ 每次调度调用预算 + 多重门控。
2. **速度侧**：**三档 perf_tier** + 滞回升降档（实验默认 `adaptive=False` 锁定档位保可复现）。
3. **效果侧**：门控"把钱花在刀刃上"（Stage A 的 relevance 与四维打分都是纯 CPU，不计成本，默认全开）。

⚠️ **诚实边界**：本模块的升降档阈值、门槛常量是**工程启发式**，不是有客观依据的最优值。
论文中只应报告"三档的 (E, C, T) 取点与 Pareto 曲线"，**不主张** `J = E − γC·C − γT·T` 是可优化目标。
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict, deque
from enum import Enum
from typing import Any, Optional


class PerfTier(str, Enum):
    FULL = "full"
    BALANCED = "balanced"
    FAST = "fast"


#: 各档参数。B 系列用**单变量扫描**验证每一列各自的贡献，不靠"三档连跑"归因。
TIER_SPEC: dict[str, dict[str, Any]] = {
    PerfTier.FULL.value:     {"llm_prior": "call",  "candidate": 50, "lambda": 0.7, "label": "效果优先"},
    PerfTier.BALANCED.value: {"llm_prior": "cache", "candidate": 30, "lambda": 0.7, "label": "默认均衡"},
    PerfTier.FAST.value:     {"llm_prior": "off",   "candidate": 20, "lambda": 1.0, "label": "速度优先"},
}

TIER_ORDER = [PerfTier.FULL.value, PerfTier.BALANCED.value, PerfTier.FAST.value]


class PriorCache:
    """LLM 先验的 LRU 缓存（key = 归一化 query + condition）。"""

    def __init__(self, size: int = 512):
        self.size = max(0, int(size))
        self._d: "OrderedDict[str, dict]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def norm_key(query: str, condition: Optional[str] = None) -> str:
        q = " ".join((query or "").strip().lower().split())
        raw = f"{q}||{condition or ''}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[dict]:
        if self.size <= 0 or key not in self._d:
            self.misses += 1
            return None
        self._d.move_to_end(key)
        self.hits += 1
        return dict(self._d[key])

    def put(self, key: str, value: dict) -> None:
        if self.size <= 0:
            return
        self._d[key] = dict(value)
        self._d.move_to_end(key)
        while len(self._d) > self.size:
            self._d.popitem(last=False)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits, "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }


class BudgetController:
    """三档均衡控制器（单次调度内的成本计数 + 跨调度的分位延迟统计）。"""

    def __init__(self, cfg=None, tier: Optional[str] = None,
                 adaptive: Optional[bool] = None,
                 latency_budget_ms: Optional[float] = None,
                 window: Optional[int] = None,
                 cache_size: Optional[int] = None,
                 llm_calls_per_schedule: Optional[int] = None,
                 min_candidates_for_prior: int = 5):
        c = cfg
        self.tier: str = tier or _cfg(c, "default_tier", PerfTier.BALANCED.value)
        if self.tier not in TIER_SPEC:
            self.tier = PerfTier.BALANCED.value
        self.adaptive: bool = bool(_cfg(c, "adaptive", False) if adaptive is None else adaptive)
        self.latency_budget_ms: float = float(
            latency_budget_ms if latency_budget_ms is not None
            else _cfg(c, "latency_budget_ms", 900.0))
        self.window: int = int(window if window is not None else _cfg(c, "window", 20))
        self.cache = PriorCache(int(cache_size if cache_size is not None
                                    else _cfg(c, "llm_prior_cache", 512)))
        self.llm_calls_per_schedule: int = int(
            llm_calls_per_schedule if llm_calls_per_schedule is not None
            else _cfg(c, "llm_calls_per_schedule", 1))
        self.min_candidates_for_prior = int(min_candidates_for_prior)

        self._lat: deque[float] = deque(maxlen=self.window)
        self._low_streaks = 0
        self.tier_changes: list[dict] = []
        self.gate_stats: dict[str, int] = {}
        # 单变量扫描旋钮（B 系列用；None = 用档位默认）
        self.force_candidate: Optional[int] = _cfg(c, "force_candidate", None)
        self.force_lambda: Optional[float] = _cfg(c, "force_lambda", None)
        self._reset_counters()

    # ------------------------------------------------------------ 档位

    @property
    def spec(self) -> dict[str, Any]:
        return TIER_SPEC[self.tier]

    @property
    def candidate_override(self) -> int:
        if self.force_candidate is not None:
            return int(self.force_candidate)
        return int(self.spec["candidate"])

    @property
    def lambda_override(self) -> float:
        if self.force_lambda is not None:
            return float(self.force_lambda)
        return float(self.spec["lambda"])

    def set_tier(self, tier: str, reason: str = "manual") -> None:
        if tier not in TIER_SPEC or tier == self.tier:
            return
        prev = self.tier
        self.tier = tier
        self.tier_changes.append({"from": prev, "to": tier, "reason": reason})

    # ------------------------------------------------------------ 成本门控

    def _gate(self, name: str) -> None:
        self.gate_stats[name] = self.gate_stats.get(name, 0) + 1

    def should_call_llm_prior(self, n_candidates: int, cache_key: str,
                              source_degraded: bool = False) -> tuple[bool, Optional[dict], str]:
        """是否值得花一次 LLM 调用。返回 `(need_call, cached_value, reason)`。

        门控顺序（先便宜后昂贵）：
          1. `tier_off`   —— 档位为 fast，直接跳过
          2. `cache_hit`  —— 命中缓存，0 成本
          3. `budget`     —— 本次调度额度已用完
          4. `few_cand`   —— 候选太少（≤ min_candidates_for_prior），四维权重影响微乎其微
          5. `low_src_var`—— Stage A 已退化为纯静态（源不可区分），源维度不提供信息
        """
        if self.spec["llm_prior"] == "off":
            self._gate("tier_off")
            return False, None, "tier_off"
        cached = self.cache.get(cache_key)
        if cached is not None:
            self._gate("cache_hit")
            return False, cached, "cache_hit"
        if self.llm_calls >= self.llm_calls_per_schedule:
            self._gate("budget")
            return False, None, "budget"
        if n_candidates <= self.min_candidates_for_prior:
            self._gate("few_cand")
            return False, None, "few_cand"
        if source_degraded and self.spec["llm_prior"] == "cache":
            self._gate("low_src_var")
            return False, None, "low_src_var"
        return True, None, "call"

    def note_prior(self, cache_key: str, prior: dict, called: bool) -> None:
        if called:
            self.llm_calls += 1
            self.cache.put(cache_key, prior)

    # ------------------------------------------------------------ 成本计数

    def _reset_counters(self) -> None:
        self.llm_calls = 0
        self.embed_calls_miss = 0
        self.injected_tokens = 0

    def reset_counters(self) -> None:
        self._reset_counters()

    def note_embed_miss(self, n: int = 1) -> None:
        """embedding 缓存未命中的调用（命中不计费，不调用本方法）。"""
        self.embed_calls_miss += int(n)

    def note_injected_tokens(self, n: int) -> None:
        self.injected_tokens += int(n)

    def counters(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "embed_calls_miss": self.embed_calls_miss,
            "injected_tokens": self.injected_tokens,
        }

    # ------------------------------------------------------------ 速度统计

    def note_latency(self, total_ms: float) -> None:
        self._lat.append(float(total_ms))

    def percentiles(self) -> dict:
        if not self._lat:
            return {"p50": 0.0, "p90": 0.0, "n": 0}
        xs = sorted(self._lat)
        return {
            "p50": round(_pct(xs, 0.50), 2),
            "p90": round(_pct(xs, 0.90), 2),
            "max": round(xs[-1], 2),
            "n": len(xs),
        }

    def maybe_adjust(self) -> Optional[dict]:
        """滞回升降档。**`adaptive=False`（实验默认）时恒为 no-op。**

        降档：P90 > budget（本窗口）→ 降一档。
        升档：P50 < 0.45·budget 连续 3 个窗口 → 升一档（滞回，避免抖动）。
        """
        if not self.adaptive or len(self._lat) < max(5, self.window // 2):
            return None
        p = self.percentiles()
        i = TIER_ORDER.index(self.tier)
        if p["p90"] > self.latency_budget_ms and i < len(TIER_ORDER) - 1:
            self.set_tier(TIER_ORDER[i + 1], f"P90 {p['p90']}ms > {self.latency_budget_ms}ms")
            self._low_streaks = 0
            return self.tier_changes[-1]
        if p["p50"] < 0.45 * self.latency_budget_ms and i > 0:
            self._low_streaks += 1
            if self._low_streaks >= 3:
                self.set_tier(TIER_ORDER[i - 1], f"P50 {p['p50']}ms 持续偏低")
                self._low_streaks = 0
                return self.tier_changes[-1]
        else:
            self._low_streaks = 0
        return None

    # ------------------------------------------------------------ 埋点

    def to_dict(self) -> dict:
        return {
            "perf_tier": self.tier,
            "adaptive": self.adaptive,
            "latency_budget_ms": self.latency_budget_ms,
            "cache": self.cache.stats(),
            "gates": dict(self.gate_stats),
            "tier_changes": len(self.tier_changes),
            "budget_percentiles": self.percentiles(),
        }


# ------------------------------------------------------------------ helpers


def _pct(sorted_xs: list[float], q: float) -> float:
    if not sorted_xs:
        return 0.0
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    idx = q * (len(sorted_xs) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_xs) - 1)
    frac = idx - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac


def _cfg(c, name: str, default):
    if c is None:
        return default
    if isinstance(c, dict):
        return c.get(name, default)
    return getattr(c, name, default)
