"""调度层配置模型（§4.1）。

对应 config/srtp.yaml。所有可调参数集中于此，缺失字段用默认值。
V2.1：无 store_impl —— 双池（常驻池/共享池）为内核具体类，不做存储后端插件化。
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class GraphConfig(BaseModel):
    """Topic 图配置（优化方案 §7.3）。`enabled=False` 时全链路行为与改造前一致。"""

    enabled: bool = False             # ★ 默认关，保证既有 5 组消融不受影响
    source_mode: str = "auto"         # auto（邻域扩散）| manual（用户勾选源节点）
    depth: int = 1                    # 邻域扩散深度（P1 封顶）
    source_topk: int = 3              # 最多取几个源
    quota_mode: str = "soft_bias"     # none | uniform | strength | soft_bias
    quota_tau: float = 0.5            # softmax 温度
    quota_lambda: float = 0.7         # 静态 strength vs 动态 relevance 的融合系数（固定超参）
    quota_mu: float = 0.2             # 软偏置系数
    min_quota: int = 1                # 保底配额，防源被饿死
    edge_topk: int = 4                # 建边候选 top-k（边密度第二道阀）
    edge_types: list[str] = Field(
        default_factory=lambda: ["derives_from", "depends_on", "references", "similar_to"])
    strength_min: float = 0.0         # 邻域展开的边强度下限
    summary_refresh_n: int = 20       # summary 重算触发：新增记忆数
    rel_var_threshold: float = 0.02   # 占位值，待附录 B 标定
    graph_dir_template: str = "data/graph/{user}"   # 图持久化目录
    auto_link: bool = True            # 建节点时自动粗识别建边
    fallback_all: bool = True         # 邻域内无候选时退回全池（冷启动保护）


class BudgetConfig(BaseModel):
    """成本 / 效果 / 速度三角均衡（优化方案 §5）。"""

    default_tier: str = "balanced"    # full | balanced | fast
    adaptive: bool = False            # 实验/消融必须 false（锁定档位，保可复现）
    latency_budget_ms: float = 900.0  # NFR-1 是 1000，留 100ms 余量
    window: int = 20                  # 分位延迟统计窗口
    llm_prior_cache: int = 512        # LLM 先验 LRU 缓存容量
    llm_calls_per_schedule: int = 1   # 每次调度的 LLM 调用预算（0 = 纯静态）
    min_candidates_for_prior: int = 5  # 候选太少时跳过 LLM 先验（门控 few_cand）
    # 单变量扫描旋钮：设置后覆盖档位自带的同名参数，用于 B 系列做单变量归因
    force_candidate: int | None = None   # 强制候选集规模（None = 用档位默认）
    force_lambda: float | None = None    # 强制 Stage A 的 λ（None = 用档位默认）


class SchedulingConfig(BaseModel):
    """调度层全部可调参数（对应 config/srtp.yaml）。"""

    # ---- 动作选择 / 双池 ----
    threshold: float = 0.6            # 动作选择初始阈值（FR-5）
    max_shared: int = 10              # 共享池上限（FR-6）

    # ---- 混合权重（FR-4）----
    alpha: float = 0.6                # LLM 先验权重占比
    beta: float = 0.4                 # 可学习权重占比
    top_k: int = 5                    # 共享池注入上限保护（FR-2）
    candidate_override: int = 50      # 调度候选集规模（统一 50，FR-2）
    task_tag: str = "main_task"       # 当前任务标签（任务注意力）
    llm_prior_samples: int = 1        # LLM 先验采样次数（实时小模型，单次，预算 ≤800ms）
    retrieval_mode: str = "pre_schedule"  # pre_schedule（默认）| async_best_effort（实验对照）
    condition_key: str = "u_tu|research|srtp"  # 当前条件键（user|scenario|business）

    # ---- embedding（D-5 / D-11）----
    embedding_provider: str = "dashscope"   # dashscope / openai / local
    embedding_model: str = "text-embedding-v4"   # embedding 模型名
    embedding_dimensions: int = 1024   # 向量维度（DashScope text-embedding-v4=1024）
    embedding_cache: bool = True       # 启用 embedding 缓存
    embedding_cache_dir: str = "data/embed_cache"  # 落盘缓存目录（真实向量写盘复用，可复现；data/ 已 gitignore）

    # ---- 监控 / 实验（FR-9 / FR-10）----
    log_dir: str = "data/metrics"     # L3 埋点目录
    run_id: str | None = None         # 实验登记号

    # ---- 插件选择（config-as-composition，§12.5；无 store_impl，V2.1）----
    retriever_impl: str = "retriever.bm25"
    attention_impl: str = "attention.full"
    weight_impl: str = "weight.full"
    selector_impl: str = "selector.full"
    embedding_impl: str = "embed.dashscope"
    reward_impl: str = "reward.explicit"
    model_impl: str = "model.deepseek"
    factory_impl: str = "factory.role"

    # ---- 个性化（condition，V2.1 确定 V1 即实现）----
    conditioning_dims: tuple[str, ...] = ("semantic", "task")  # 仅这两维随 condition 调制
    condition_emb_dim: int = 32       # condition_emb 维度

    # ---- 图结构（优化方案；graph.enabled=False 时行为与改造前一致）----
    graph: GraphConfig = Field(default_factory=GraphConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)

    @classmethod
    def load(cls, path: str | Path) -> "SchedulingConfig":
        """从 YAML 读取配置并校验；缺失字段用默认值。"""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"配置文件不存在: {p}")
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        # 允许 YAML 顶层直接写字段；`fixed` 块兼容文档示例写法
        if "fixed" in raw and isinstance(raw["fixed"], dict):
            raw = {**raw["fixed"], **raw}
        return cls(**raw)
