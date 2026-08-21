"""调度层配置模型（§4.1）。

对应 config/srtp.yaml。所有可调参数集中于此，缺失字段用默认值。
V2.1：无 store_impl —— 双池（常驻池/共享池）为内核具体类，不做存储后端插件化。
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


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
    embedding_cache: bool = True       # 启用 AgentScope 内置 embedding 缓存

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
