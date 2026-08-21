"""调度中间件（§4.7，核心总入口）。

MemorySchedulingMiddleware 挂载在 ReMeMiddleware 之前（§3.4）：on_reply 先调度（pre_schedule），
on_reasoning 先注入 HintBlock。检索由 BaseRetriever 在常驻池上接管（D-9），ReMe 仅写回。

V2.1：本文件为 headless 核心的中间件骨架 —— 与 AgentScope 中间件基类的完整集成
（on_reply/on_reasoning 的 async generator 语义）在接真实 AgentScope 环境时完成；
本骨架暴露纯 Python 的 `schedule_once()` 同步入口，供 CLI/测试/消融直接驱动。
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from .config import SchedulingConfig
from .condition import ConditionKey
from .coordinator import MemoryCoordinator, ScheduleResult
from .plugins import get_plugin, register_all  # 注册表装配（§12.4/§12.5）
from .resident_pool import ResidentMemoryPool
from .shared_pool import SharedMemoryPool
from .weights import HybridWeightCalculator


class MemorySchedulingMiddleware:
    """调度中间件（headless 骨架）。"""

    def __init__(self, config: SchedulingConfig | None = None,
                 user_id: str = "u_tu", session_id: str = "sess_001",
                 log_path: str = "data/metrics/schedule.jsonl",
                 llm_prior_fn: Optional[Callable[[str], dict]] = None,
                 reme_search_callable: Optional[Callable] = None):
        from .monitor.schedule_logger import ScheduleLogger
        self.config = config or SchedulingConfig()
        self.user_id = user_id
        self.session_id = session_id
        self.logger = ScheduleLogger(log_path)

        # 常驻完整池（per-user 真池）
        pool_path = f"data/reme/{user_id}/main/resident_pool.jsonl"
        self.resident_pool = ResidentMemoryPool(user_id, session_id, pool_path)

        # 共享池（副线 workspace）
        sp_path = f"data/reme/{user_id}/sub1/shared_pool.json"
        self.shared_pool = SharedMemoryPool(max_size=self.config.max_shared, persist_path=sp_path)

        # 检索插件（config-as-composition：按 cfg.retriever_impl 从注册表取，§12.5）
        register_all()
        if self.config.retriever_impl == "retriever.reme":
            self.retriever = get_plugin(self.config.retriever_impl,
                                        reme_search_callable=reme_search_callable)
        elif self.config.retriever_impl == "retriever.vector":
            # vector 检索器绑定常驻池的 faiss 索引（ANN 查询，NFR-2）
            self.retriever = get_plugin(self.config.retriever_impl,
                                        index=self.resident_pool.vector_index)
        else:
            self.retriever = get_plugin(self.config.retriever_impl)

        # 算法层插件：按 config 从注册表装配（config-as-composition，§12.5）
        # 注意：weight.full 用完整 HybridWeightCalculator（含 LLM 先验注入）；
        # 桩/参考实现（uniform/reme）不依赖 llm_prior_fn。
        if self.config.weight_impl == "weight.full":
            self.weights = HybridWeightCalculator(
                alpha=self.config.alpha, beta=self.config.beta,
                prior_samples=self.config.llm_prior_samples,
                conditioning_dims=self.config.conditioning_dims,
                condition_emb_dim=self.config.condition_emb_dim,
                llm_prior_fn=llm_prior_fn,
            )
        else:
            self.weights = get_plugin(self.config.weight_impl)

        self.attention = get_plugin(self.config.attention_impl)
        if self.config.selector_impl == "selector.full":
            self.selector = get_plugin(self.config.selector_impl,
                                       threshold=self.config.threshold,
                                       max_shared=self.config.max_shared)
        else:
            self.selector = get_plugin(self.config.selector_impl,
                                       top_k=self.config.top_k)

        # 协调器（总装配）
        self.coordinator = MemoryCoordinator(
            resident_pool=self.resident_pool,
            retriever=self.retriever,
            scorer=self.attention,
            weights=self.weights,
            selector=self.selector,
            shared_pool=self.shared_pool,
            user_id=user_id, session_id=session_id,
            logger=self.logger,
        )

    # ---- headless 同步入口（CLI / 测试 / 消融直接驱动）----

    def schedule_once(self, query: str, query_emb=None, task_tag: str | None = None,
                      now: float | None = None,
                      enabled_dims: set[str] | None = None,
                      condition: ConditionKey | None = None) -> ScheduleResult:
        """一次完整调度：LLM 先验（若有文本）→ 协调器 schedule → 返回结果。"""
        import time
        now = now if now is not None else time.time()
        if query_emb is None:
            query_emb = _dummy_emb(self.config.embedding_dimensions)
        # LLM 先验：query 文本 → 轻量小模型（可注入 fn；未注入回退默认权重）。
        # 桩/参考权重实现（weight.uniform / weight.reme）无 llm_prior/set_prior，跳过。
        if hasattr(self.weights, "llm_prior") and hasattr(self.weights, "set_prior"):
            prior = self.weights.llm_prior(query)
            self.weights.set_prior(prior)
        cond = condition or ConditionKey.parse(self.config.condition_key)
        return self.coordinator.schedule(
            query=query, query_emb=query_emb, task_tag=task_tag,
            now=now, enabled_dims=enabled_dims, condition=cond,
            top_k=self.config.candidate_override,
        )

    def flush(self) -> None:
        self.logger.flush()
        self.shared_pool.persist()
        self.resident_pool.persist()


def _dummy_emb(dim: int = 1024):
    """无外部 embedding 时的占位向量（零向量 + 单位扰动，保证 cosine 可算）。"""
    import numpy as np
    emb = np.zeros(dim, dtype=np.float32)
    emb[0] = 1.0
    return emb
