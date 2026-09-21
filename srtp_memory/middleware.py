"""调度中间件（§4.7，核心总入口）+ 图架构装配（优化方案 §7.2）。

MemorySchedulingMiddleware 挂载在 ReMeMiddleware 之前（§3.4）：on_reply 先调度（pre_schedule），
on_reasoning 先注入 HintBlock。检索由 BaseRetriever 在常驻池上接管（D-9），ReMe 仅写回。

V2.1：暴露纯 Python 的 `schedule_once()` 同步入口，供 CLI/测试/消融直接驱动。
V3.0（图架构）：装配 `GraphStore` / `EdgeBuilder` / `SourceSelector` / `BudgetController`；
`config.graph.enabled=False`（默认）时**完全走改造前路径**。
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from .config import SchedulingConfig
from .condition import ConditionKey
from .coordinator import MemoryCoordinator, ScheduleResult
from .plugins import get_plugin, register_all  # 注册表装配（§12.4/§12.5）
from .resident_pool import ResidentMemoryPool
from .shared_pool import SharedMemoryPool
from .weights import DEFAULT_WEIGHTS, HybridWeightCalculator
from .attention import MemoryCandidate


class MemorySchedulingMiddleware:
    """调度中间件（headless 骨架 + 图架构）。"""

    def __init__(self, config: SchedulingConfig | None = None,
                 user_id: str = "u_tu", session_id: str = "sess_001",
                 log_path: str = "data/metrics/schedule.jsonl",
                 llm_prior_fn: Optional[Callable[[str], dict]] = None,
                 reme_search_callable: Optional[Callable] = None,
                 resident_pool_path: Optional[str] = None,
                 shared_pool_path: Optional[str] = None,
                 graph_dir: Optional[str] = None):
        """resident_pool_path / shared_pool_path：可选覆盖双池落盘路径。
        默认 data/reme/<user>/main|sub1（真实环境）；测试可注入 tmp_path 实现完全隔离。
        graph_dir：图持久化目录（默认按 config.graph.graph_dir_template 生成）。"""
        from .monitor.schedule_logger import ScheduleLogger
        self.config = config or SchedulingConfig()
        self.user_id = user_id
        self.session_id = session_id
        self.logger = ScheduleLogger(log_path)

        # 真实 embedding 后端（D-5/D-11）：按 config.embedding_impl 从注册表装配。
        # 未配置 key 时插件内部自动降级为确定性占位向量，pipeline 不中断。
        register_all()
        self.embedding = get_plugin(self.config.embedding_impl,
                                    dim=self.config.embedding_dimensions,
                                    model_name=self.config.embedding_model,
                                    cache_dir=self.config.embedding_cache_dir)

        # 真实 LLM 先验：未显式注入时，按 config.model_impl 装配轻量小模型。
        # 无 key/解析失败回退 DEFAULT_WEIGHTS（与 D-11 同哲学），测试/离线不触发网络。
        if llm_prior_fn is None and self.config.weight_impl == "weight.full":
            llm_prior_fn = _make_real_llm_prior(self.config.model_impl)

        # 常驻完整池（per-user 真池）
        pool_path = resident_pool_path or f"data/reme/{user_id}/main/resident_pool.jsonl"
        self.resident_pool = ResidentMemoryPool(user_id, session_id, pool_path)

        # 共享池（副线 workspace）
        sp_path = shared_pool_path or f"data/reme/{user_id}/sub1/shared_pool.json"
        self.shared_pool = SharedMemoryPool(max_size=self.config.max_shared, persist_path=sp_path)

        # 检索插件（config-as-composition：按 cfg.retriever_impl 从注册表取，§12.5）
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

        # ---------------- 图架构装配（§7.2）----------------
        self.graph_cfg = self.config.graph
        self.budget_cfg = self.config.budget
        self.graph = None
        self.edge_builder = None
        self.source_selector = None
        self.budget = None
        self.cur_node = "main"
        if self.graph_cfg.enabled:
            from .graph import BudgetController, EdgeBuilder, GraphStore, SourceSelector
            gdir = graph_dir or self.graph_cfg.graph_dir_template.format(user=user_id)
            self.graph = GraphStore(user_id=user_id, root_dir=gdir).load()
            self.edge_builder = EdgeBuilder(
                self.graph, cfg=self.graph_cfg,
                embed_fn=lambda t: self._encode_counting(t),
            )
            self.source_selector = SourceSelector(self.graph, cfg=self.graph_cfg)
            self.budget = BudgetController(
                cfg=self.budget_cfg, tier=self.budget_cfg.default_tier)

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
            source_selector=self.source_selector,
            graph_cfg=self.config if self.graph_cfg.enabled else None,
            budget=self.budget,
        )

    # ---- headless 同步入口（CLI / 测试 / 消融直接驱动）----

    def schedule_once(self, query: str, query_emb=None, task_tag: str | None = None,
                      now: float | None = None,
                      enabled_dims: set[str] | None = None,
                      condition: ConditionKey | None = None,
                      cur_node: str | None = None,
                      manual_sources=None) -> ScheduleResult:
        """一次完整调度：预算档位 → (Stage A) → LLM 先验(门控/缓存) → 协调器 schedule。"""
        import time
        now = now if now is not None else time.time()
        cond = condition or ConditionKey.parse(self.config.condition_key)

        # 预算：每次调度重置成本计数；档位决定候选集规模与 λ
        top_k = self.config.candidate_override
        lambda_override = None
        if self.budget is not None:
            self.budget.reset_counters()
            top_k = self.budget.candidate_override
            lambda_override = self.budget.lambda_override

        # 真实 embedding：调用装配好的后端（无 key 自动降级占位向量，D-11）
        if query_emb is None:
            hit = _embedding_is_cached(self.embedding, query)
            query_emb = self.embedding.encode(query)
            if self.budget is not None and not hit:
                self.budget.note_embed_miss(1)

        # LLM 先验：门控 + 缓存（成本侧机制一）→ 未命中才真调用
        self._apply_prior(query, cond, n_candidates=top_k)

        res = self.coordinator.schedule(
            query=query, query_emb=query_emb, task_tag=task_tag,
            now=now, enabled_dims=enabled_dims, condition=cond,
            top_k=top_k, cur_node=cur_node or self.cur_node,
            manual_sources=manual_sources,
            lambda_override=lambda_override,
        )
        if self.budget is not None:
            self.budget.note_latency(res.latency_ms.get("total", 0.0))
            changed = self.budget.maybe_adjust()
            if changed:
                res.perf_tier = self.budget.tier
        return res

    def _apply_prior(self, query: str, cond: ConditionKey,
                     n_candidates: int = 50) -> None:
        """LLM 先验（α 侧）：门控 → 缓存 → 调用；不可用时回落 DEFAULT_WEIGHTS。"""
        if not (hasattr(self.weights, "set_prior") and hasattr(self.weights, "llm_prior")):
            return
        if self.budget is None:
            self.weights.set_prior(self.weights.llm_prior(query))
            return
        from .graph.budget import PriorCache
        key = PriorCache.norm_key(query, "|".join(x or "" for x in cond.to_tuple()))
        need, cached, _reason = self.budget.should_call_llm_prior(n_candidates, key)
        if cached is not None:
            self.weights.set_prior(cached)
            return
        if need:
            prior = self.weights.llm_prior(query)
            self.budget.note_prior(key, prior, called=True)
            self.weights.set_prior(prior)
        else:
            # 不调用：显式回落默认权重，避免沿用上一次 query 的陈旧先验
            self.weights.set_prior(dict(DEFAULT_WEIGHTS))

    @property
    def last_prior_gate(self) -> dict:
        """最近一次先验门控统计（调试/埋点用）。"""
        return dict(self.budget.gate_stats) if self.budget is not None else {}

    # ---- 图节点管理 ----

    def create_node(self, topic: str, first_query: str = "",
                    session_id: str | None = None,
                    summary: str = "") -> Optional[str]:
        """创建一个 topic 会话节点（= 一个聊天会话进程），并做**粗识别建边**。

        返回 node_id；`graph.enabled=False` 时返回 None（不做图）。
        """
        if self.graph is None:
            return None
        node = self.graph.create_node(
            topic=topic, session_id=session_id or self.session_id, summary=summary)
        if self.edge_builder is not None:
            node.emb = self.edge_builder.embed(f"{topic} {first_query}".strip())
            if self.graph_cfg.auto_link:
                self.edge_builder.on_node_created(node, first_query, auto_link=True)
            else:
                self.graph.persist()
        return node.node_id

    def refine_node(self, node_id: str, summary: str) -> dict:
        """summary 生成后精识别建边（可覆盖粗识别，写 evidence 溯源）。"""
        if self.graph is None or self.edge_builder is None:
            return {}
        node = self.graph.get_node(node_id)
        if node is None:
            return {}
        return self.edge_builder.refine(node, summary)

    def use_node(self, node_id: str) -> None:
        """把后续调度的"当前节点"切到 node_id（生成候选视野的起点）。"""
        self.cur_node = node_id

    # ---- 记忆写入 ----

    def add_memory(self, text: str, memory_id: str | None = None,
                   task_tag: str | None = None, path: str = "memory.md",
                   timestamp: float | None = None,
                   node_id: str | None = None) -> "MemoryCandidate":
        """写入一条记忆（真实场景由 ReMe auto_memory / 副线回写触发）。

        关键：入库即算 embedding 并 upsert 进常驻池向量索引 —— 否则 retriever.vector
        无语料可检（空结果）。embedding 走装配后端（真实 API + 落盘缓存 / 无 key 降级占位）。
        不触发 LLM，纯 embedding 调用；与 schedule_once 解耦，可批量灌库（消融数据集加载）。

        图架构（§2.3）：`node_id` 为生产者节点 → 写入 `owner_node` **标记**（不复制记忆），
        节点专属记忆 = 大池按该标记过滤的零拷贝视图。
        """
        import time
        import uuid

        ts = timestamp if timestamp is not None else time.time()
        mid = memory_id or f"m_{uuid.uuid4().hex[:12]}"
        hit = _embedding_is_cached(self.embedding, text)
        emb = self.embedding.encode(text)
        owner = node_id or self.cur_node or "main"
        cand = MemoryCandidate(
            memory_id=mid, text=text, path=path, timestamp=ts,
            task_tag=task_tag, user_id=self.user_id, session_id=self.session_id,
            embedding=emb, owner_node=owner,
        )
        self.resident_pool.upsert(cand)
        if self.budget is not None and not hit:
            self.budget.note_embed_miss(1)
        if self.graph is not None:
            self.graph.bump_memory_count(owner, 1)
        return cand

    def flush(self) -> None:
        self.logger.flush()
        self.shared_pool.persist()
        self.resident_pool.persist()
        if self.graph is not None:
            self.graph.persist()

    # ---- 状态 / 诊断 ----

    def graph_status(self) -> dict:
        """图 + 预算的当前状态（供控制台 / 实验报告）。"""
        out: dict[str, Any] = {
            "graph_enabled": self.graph is not None,
            "cur_node": self.cur_node,
        }
        if self.graph is not None:
            out["graph"] = self.graph.stats()
            out["node_index"] = self.resident_pool.node_index.counts()
            if self.edge_builder is not None:
                out["refine_coverage"] = self.edge_builder.refine_coverage()
        if self.budget is not None:
            out["budget"] = self.budget.to_dict()
        return out

    def api_status(self) -> dict:
        """真实 API 运行状态（供控制台/监控告警）。

        返回每个后端 {mode: real|fallback, error: None|{kind,code,message,ts}}。
        kind ∈ balance/auth/forbidden/rate_limit/network/other —— 便于上层提示"余额不足"等。
        """
        llm_client = None
        llm_fn = getattr(self.weights, "_llm_prior_fn", None)
        if llm_fn is not None:
            llm_client = getattr(llm_fn, "client", None)
        return {
            "embedding": {
                "mode": _api_mode(self.embedding),
                "error": getattr(self.embedding, "last_error", None),
            },
            "llm_prior": {
                "mode": _api_mode(llm_client),
                "error": getattr(llm_client, "last_error", None) if llm_client else None,
            },
        }

    # ---- 内部 ----

    def _encode_counting(self, text: str) -> list[float]:
        """供建边器使用的 embedding 入口（复用同一后端与落盘缓存）。"""
        hit = _embedding_is_cached(self.embedding, text)
        v = self.embedding.encode(text)
        if self.budget is not None and not hit:
            self.budget.note_embed_miss(1)
        try:
            return list(v.tolist()) if hasattr(v, "tolist") else list(v)
        except Exception:  # noqa: BLE001
            return []


def _embedding_is_cached(embedding, text: str) -> bool:
    """探测 embedding 是否命中缓存（用于"只计未命中"的成本埋点）。"""
    cache = getattr(embedding, "_cache", None)
    try:
        return isinstance(cache, dict) and text in cache
    except Exception:  # noqa: BLE001
        return False


def _make_real_llm_prior(model_impl: str):
    """装配真实 LLM 先验函数（D-13）：调用 config.model_impl 轻量小模型。

    提示词 + JSON 解析失败 / 无 key / 网络异常 → 回退 DEFAULT_WEIGHTS
    （与 D-11 同哲学：真实链路主跑，异常自动降级，测试/离线不触发网络）。
    """
    from .weights import LLM_PRIOR_PROMPT

    try:
        client = get_plugin(model_impl)
    except Exception:  # noqa: BLE001 - 插件缺失时不阻断装配
        client = None
    if client is None or not getattr(client, "api_key", None):
        return None

    import json
    import re

    def _prior(query: str) -> dict:
        try:
            raw = client.complete(f"{LLM_PRIOR_PROMPT}\n用户查询：{query}", max_tokens=64)
            m = re.search(r"\{[^}]*\}", raw or "")
            w = json.loads(m.group(0)) if m else {}
            w = {k: float(v) for k, v in w.items() if k in DEFAULT_WEIGHTS}
            if len(w) != 4 or sum(w.values()) <= 0:
                return dict(DEFAULT_WEIGHTS)
            s = sum(w.values())
            return {k: v / s for k, v in w.items()}
        except Exception:  # noqa: BLE001 - 解析失败/网络异常降级默认权重
            return dict(DEFAULT_WEIGHTS)

    # 挂载 client 供 api_status() 读取 last_error（模型插件失败不再静默）
    _prior.client = client  # type: ignore[attr-defined]
    return _prior


def _api_mode(client) -> str:
    """client 是否处于真实模式（配置了 key）。"""
    return "real" if getattr(client, "api_key", None) else "fallback"
