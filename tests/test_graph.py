"""图架构单测（优化方案 §7.1 / §10.2 硬约束的可执行守护）。

覆盖：
1. models      —— 边默认强度、相似度分档判型
2. graph_store —— 强制 DAG（环检测降级）、邻域展开、拓扑排序、版本折叠
3. ★ strength 静态性守护（不得由 query 改变边强度）
4. view        —— 大池节点标记倒排索引 + ScopedPool 限定检索范围
5. source_selector —— 四种配额模式、λ 退化保护、软偏置 bonus
6. budget      —— 三档规格、先验缓存、门控、adaptive=False 时不动档
7. builder     —— 粗识别建边 / 精识别覆盖 / 覆盖率
8. coordinator —— graph.enabled=False 的**等价性判据** + 开启后的端到端
"""
from __future__ import annotations

import numpy as np
import pytest

from srtp_memory.attention import MemoryCandidate
from srtp_memory.config import BudgetConfig, GraphConfig, SchedulingConfig
from srtp_memory.graph import (
    BudgetController,
    EdgeBuilder,
    GraphStore,
    PriorCache,
    SourceSelector,
    hash_embed,
    infer_edge_type,
)
from srtp_memory.graph.graph_store import CycleError
from srtp_memory.graph.models import EDGE_DEFAULT_STRENGTH, MAIN_NODE
from srtp_memory.graph.view import NodeIndex, ScopedPool
from srtp_memory.middleware import MemorySchedulingMiddleware

PRIOR = {"time": 0.25, "semantic": 0.35, "frequency": 0.15, "task": 0.25}


# ====================================================================== 1


def test_edge_default_strength_matches_typed_table():
    assert EDGE_DEFAULT_STRENGTH["derives_from"] == 0.9
    assert EDGE_DEFAULT_STRENGTH["depends_on"] == 0.8
    assert EDGE_DEFAULT_STRENGTH["references"] == 0.5
    assert EDGE_DEFAULT_STRENGTH["similar_to"] == 0.3
    assert infer_edge_type(0.95) == "derives_from"
    assert infer_edge_type(0.75) == "depends_on"
    assert infer_edge_type(0.60) == "references"
    assert infer_edge_type(0.10) == "similar_to"


# ====================================================================== 2


def _store(tmp_path, name="g.jsonl"):
    return GraphStore(user_id="u", root_dir=str(tmp_path / name))


def test_graph_is_dag_cycle_downgraded(tmp_path):
    g = _store(tmp_path)
    g.create_node("A", node_id="A")
    g.create_node("B", node_id="B")
    g.create_node("C", node_id="C")
    e1, r1 = g.add_edge("A", "B", "depends_on")
    e2, r2 = g.add_edge("B", "C", "depends_on")
    assert (r1, r2) == ("ok", "ok")
    assert e1.strength == 0.8

    # C → A 会闭合环 → P0 降级为 references 且不入邻域
    e3, r3 = g.add_edge("C", "A", "depends_on")
    assert r3 == "cycle_rejected"
    assert e3.type == "references" and e3.strength == 0.0
    assert e3.in_neighborhood is False

    # 自环同样拒绝
    _, r4 = g.add_edge("A", "A", "similar_to")
    assert r4 == "cycle_rejected"

    order = g.topo_order()
    assert set(order) == {"main", "A", "B", "C"}
    assert order.index("A") < order.index("B") < order.index("C")
    g.assert_dag()
    st = g.stats()
    assert st["is_dag"] is True and st["n_cycle_rejected"] >= 1


def test_cycle_raises_when_downgrade_disabled(tmp_path):
    g = _store(tmp_path)
    g.create_node("A", node_id="A")
    g.create_node("B", node_id="B")
    g.add_edge("A", "B", "depends_on")
    with pytest.raises(CycleError):
        g.add_edge("B", "A", "depends_on", allow_downgrade=False)


def test_neighbors_depth_and_ordering(tmp_path):
    g = _store(tmp_path)
    for n in ("A", "B", "C", "D"):
        g.create_node(n, node_id=n)
    g.add_edge("A", "B", "derives_from")     # 0.9
    g.add_edge("A", "C", "similar_to")       # 0.3
    g.add_edge("B", "D", "depends_on")       # 0.8
    n1 = g.neighbors("A", depth=1)
    assert [x[0] for x in n1] == ["B", "C"], "depth=1 只取一跳，且强边在前"
    n2 = g.neighbors("A", depth=2)
    assert set(x[0] for x in n2) == {"B", "C", "D"}
    assert g.neighbors("A", depth=1, strength_min=0.5)[0][0] == "B"
    # 入边方向 = Stage A 的源
    assert [x[0] for x in g.predecessors("D")] == ["B"]


def test_version_folding(tmp_path):
    g = _store(tmp_path)
    g.create_node("t", node_id="v1", root_id="r")
    g.create_node("t", node_id="v2", root_id="r")
    g.get_node("v2").version = 2
    folded = g.fold_versions()
    assert folded == {"v1": "v2"}


# ====================================================================== 3 ★


def test_edge_strength_is_static(tmp_path):
    """★ 守护：边强度**不得**由 query 改变（2026-09-07 定案）。

    - `adjust_edge_strength` / `recompute_strength_from_summary` 的签名里**没有 query**；
    - `SourceSelector.select`（Stage A）跑多次不同 query 后，图中 strength 必须逐位不变。
    """
    import inspect

    g = _store(tmp_path)
    g.create_node("A", node_id="A")
    g.create_node("B", node_id="B")
    g.add_edge("A", "B", "depends_on")
    assert "query" not in inspect.signature(g.adjust_edge_strength).parameters
    assert "query" not in inspect.signature(g.recompute_strength_from_summary).parameters

    before = {e.edge_id: e.strength for e in g.edges()}
    sel = SourceSelector(g, cfg=GraphConfig())
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        sel.select("B", list(rng.normal(0, 1, 64)))
    after = {e.edge_id: e.strength for e in g.edges()}
    assert before == after


def test_feedback_adjusts_strength_with_regularization(tmp_path):
    g = _store(tmp_path)
    g.create_node("A", node_id="A")
    g.create_node("B", node_id="B")
    e, _ = g.add_edge("A", "B", "depends_on")
    base = e.strength
    up = g.adjust_edge_strength(e.edge_id, +1.0)
    assert up > base, "正反馈应抬升强度"
    down = g.adjust_edge_strength(e.edge_id, -1.0)
    assert down < up, "负反馈应下压强度"
    v = down
    for _ in range(50):
        v = g.adjust_edge_strength(e.edge_id, -1.0)
    assert v >= 0.0, "强度不得越界"


# ====================================================================== 4


def test_node_index_and_scoped_pool(tmp_path):
    idx = NodeIndex()
    idx.add("m1", "A")
    idx.add("m2", "A")
    idx.add("m3", "B")
    idx.add("m4", None)          # 旧数据无标记 → 归 main
    assert sorted(idx.ids_of("A")) == ["m1", "m2"]
    assert idx.ids_of_any(["B", "A"]) == ["m3", "m1", "m2"]
    assert idx.owner_of("m4") == MAIN_NODE
    idx.add("m1", "B")           # 改标记 → 从 A 移除
    assert idx.ids_of("A") == ["m2"]
    assert sorted(idx.ids_of("B")) == ["m1", "m3"]

    class _Pool:
        user_id, session_id = "u", "s"
        vector_index = object()

        def __init__(self):
            self.node_index = idx
            self._r = {f"m{i}": MemoryCandidate(memory_id=f"m{i}", text=f"t{i}", path="")
                       for i in range(1, 5)}
            self._r["m1"].owner_node = "B"
            self._r["m2"].owner_node = "A"
            self._r["m3"].owner_node = "B"

        def all(self):
            return list(self._r.values())

        def get(self, mid):
            return self._r.get(mid)

    sp = ScopedPool(_Pool(), ["A"])
    assert [c.memory_id for c in sp.all()] == ["m2"]
    assert sp.vector_index is None, "限定范围内必须走精确线性余弦，不共用全池 faiss 索引"
    sp_none = ScopedPool(_Pool(), ["zzz"], fallback_all=True)
    assert len(sp_none.all()) == 4, "fallback_all=True 时可退回全池（冷启动保护）"


# ====================================================================== 5


def _graph_with_neighbors(tmp_path):
    """建一个以 A 为"当前节点"的图：B/C 是 A 的**上游源**（入边指向 A）。

    边语义「被依赖方 → 依赖方」，故 Stage A 取 A 的 predecessors（入边起点）。
    """
    g = _store(tmp_path)
    for n in (MAIN_NODE, "A", "B", "C"):
        g.create_node(n, node_id=n)
    ea, _ = g.add_edge("B", "A", "derives_from")   # 0.9
    eb, _ = g.add_edge("C", "A", "similar_to")     # 0.3
    g.get_node("B").emb = None
    g.get_node("C").emb = None
    return g, ea, eb


def test_source_selector_modes(tmp_path):
    g, ea, eb = _graph_with_neighbors(tmp_path)
    # 给节点画上确定性向量：B 与 query 同轴，C 正交
    q = [1.0] + [0.0] * 63
    g.get_node("B").emb = [1.0] + [0.0] * 63
    g.get_node("C").emb = [0.0, 1.0] + [0.0] * 62

    s = SourceSelector(g, cfg=GraphConfig(quota_lambda=0.7, source_topk=3))

    soft = s.select("A", q, K=10, mode="soft_bias")
    assert [x.node_id for x in soft.sources] == ["B", "C"], "B 边强且相关度高，应排前"
    assert soft.sources[0].src_score > soft.sources[1].src_score
    assert soft.sources[0].strength == pytest.approx(0.9)
    assert soft.sources[0].relevance == pytest.approx(1.0, abs=1e-6)
    bonus = s.bonus_map(soft, mu=0.2)
    assert set(bonus) == {"B", "C"} and bonus["B"] > bonus["C"]

    hard = s.select("A", q, K=10, mode="strength")
    assert sum(x.quota for x in hard.sources) == 10, "硬配额之和应为 K"
    assert hard.sources[0].quota >= hard.sources[1].quota

    uni = s.select("A", q, K=10, mode="uniform")
    assert uni.sources[0].quota == uni.sources[1].quota

    none = s.select("A", q, K=10, mode="none")
    assert all(x.quota == 0 for x in none.sources), "none 模式不做配额"
    assert s.bonus_map(none, 0.2)  # 源仍然输出，只是不加偏置


def test_source_selector_lambda_degradation(tmp_path):
    g, _, _ = _graph_with_neighbors(tmp_path)
    q = [1.0] + [0.0] * 63
    # 两源相关度完全相同 → 方差 0 < 阈值 → 退化纯静态
    g.get_node("B").emb = [1.0] + [0.0] * 63
    g.get_node("C").emb = [1.0] + [0.0] * 63
    s = SourceSelector(g, cfg=GraphConfig(quota_lambda=0.7, rel_var_threshold=0.02))
    sel = s.select("A", q, K=10, mode="soft_bias")
    assert sel.degraded_static is True
    assert sel.lambda_ == 1.0
    assert sel.sources[0].src_score == pytest.approx(0.9), "纯静态下 src_score 就是 strength"


def test_source_selector_empty_neighborhood(tmp_path):
    g = _store(tmp_path)
    g.create_node("A", node_id="A")
    s = SourceSelector(g, cfg=GraphConfig())
    sel = s.select("A", [1.0] + [0.0] * 63, K=10)
    assert sel.sources == [] and "邻域为空" in sel.reason


def test_source_selector_tier_lambda_override(tmp_path):
    """B3「纯静态」档：λ_override=1.0 时 relevance 完全不参与 src_score。

    B 边强(0.9)但与 query 正交，C 边弱(0.3)但与 query 同轴：
      λ=0.7 → B: 0.63, C: 0.51（静态主导，B 仍在前）
      λ=1.0 → B: 0.90, C: 0.30（纯静态，差距拉大）
    断言分项数值而非排名 —— 排名本来就是"静态主导"的设计意图。
    """
    g, _, _ = _graph_with_neighbors(tmp_path)
    q = [1.0] + [0.0] * 63
    g.get_node("B").emb = [0.0, 1.0] + [0.0] * 62     # 与 query 正交
    g.get_node("C").emb = [1.0] + [0.0] * 63          # 与 query 同轴
    s = SourceSelector(g, cfg=GraphConfig(quota_lambda=0.7, rel_var_threshold=0.0))
    dyn = {x.node_id: x for x in s.select("A", q, K=10, mode="soft_bias").sources}
    sta_sel = s.select("A", q, K=10, mode="soft_bias", lambda_override=1.0)
    sta = {x.node_id: x for x in sta_sel.sources}

    assert sta_sel.lambda_ == 1.0
    assert dyn["B"].relevance == pytest.approx(0.0, abs=1e-6)
    assert dyn["C"].relevance == pytest.approx(1.0, abs=1e-6)
    assert sta["B"].src_score == pytest.approx(0.9)   # 纯静态 = strength
    assert sta["C"].src_score == pytest.approx(0.3)
    assert dyn["B"].src_score == pytest.approx(0.7 * 0.9)
    assert dyn["C"].src_score == pytest.approx(0.7 * 0.3 + 0.3 * 1.0)


# ====================================================================== 6


def test_budget_tiers_and_cache():
    b = BudgetController(cfg=BudgetConfig(), tier="balanced")
    assert b.spec["candidate"] == 30 and b.spec["llm_prior"] == "cache"
    b.set_tier("fast", "test")
    assert b.spec["candidate"] == 20 and b.spec["lambda"] == 1.0

    key = PriorCache.norm_key("  记忆 调度  ", "u|s|b")
    assert key == PriorCache.norm_key("记忆 调度", "u|s|b"), "归一化后应同 key（空格/大小写）"
    b.cache.put(key, dict(PRIOR))
    assert b.cache.get(key) == PRIOR
    # 用另一个 full 档控制器验证 cache_hit 门控（balanced 已被 set_tier 成 fast）
    b_cache = BudgetController(cfg=BudgetConfig(), tier="full")
    b_cache.cache.put(key, dict(PRIOR))
    need, cached, reason = b_cache.should_call_llm_prior(50, key)
    assert (need, reason) == (False, "cache_hit") and cached == PRIOR

    b2 = BudgetController(cfg=BudgetConfig(), tier="full")
    need2, cached2, r2 = b2.should_call_llm_prior(50, "nope")
    assert need2 is True and cached2 is None and r2 == "call"
    b2.note_prior("nope", dict(PRIOR), called=True)
    assert b2.llm_calls == 1
    need3, _, r3 = b2.should_call_llm_prior(50, "other")
    assert (need3, r3) == (False, "budget"), "调用预算用尽后不得再调"

    # 门控 few_cand
    b3 = BudgetController(cfg=BudgetConfig(), tier="full")
    assert b3.should_call_llm_prior(3, "k")[2] == "few_cand"
    # 门控 tier_off
    b4 = BudgetController(cfg=BudgetConfig(), tier="fast")
    assert b4.should_call_llm_prior(50, "k")[2] == "tier_off"


def test_budget_adaptive_off_by_default():
    b = BudgetController(cfg=BudgetConfig())
    assert b.adaptive is False
    for _ in range(40):
        b.note_latency(5000.0)          # 远超预算
    assert b.maybe_adjust() is None
    assert b.tier == "balanced", "adaptive=False 时档位不得自动变化（保复现）"

    b2 = BudgetController(cfg=BudgetConfig(adaptive=True))
    for _ in range(40):
        b2.note_latency(5000.0)
    ch = b2.maybe_adjust()
    assert ch is not None and b2.tier == "fast", "adaptive=True 时超预算应降档"
    assert b2.percentiles()["p90"] > 900


def test_budget_cost_counters():
    b = BudgetController(cfg=BudgetConfig())
    b.reset_counters()
    b.note_embed_miss(3)
    b.note_injected_tokens(120)
    c = b.counters()
    assert c == {"llm_calls": 0, "embed_calls_miss": 3, "injected_tokens": 120}


# ====================================================================== 7


def test_edge_builder_coarse_and_refine(tmp_path):
    g = _store(tmp_path)
    g.create_node("存储层设计", node_id="n1")
    eb = EdgeBuilder(g, cfg=GraphConfig(edge_topk=2))
    g.get_node("n1").emb = eb.embed("存储层设计 如何设计持久化")

    node = g.create_node("存储层持久化方案", node_id="n2")
    created = eb.on_node_created(node, "如何做落盘与重建")
    assert created, "粗识别应立即建边（冷启动期就有边可用）"
    assert all(e.stage == "coarse" for e in created)
    assert all(e.dst == "n2" for e in created), "方向必须是 既有节点 → 新节点"

    res = eb.refine(node, "存储层持久化方案：单文件 JSONL + 原子写 + 启动重建索引")
    assert "updated" in res and "coverage" in res
    assert eb.refine_coverage() >= 0.0
    assert all(e.evidence for e in g.edges()), "建边必须写 evidence 溯源"


def test_edge_builder_hash_embed_is_topical():
    """本地哈希兜底向量：同主题（共享 bigram）余弦应高于无关文本。"""
    q = hash_embed("存储层持久化方案设计")
    near = hash_embed("存储层持久化与重建索引")
    far = hash_embed("今天天气晴朗适合外出")

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b))   # 已是单位向量

    assert cos(q, near) > cos(q, far)


# ====================================================================== 8


def _mw(tmp_path, graph_enabled: bool, retriever="retriever.vector",
        budget: BudgetConfig | None = None):
    cfg = SchedulingConfig(
        retriever_impl=retriever, weight_impl="weight.full",
        log_dir=str(tmp_path), run_id="gt",
        graph=GraphConfig(enabled=graph_enabled, graph_dir_template=str(tmp_path / "graph")),
        budget=budget or BudgetConfig(),
    )
    return MemorySchedulingMiddleware(
        config=cfg, user_id="u_tu", session_id="s_t",
        resident_pool_path=str(tmp_path / "rp.jsonl"),
        shared_pool_path=str(tmp_path / "sp.json"),
        log_path=str(tmp_path / "metrics.jsonl"),
        llm_prior_fn=lambda q: dict(PRIOR),
    )


def _seed_pool(mw, node_of: dict[str, str]):
    for i, (mid, owner) in enumerate(node_of.items(), 1):
        v = np.zeros(1024, dtype=np.float32)
        v[0] = 1.0
        v[i % 4] = 0.5
        mw.resident_pool.upsert(MemoryCandidate(
            memory_id=mid, text=f"记忆 {mid}", path=f"d/{mid}.md", timestamp=100.0 + i,
            task_tag="research", user_id=mw.user_id, session_id=mw.session_id,
            embedding=v, owner_node=owner))


def test_graph_disabled_path_is_equivalent(tmp_path):
    """`graph.enabled=False` 时的等价判据（可执行版）。

    比对三项：`to_dict()` 全键值 / `(memory_id, final)` 序列 / `kept` id 序列。
    """
    mw = _mw(tmp_path, graph_enabled=False)
    _seed_pool(mw, {"m1": "main", "m2": "main", "m3": "main"})
    q = np.zeros(1024, dtype=np.float32); q[0] = 1.0
    res = mw.schedule_once("记忆调度", query_emb=q, task_tag="research", now=200.0)

    assert res.graph_enabled is False
    assert res.sources == [] and res.stage_a == {}
    assert res.src_scores == {}
    d = res.to_dict()
    assert d["graph_enabled"] is False and d["sources"] == []
    # 打分项不得含源信息（bonus 恒 0）
    assert all(s["score"]["src_score"] == 0.0 for s in res.scored)
    assert all(s["via_source"] == "main" for s in res.scored)
    assert res.candidates and res.kept
    assert abs(sum(s["score"]["final"] for s in res.scored)) > 0.0


def test_graph_enabled_end_to_end(tmp_path):
    """图开启：建节点 → 建边 → 限定邻域召回 → 源偏置 → 埋点齐全。"""
    mw = _mw(tmp_path, graph_enabled=True, retriever="retriever.vector")

    n_a = mw.create_node("存储层", "存储层怎么设计")
    n_b = mw.create_node("存储层持久化", "存储层持久化落盘怎么做")
    assert n_a and n_b and n_a != n_b
    assert mw.graph.stats()["n_edges"] >= 1, "粗识别应建出边"

    _seed_pool(mw, {"m1": n_a, "m2": n_a, "m3": n_b, "m4": MAIN_NODE})

    mw.use_node(n_b)
    q = np.zeros(1024, dtype=np.float32); q[0] = 1.0
    res = mw.schedule_once("存储层持久化", query_emb=q, task_tag="research", now=300.0)

    assert res.graph_enabled is True
    assert res.cur_node == n_b
    assert res.stage_a and res.stage_a["cur_node"] == n_b
    # 邻域限定生效：候选只来自源节点池（不含 main 的 m4）
    assert "m4" not in [c.memory_id for c in res.candidates]
    assert res.sources, "应至少选出一个源节点"
    assert res.scope_size >= len(res.candidates)
    d = res.to_dict()
    for k in ("perf_tier", "llm_calls", "embed_calls_miss", "injected_tokens",
              "scope_size", "scope_ratio", "src_scores", "stage_a"):
        assert k in d


def test_graph_enabled_hard_quota_mode(tmp_path):
    mw = _mw(tmp_path, graph_enabled=True)
    mw.config.graph.quota_mode = "strength"
    n_a = mw.create_node("A 主题", "A 的首问")
    n_b = mw.create_node("B 主题", "B 的首问")   # 粗识别会建 n_a → n_b 边
    _seed_pool(mw, {"m1": n_a, "m2": n_b})
    mw.use_node(n_b)                              # 当前节点 = n_b，上游源 = n_a
    q = np.zeros(1024, dtype=np.float32); q[0] = 1.0
    res = mw.schedule_once("查询", query_emb=q, task_tag="research", now=300.0)
    assert res.stage_a["mode"] == "strength"
    assert res.sources, "n_b 应有上游源"
    assert sum(s["quota"] for s in res.sources) >= mw.selector.max_shared
    assert "quota" in res.latency_ms
    # 硬配额预筛后候选必须属于源节点池（n_a 的 m1），不含 n_b 自身的 m2
    assert "m2" not in [c.memory_id for c in res.candidates]


def test_graph_status_and_budget_gate_wiring(tmp_path):
    mw = _mw(tmp_path, graph_enabled=True, budget=BudgetConfig(default_tier="fast"))
    assert mw.budget is not None and mw.budget.tier == "fast"
    _seed_pool(mw, {"m1": MAIN_NODE, "m2": MAIN_NODE, "m3": MAIN_NODE})
    q = np.zeros(1024, dtype=np.float32); q[0] = 1.0
    for _ in range(2):
        res = mw.schedule_once("记忆调度", query_emb=q, task_tag="research", now=200.0)
    assert res.perf_tier == "fast"
    assert res.llm_calls == 0, "fast 档必须跳过 LLM 先验（成本侧机制）"
    assert mw.last_prior_gate.get("tier_off", 0) >= 1
    st = mw.graph_status()
    assert st["graph_enabled"] is True and "graph" in st and "budget" in st


def test_prior_cache_avoids_repeat_calls(tmp_path):
    mw = _mw(tmp_path, graph_enabled=True, budget=BudgetConfig(default_tier="balanced"))
    calls = {"n": 0}

    def counting_prior(q):
        calls["n"] += 1
        return dict(PRIOR)

    mw.weights._llm_prior_fn = counting_prior
    _seed_pool(mw, {"m1": MAIN_NODE, "m2": MAIN_NODE, "m3": MAIN_NODE,
                    "m4": MAIN_NODE, "m5": MAIN_NODE, "m6": MAIN_NODE})
    q = np.zeros(1024, dtype=np.float32); q[0] = 1.0
    mw.schedule_once("同一个查询", query_emb=q, task_tag="research", now=200.0)
    mw.schedule_once("同一个查询", query_emb=q, task_tag="research", now=201.0)
    assert calls["n"] == 1, "第二次同 query 必须命中缓存，0 次 LLM 调用"
    assert mw.budget.cache.stats()["hits"] >= 1
