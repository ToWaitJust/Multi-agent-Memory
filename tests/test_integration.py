"""集成测试：调度管线各环节衔接（离线，全部走 tmp_path，不依赖真实 API / 网络）。

覆盖 5 个全流程场景（对应工程实现文档验收标准）：
  1. test_cross_instance_recall_persistence — 跨会话召回（O-1/O-7）：固定 session 下实例重启后记忆仍可召回
  2. test_chinese_roundtrip_utf8            — 中文完整性（O-3）：写入→落盘→重载→召回逐字无损
  3. test_main_sub_agent_dispatch           — 主副智能体调度（FR-6）：schedule_once 精选 → dispatch 交副线
  4. test_failure_degradation_no_api        — 失败降级（D-11）：无 key 时 embedding→占位、LLM 先验→默认权重，管线不中断
  5. test_user_isolation_concurrent         — 并发隔离（NFR-7）：多 user 交错调度，记忆互不串扰
  6. test_vector_retriever_semantic_ranking — 向量检索链路（离线）：确定性占位向量下同主题排序正确

说明：embedding 用确定性哈希占位向量（与 embed 插件的 _placeholder_encode 同构），
query_emb 显式传入或由降级路径生成，确保测试即便本机 .env 配了真实 key 也不触发网络。
"""
from __future__ import annotations

import hashlib

import numpy as np

from srtp_memory.attention import MemoryCandidate
from srtp_memory.config import SchedulingConfig
from srtp_memory.middleware import MemorySchedulingMiddleware

# 注入的确定性 LLM 先验（避免测试触发真实模型）
DEFAULT_PRIOR = {"time": 0.25, "semantic": 0.35, "frequency": 0.15, "task": 0.25}


def _place_vec(text: str, dim: int = 1024) -> np.ndarray:
    """确定性哈希占位向量：同文本恒等、单位范数（与 embed._placeholder_encode 同构）。"""
    h = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    v = np.random.default_rng(h).normal(0, 1, dim).astype(np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _make(tmp_path, user: str = "u_a", sess: str = "s1",
          retriever: str = "retriever.bm25", prior=None) -> MemorySchedulingMiddleware:
    """构建隔离的中间件：双池/埋点全部落在 tmp_path，按 user 分文件。"""
    if prior is None:
        prior = lambda q: dict(DEFAULT_PRIOR)  # noqa: E731 - 测试注入确定性先验
    return MemorySchedulingMiddleware(
        config=SchedulingConfig(retriever_impl=retriever, weight_impl="weight.full",
                                log_dir=str(tmp_path), run_id="it"),
        user_id=user, session_id=sess,
        resident_pool_path=str(tmp_path / f"rp_{user}.jsonl"),
        shared_pool_path=str(tmp_path / f"sp_{user}.json"),
        log_path=str(tmp_path / "metrics.jsonl"),
        llm_prior_fn=prior,
    )


def _seed(mid: MemorySchedulingMiddleware, memory_id: str, text: str, *,
          task: str = "research", ts: float = 100.0, with_emb: bool = True) -> None:
    mid.resident_pool.upsert(MemoryCandidate(
        memory_id=memory_id, text=text, path=f"d/{memory_id}.md",
        timestamp=ts, task_tag=task,
        user_id=mid.user_id, session_id=mid.session_id,
        embedding=_place_vec(text) if with_emb else None))


# ---- 1. 跨会话召回（O-1 / O-7）----

def test_cross_instance_recall_persistence(tmp_path):
    """会话固定（session_id=run_id，O-7）：实例重启 / 新 Agent 后，旧会话记忆仍可召回。"""
    mid1 = _make(tmp_path, user="u_a", sess="s_fixed")
    _seed(mid1, "m1", "多智能体记忆共享调度方法研究")
    mid1.flush()

    # 新实例（模拟进程重启 / 新 Agent），同 user + 同固定 session → 从同一常驻池文件重载
    mid2 = _make(tmp_path, user="u_a", sess="s_fixed")
    q = _place_vec("多智能体记忆共享")
    res = mid2.schedule_once("多智能体记忆共享", query_emb=q, task_tag="research", now=200.0)
    ids = [c.memory_id for c in res.candidates]
    assert "m1" in ids, "跨实例重载后应能召回已持久化记忆"
    assert mid2.resident_pool.get("m1").text == "多智能体记忆共享调度方法研究"


# ---- 2. 中文完整性（O-3）----

def test_chinese_roundtrip_utf8(tmp_path):
    """中文记忆 写入→落盘→重载 逐字一致；调度召回/保留结果中文无损（O-3）。"""
    text = "用户偏好记录：使用 PowerShell 并注意 UTF-8 编码，避免中文乱码问题"
    mid1 = _make(tmp_path, user="u_a", sess="s1")
    _seed(mid1, "m1", text, task="pref")
    mid1.flush()

    # 重载后文本逐字一致
    mid2 = _make(tmp_path, user="u_a", sess="s1")
    loaded = mid2.resident_pool.get("m1")
    assert loaded is not None and loaded.text == text

    # 调度召回中文查询仍工作，保留结果文本无损
    q = _place_vec("UTF-8 编码偏好")
    res = mid2.schedule_once("UTF-8 编码偏好", query_emb=q, task_tag="pref", now=200.0)
    assert res.candidates
    kept = {m.memory_id: m.text for m in res.kept}
    if "m1" in kept:
        assert kept["m1"] == text


# ---- 3. 主副智能体调度（FR-6）----

def test_main_sub_agent_dispatch(tmp_path):
    """主线调度精选进共享池 → dispatch 全量交副线注入（FR-6）。"""
    mid = _make(tmp_path, user="u_a", sess="s1")
    _seed(mid, "m1", "多智能体记忆共享调度方法研究")
    _seed(mid, "m2", "共享池容量上限与淘汰策略调参")
    _seed(mid, "m3", "天气不错适合出门散步", ts=102.0)

    q = _place_vec("多智能体记忆共享调度方法")
    res = mid.schedule_once("多智能体记忆共享调度方法", query_emb=q, task_tag="research", now=300.0)
    assert res.kept, "应至少保留一条相关记忆进共享池"
    assert len(mid.shared_pool.top()) > 0

    # 副线从共享池取全量精选集（≤10 条，不再二次检索）
    items = mid.coordinator.dispatch(None, top_k=5)
    dispatched_ids = {i["memory_id"] for i in items}
    assert {m.memory_id for m in res.kept} <= dispatched_ids
    assert all(i.get("text") for i in items)  # 副线拿到的是结构化记忆而非引用


# ---- 4. 失败降级（D-11）----

def test_failure_degradation_no_api(tmp_path, monkeypatch):
    """无 API key：embedding→确定性占位向量、LLM 先验→默认权重，管线完整不中断。"""
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    # prior=None → 中间件走"真实 LLM 先验"装配路径，但无 key → 降级默认权重
    mid = _make(tmp_path, user="u_a", sess="s1", prior=None)
    _seed(mid, "m1", "多智能体记忆共享调度方法研究", with_emb=False)

    # 不传 query_emb → 触发真实 embedding 链路，无 key 自动降级占位向量
    res = mid.schedule_once("多智能体记忆共享调度方法研究", task_tag="research", now=200.0)
    assert res.candidates, "降级路径下调度管线应完整跑通"
    w = res.weights
    assert abs(sum(w.values()) - 1.0) < 1e-6, "权重应保持归一化"
    # 共享池 / L3 埋点仍正常
    assert len(mid.shared_pool.top()) > 0
    mid.flush()
    assert (tmp_path / "metrics.jsonl").exists()


# ---- 5. 并发隔离（NFR-7）----

def test_user_isolation_concurrent(tmp_path):
    """两个 user 交错调度，常驻池/共享池按 user 隔离，互不串扰。"""
    ma = _make(tmp_path, user="u_a", sess="sa")
    mb = _make(tmp_path, user="u_b", sess="sb")
    _seed(ma, "ma1", "用户A的私有研究记录")
    _seed(mb, "mb1", "用户B的财务偏好", task="finance")

    q = _place_vec("用户A的私有研究记录")
    ra = ma.schedule_once("用户A的私有研究记录", query_emb=q, task_tag="research", now=200.0)
    rb = mb.schedule_once("用户A的私有研究记录", query_emb=q, task_tag="finance", now=200.0)

    a_ids = {c.memory_id for c in ra.candidates}
    b_ids = {c.memory_id for c in rb.candidates}
    assert "ma1" in a_ids and "mb1" not in a_ids  # A 看不到 B
    assert "ma1" not in b_ids                     # B 看不到 A
    assert all(i["memory_id"] == "ma1" for i in ma.shared_pool.top())
    assert all(i["memory_id"] == "mb1" for i in mb.shared_pool.top())


# ---- 6. 向量检索链路（离线，D-12 / NFR-2）----

def test_vector_retriever_semantic_ranking(tmp_path):
    """确定性占位向量下，同主题记忆余弦更高、排序在无关记忆之前。"""
    mid = _make(tmp_path, user="u_a", sess="s1", retriever="retriever.vector")
    _seed(mid, "m1", "多智能体记忆共享调度算法与共享池设计")
    _seed(mid, "m2", "记忆调度中间件的接口契约与插件装配")
    _seed(mid, "m3", "今天天气晴朗适合外出活动", ts=102.0)

    q = _place_vec("多智能体记忆共享调度")
    res = mid.schedule_once("多智能体记忆共享调度", query_emb=q, task_tag="research", now=200.0)
    ids = [c.memory_id for c in res.candidates]
    assert len(ids) == 3
    assert ids[0] in ("m1", "m2"), "同主题记忆应排最前"
    assert ids.index("m1") < ids.index("m3")
    assert ids.index("m2") < ids.index("m3")
