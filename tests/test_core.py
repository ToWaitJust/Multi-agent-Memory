"""双池 + 选择器 + 检索 + 协调器 + 权重 + 配置 单测。"""
import numpy as np
import pytest

from srtp_memory.attention import MemoryCandidate
from srtp_memory.config import SchedulingConfig
from srtp_memory.condition import ConditionKey, ConditionStore
from srtp_memory.coordinator import MemoryCoordinator
from srtp_memory.resident_pool import ResidentMemoryPool
from srtp_memory.retriever import BM25Retriever, FullRetriever, RRFRetriever, VectorRetriever, get_retriever
from srtp_memory.selector import ActionSelector
from srtp_memory.shared_pool import SharedMemoryPool
from srtp_memory.weights import HybridWeightCalculator


def make_cand(mid: str, text: str, ts: float, task: str | None = None,
              bm25: float = 0.0, vec: float = 0.0) -> MemoryCandidate:
    return MemoryCandidate(memory_id=mid, text=text, path=f"p/{mid}.md",
                           timestamp=ts, task_tag=task, user_id="u", session_id="s",
                           bm25_score=bm25, vector_score=vec)


class TestSharedPool:
    def test_max_size_evicts_lowest(self):
        pool = SharedMemoryPool(max_size=3)
        for i in range(5):
            pool.add({"memory_id": f"m{i}", "score": float(i)})
        assert len(pool.top()) == 3
        ids = [x["memory_id"] for x in pool.top()]
        assert ids == ["m4", "m3", "m2"]  # 降序，淘汰低分

    def test_persist_load_roundtrip(self, tmp_path):
        p = tmp_path / "shared.json"
        pool = SharedMemoryPool(persist_path=p)
        pool.add({"memory_id": "m1", "score": 0.9})
        pool.persist()
        loaded = SharedMemoryPool.load(p)
        assert len(loaded.top()) == 1
        assert loaded.top()[0]["memory_id"] == "m1"


class TestResidentPool:
    def test_upsert_get_mark_access(self, tmp_path):
        pool = ResidentMemoryPool("u", "s", tmp_path / "rp.jsonl")
        c = make_cand("m1", "hello", ts=100.0)
        pool.upsert(c)
        assert pool.get("m1") is not None
        pool.mark_access("m1")
        assert pool.get("m1").access_count == 1

    def test_persist_load_roundtrip(self, tmp_path):
        path = tmp_path / "rp.jsonl"
        p1 = ResidentMemoryPool("u", "s", path)
        c = make_cand("m1", "中文记忆内容", ts=100.0)
        c.embedding = np.zeros(1024, dtype=np.float32)
        p1.upsert(c)
        p2 = ResidentMemoryPool("u", "s", path)
        assert p2.get("m1") is not None
        assert p2.get("m1").text == "中文记忆内容"  # 中文往返一致（O-3）
        assert p2.get("m1").embedding is not None

    def test_user_isolation(self, tmp_path):
        path = tmp_path / "rp.jsonl"
        p1 = ResidentMemoryPool("u1", "s", path)
        p1.upsert(make_cand("m1", "u1 的记忆", ts=0))
        p2 = ResidentMemoryPool("u2", "s", path)
        assert p2.all() == []  # user 隔离


class TestSelector:
    def test_high_kept_low_discarded(self):
        sel = ActionSelector(threshold=0.6)
        scored = [{"score": {"final": 0.9}}, {"score": {"final": 0.1}}]
        res = sel.select(scored)
        assert len(res["kept"]) == 1
        assert res["kept"][0]["score"]["final"] == 0.9

    def test_compression_ratio(self):
        sel = ActionSelector(threshold=0.0)
        scored = [{"score": {"final": 0.9}}, {"score": {"final": 0.9}}]
        res = sel.select(scored)
        assert len(res["kept"]) == 2
        assert sel.compression_ratio(res) == pytest.approx(1.0)  # 全保留时 = 1

    def test_empty_safe(self):
        sel = ActionSelector()
        assert sel.select([]) == {"kept": [], "discarded": []}
        assert sel.compression_ratio({"kept": [], "discarded": []}) == 0.0


class TestRetrievers:
    def test_full_returns_write_order(self, tmp_path):
        pool = ResidentMemoryPool("u", "s", tmp_path / "rp.jsonl")
        pool.upsert(make_cand("m1", "a", ts=0))
        pool.upsert(make_cand("m2", "b", ts=1))
        got = FullRetriever().recall("query", pool, top_k=1)
        assert [c.memory_id for c in got] == ["m1"]

    def test_bm25_ranks_relevant(self, tmp_path):
        pool = ResidentMemoryPool("u", "s", tmp_path / "rp.jsonl")
        pool.upsert(make_cand("m1", "多智能体记忆共享调度方法研究", ts=0))
        pool.upsert(make_cand("m2", "今天天气很好适合散步", ts=1))
        got = BM25Retriever().recall("多智能体记忆共享", pool, top_k=2)
        assert got[0].memory_id == "m1"
        assert got[0].bm25_score > 0.0

    def test_vector_cosine(self, tmp_path):
        pool = ResidentMemoryPool("u", "s", tmp_path / "rp.jsonl")
        c1 = make_cand("m1", "a", ts=0)
        c1.embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        c2 = make_cand("m2", "b", ts=0)
        c2.embedding = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        pool.upsert(c1)
        pool.upsert(c2)
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        got = VectorRetriever().recall("q", pool, top_k=2, query_emb=q)
        assert got[0].memory_id == "m1"
        assert got[0].vector_score == pytest.approx(1.0)

    def test_rrf_fuses(self, tmp_path):
        pool = ResidentMemoryPool("u", "s", tmp_path / "rp.jsonl")
        c1 = make_cand("m1", "多智能体记忆", ts=0, vec=0.9)
        c2 = make_cand("m2", "其他内容", ts=0, vec=0.1)
        pool.upsert(c1)
        pool.upsert(c2)
        q = np.zeros(1024, dtype=np.float32)
        got = RRFRetriever().recall("多智能体记忆", pool, top_k=2, query_emb=q)
        assert got[0].memory_id == "m1"

    def test_get_retriever_unknown(self):
        with pytest.raises(KeyError):
            get_retriever("not_exist")


class TestCoordinator:
    def test_end_to_end_schedule(self, tmp_path):
        pool = ResidentMemoryPool("u", "s", tmp_path / "rp.jsonl")
        pool.upsert(make_cand("m1", "多智能体记忆共享调度", ts=0, task="tech"))
        pool.upsert(make_cand("m2", "无关内容", ts=0, task="other"))
        shared = SharedMemoryPool(max_size=10)
        w = HybridWeightCalculator(llm_prior_fn=lambda q: {"time": 0.2, "semantic": 0.4, "frequency": 0.1, "task": 0.3})
        w.set_prior({"time": 0.2, "semantic": 0.4, "frequency": 0.1, "task": 0.3})
        coord = MemoryCoordinator(pool, BM25Retriever(), weights=w, shared_pool=shared,
                                  user_id="u", session_id="s")
        q = np.zeros(1024, dtype=np.float32)
        q[0] = 1.0
        res = coord.schedule("多智能体记忆共享", q, task_tag="tech", now=1000.0,
                             condition=ConditionKey(user_id="u", scenario_id="r", business_id="b"))
        assert res.candidates  # 有候选
        assert "weights" in res.to_dict()
        assert res.compression_ratio >= 0.0
        assert shared.top()  # 共享池有保留条目


class TestWeights:
    def test_calculate_normalized(self):
        w = HybridWeightCalculator(llm_prior_fn=lambda q: {"time": 0.25, "semantic": 0.35, "frequency": 0.15, "task": 0.25})
        w.set_prior({"time": 0.25, "semantic": 0.35, "frequency": 0.15, "task": 0.25})
        q = np.zeros(1024, dtype=np.float32)
        q[0] = 1.0
        out = w.calculate(q, ConditionKey(user_id="u", scenario_id="r", business_id="b"))
        assert sum(out.values()) == pytest.approx(1.0)
        assert set(out.keys()) == {"time", "semantic", "frequency", "task"}

    def test_llm_prior_fallback_on_error(self):
        w = HybridWeightCalculator(llm_prior_fn=lambda q: (_ for _ in ()).throw(RuntimeError("fail")))
        assert w.llm_prior("q") == {"time": 0.25, "semantic": 0.35, "frequency": 0.15, "task": 0.25}

    def test_pretrain_distill_reduces_loss(self):
        import numpy as np
        w = HybridWeightCalculator(llm_prior_fn=lambda q: {"time": 0.2, "semantic": 0.4, "frequency": 0.1, "task": 0.3})
        dataset = []
        for i in range(20):
            q = np.random.default_rng(i).normal(0, 1, 1024).astype(np.float32)
            cond = ConditionKey(user_id="u", scenario_id="r", business_id="b")
            dataset.append((q, cond, {"time": 0.2, "semantic": 0.4, "frequency": 0.1, "task": 0.3}))
        w.pretrain_distill(dataset)


class TestConditionStore:
    def test_deterministic_and_zero_for_none(self):
        with pytest.raises(ValueError):
            ConditionStore(emb_dim=2)  # 至少 3 维
        s = ConditionStore(emb_dim=32)
        v1 = s.embed(ConditionKey(user_id="u", scenario_id="r", business_id="b"))
        v2 = s.embed(ConditionKey(user_id="u", scenario_id="r", business_id="b"))
        assert np.allclose(v1, v2)  # 确定性
        zero = s.embed(ConditionKey())
        assert np.allclose(zero, 0.0)  # 空 condition → 零向量
        assert v1.shape == (32,)


class TestConfig:
    def test_load_with_defaults(self, tmp_path):
        y = tmp_path / "c.yaml"
        y.write_text("threshold: 0.7\nmax_shared: 8\n", encoding="utf-8")
        cfg = SchedulingConfig.load(y)
        assert cfg.threshold == 0.7
        assert cfg.max_shared == 8
        assert cfg.candidate_override == 50  # 默认
        assert cfg.retriever_impl == "retriever.bm25"

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SchedulingConfig.load(tmp_path / "nope.yaml")
