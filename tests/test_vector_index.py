"""向量索引（faiss）单测 —— 锁定 VectorIndex / VectorRetriever 行为（NFR-2、D-11 兜底）。"""
import numpy as np
import pytest

from srtp_memory.attention import MemoryCandidate
from srtp_memory.resident_pool import ResidentMemoryPool
from srtp_memory.retriever import VectorRetriever
from srtp_memory.vector_index import VectorIndex, linear_cosine_rank


def test_index_add_query_returns_nn():
    idx = VectorIndex(dim=4)
    idx.add("m_a", np.array([1.0, 0, 0, 0], dtype=np.float32))
    idx.add("m_b", np.array([0, 1.0, 0, 0], dtype=np.float32))
    idx.add("m_c", np.array([0.9, 0.1, 0, 0], dtype=np.float32))
    ids, sims = idx.query(np.array([1.0, 0, 0, 0], dtype=np.float32), top_k=3)
    assert ids[0] == "m_a"          # 最近邻
    assert "m_c" in ids[:2]         # 第二近（0.9,0.1 vs 正交 m_b）
    assert sims[0] == pytest.approx(1.0, abs=1e-5)


def test_index_remove_and_rebuild():
    idx = VectorIndex(dim=4)
    idx.add("m1", np.array([1.0, 0, 0, 0], dtype=np.float32))
    idx.add("m2", np.array([0, 1.0, 0, 0], dtype=np.float32))
    idx.remove("m1")
    ids, _ = idx.query(np.array([1.0, 0, 0, 0], dtype=np.float32), top_k=5)
    assert "m1" not in ids
    assert idx.size() == 1


def test_index_update_same_id():
    idx = VectorIndex(dim=4)
    idx.add("m1", np.array([1.0, 0, 0, 0], dtype=np.float32))
    idx.add("m1", np.array([0, 1.0, 0, 0], dtype=np.float32))  # 同 id 覆盖
    ids, sims = idx.query(np.array([1.0, 0, 0, 0], dtype=np.float32), top_k=5)
    # 新向量与 query 正交 → 相似度≈0（索引仍返回但分数为 0，旧向量已被覆盖）
    assert len(ids) == 1 and ids[0] == "m1"
    assert sims[0] == pytest.approx(0.0, abs=1e-5)
    assert idx.size() == 1            # 覆盖后只有 1 条，无重复


def test_linear_cosine_rank_fallback():
    class _C:
        def __init__(self, mem_id, emb):
            self.memory_id = mem_id
            self.embedding = emb
            self.vector_score = 0.0
    cands = [_C("a", np.array([1.0, 0], dtype=np.float32)),
             _C("b", np.array([0, 1.0], dtype=np.float32))]
    ranked = linear_cosine_rank(np.array([1.0, 0], dtype=np.float32), cands, top_k=2)
    assert ranked[0].memory_id == "a"
    assert ranked[0].vector_score == pytest.approx(1.0)


def test_vector_retriever_uses_index(tmp_path):
    pool = ResidentMemoryPool("u", "s", tmp_path / "rp.jsonl", with_vector_index=True)
    pool.upsert(MemoryCandidate(memory_id="m1", text="多智能体记忆共享调度", path="p.md",
                                timestamp=0, user_id="u", session_id="s",
                                embedding=np.array([1.0, 0, 0, 0], dtype=np.float32)))
    pool.upsert(MemoryCandidate(memory_id="m2", text="无关内容", path="p2.md",
                                timestamp=0, user_id="u", session_id="s",
                                embedding=np.array([0, 1.0, 0, 0], dtype=np.float32)))
    r = VectorRetriever(index=pool.vector_index)
    got = r.recall("q", pool, top_k=2,
                   query_emb=np.array([1.0, 0, 0, 0], dtype=np.float32))
    assert got[0].memory_id == "m1"
    assert got[0].vector_score == pytest.approx(1.0, abs=1e-5)


def test_vector_retriever_linear_fallback_without_index(tmp_path):
    """faiss 索引未绑定时退化为线性余弦（D-11 兜底），行为一致。"""
    pool = ResidentMemoryPool("u", "s", tmp_path / "rp2.jsonl", with_vector_index=False)
    pool.upsert(MemoryCandidate(memory_id="m1", text="a", path="p.md",
                                timestamp=0, user_id="u", session_id="s",
                                embedding=np.array([1.0, 0], dtype=np.float32)))
    pool.upsert(MemoryCandidate(memory_id="m2", text="b", path="p2.md",
                                timestamp=0, user_id="u", session_id="s",
                                embedding=np.array([0, 1.0], dtype=np.float32)))
    r = VectorRetriever(index=None)
    got = r.recall("q", pool, top_k=2,
                   query_emb=np.array([1.0, 0], dtype=np.float32))
    assert got[0].memory_id == "m1"


def test_resident_pool_rebuilds_index_on_load(tmp_path):
    """持久化后重新加载，索引从 embedding 重建（锁定该行为）。"""
    path = tmp_path / "rp3.jsonl"
    p1 = ResidentMemoryPool("u", "s", path, with_vector_index=True)
    p1.upsert(MemoryCandidate(memory_id="m1", text="a", path="p.md",
                              timestamp=0, user_id="u", session_id="s",
                              embedding=np.array([1.0, 0], dtype=np.float32)))
    p2 = ResidentMemoryPool("u", "s", path, with_vector_index=True)
    assert p2.vector_index is not None
    assert p2.vector_index.size() == 1
