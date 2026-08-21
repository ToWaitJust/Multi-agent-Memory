"""attention 模块单测（§9.1）。"""
import math

import numpy as np
import pytest

from srtp_memory.attention import (
    ALL_DIMS,
    DEFAULT_WEIGHTS,
    LAMBDA_TIME,
    MultiHeadAttentionMemoryScorer,
    MemoryCandidate,
)


def make_mem(**kw) -> MemoryCandidate:
    base = dict(
        memory_id="m1", text="测试记忆", path="daily/2026-08-14/x.md",
        start_line=1, end_line=5, timestamp=0.0, access_count=0,
        task_tag="tech", user_id="u", session_id="s",
        bm25_score=0.0, vector_score=0.0, embedding=None,
    )
    base.update(kw)
    return MemoryCandidate(**base)


class TestTimeScore:
    def test_monotonic_decreasing(self):
        scorer = MultiHeadAttentionMemoryScorer()
        m = make_mem(timestamp=0.0)
        now = 3600.0
        t1 = scorer._time_score(m, now)
        t2 = scorer._time_score(m, now * 2)
        assert t1 > t2

    def test_exact_formula(self):
        scorer = MultiHeadAttentionMemoryScorer()
        m = make_mem(timestamp=0.0)
        assert scorer._time_score(m, 3600.0) == pytest.approx(math.exp(-LAMBDA_TIME))
        assert scorer._time_score(m, 0.0) == pytest.approx(1.0)


class TestTaskScore:
    def test_match_1_0(self):
        scorer = MultiHeadAttentionMemoryScorer()
        assert scorer._task_score(make_mem(task_tag="tech"), "tech") == 1.0

    def test_mismatch_0_3(self):
        scorer = MultiHeadAttentionMemoryScorer()
        assert scorer._task_score(make_mem(task_tag="tech"), "other") == 0.3


class TestSemanticScore:
    def test_embedding_missing_falls_back_to_bm25(self):
        """D-11：embedding 缺失时 cosine 记 0，只剩 bm25 分量。"""
        scorer = MultiHeadAttentionMemoryScorer()
        m = make_mem(embedding=None, bm25_score=1.0, vector_score=0.0)
        s = scorer._semantic_score(None, m)
        # W_BM25 * 1.0 = 0.2
        assert s == pytest.approx(0.2)

    def test_cosine_with_embedding(self):
        scorer = MultiHeadAttentionMemoryScorer()
        q = np.array([1.0, 0.0], dtype=np.float32)
        m = make_mem(embedding=np.array([1.0, 0.0], dtype=np.float32),
                     bm25_score=0.0, vector_score=0.0)
        s = scorer._semantic_score(q, m)
        assert s == pytest.approx(0.5)  # W_COS * 1.0


class TestScore:
    def test_output_all_dims(self):
        scorer = MultiHeadAttentionMemoryScorer()
        m = make_mem(timestamp=0.0, access_count=3, task_tag="tech", bm25_score=0.5)
        q = np.zeros(1024, dtype=np.float32)
        q[0] = 1.0
        out = scorer.score(q, m, current_time=0.0, task_tag="tech")
        assert set(ALL_DIMS) <= set(out.keys())
        assert "final" in out

    def test_enabled_dims_mask(self):
        """REQ-610：关掉的维权重强制 0。"""
        scorer = MultiHeadAttentionMemoryScorer()
        m = make_mem(timestamp=0.0, access_count=0, task_tag="tech")
        q = np.zeros(1024, dtype=np.float32)
        # 只开 semantic：其余维权重 0 → final = semantic_score
        out = scorer.score(q, m, current_time=0.0, task_tag="tech",
                           enabled_dims={"semantic"})
        assert out["final"] == pytest.approx(out["semantic"])

    def test_default_weights(self):
        assert DEFAULT_WEIGHTS == {"time": 0.25, "semantic": 0.35, "frequency": 0.15, "task": 0.25}
        assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)
