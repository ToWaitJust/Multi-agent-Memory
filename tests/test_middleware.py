"""middleware + 端到端调度单测。"""
import numpy as np

from srtp_memory.attention import MemoryCandidate
from srtp_memory.config import SchedulingConfig
from srtp_memory.middleware import MemorySchedulingMiddleware
from srtp_memory.resident_pool import ResidentMemoryPool


def test_middleware_schedule_once(tmp_path, monkeypatch):
    # 注入常驻池种子记忆（绕过 data/ 真实路径）
    mid = MemorySchedulingMiddleware(
        config=SchedulingConfig(retriever_impl="retriever.bm25",
                                log_dir=str(tmp_path), run_id="test_run"),
        user_id="u_tu", session_id="sess_t",
        log_path=str(tmp_path / "schedule.jsonl"),
        llm_prior_fn=lambda q: {"time": 0.25, "semantic": 0.35, "frequency": 0.15, "task": 0.25},
    )
    # 手动种入记忆（真实场景由 ReMe auto_memory → upsert）
    mid.resident_pool.upsert(MemoryCandidate(
        memory_id="m1", text="多智能体记忆共享调度方法研究", path="d/x.md",
        timestamp=0.0, task_tag="tech", user_id="u_tu", session_id="sess_t"))
    mid.resident_pool.upsert(MemoryCandidate(
        memory_id="m2", text="无关内容", path="d/y.md",
        timestamp=0.0, task_tag="other", user_id="u_tu", session_id="sess_t"))

    q = np.zeros(1024, dtype=np.float32)
    q[0] = 1.0
    res = mid.schedule_once("多智能体记忆共享", query_emb=q, task_tag="tech", now=1000.0)
    assert res.candidates
    assert res.kept
    assert len(mid.shared_pool.top()) > 0  # 共享池有注入内容
    mid.flush()
    assert (tmp_path / "schedule.jsonl").exists()  # L3 埋点落盘
