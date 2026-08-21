"""插件注册表 + 桩/参考实现单测（§12.4 / §12.6）。"""
import pytest

from srtp_memory.plugins import get_plugin, list_plugins, register, register_all


def test_register_all_populates():
    register_all()
    names = sorted(list_plugins())
    # 关键插件必须存在（消融 5 组依赖）
    for required in ["attention.full", "attention.noop",
                     "weight.full", "weight.uniform", "weight.reme",
                     "selector.full", "selector.topk", "selector.recency",
                     "reward.explicit", "reward.none",
                     "retriever.full", "retriever.bm25", "retriever.vector",
                     "retriever.rrf", "retriever.reme",
                     "model.deepseek", "embed.dashscope", "factory.role"]:
        assert required in names, f"缺少插件: {required}"


def test_list_by_kind():
    register_all()
    atts = list_plugins("attention")
    assert "attention.full" in atts
    assert "attention.noop" in atts
    rewards = list_plugins("reward")
    assert "reward.none" in rewards


def test_duplicate_register_raises():
    with pytest.raises(ValueError):
        @register("dupe.test", "test")
        class _A:
            pass

        @register("dupe.test", "test")
        class _B:
            pass


def test_get_plugin_unknown():
    with pytest.raises(KeyError):
        get_plugin("not.exists")


def test_noop_attention():
    a = get_plugin("attention.noop")
    out = a.score(None, None, 0.0, None)
    assert out == {"final": 0.25, "time": 0.25, "semantic": 0.25,
                   "frequency": 0.25, "task": 0.25}
    assert a.names() == ["time", "semantic", "frequency", "task"]


def test_uniform_weight():
    w = get_plugin("weight.uniform")
    out = w.calculate(None, None)
    assert out == {"time": 0.25, "semantic": 0.35, "frequency": 0.15, "task": 0.25}


def test_reme_native_weight():
    w = get_plugin("weight.reme")
    out = w.calculate(None, None)
    assert out["semantic"] == 0.7  # ReMe vector_weight=0.7


def test_topk_selector():
    s = get_plugin("selector.topk", top_k=2)
    scored = [
        {"score": {"final": 0.3}}, {"score": {"final": 0.9}}, {"score": {"final": 0.6}},
    ]
    res = s.select(scored)
    finals = sorted(m["score"]["final"] for m in res["kept"])
    assert finals == [0.6, 0.9]
    assert len(res["discarded"]) == 1


def test_rewards():
    exp = get_plugin("reward.explicit")
    assert exp.from_feedback(True) == 1.0
    assert exp.from_feedback(False) == -1.0
    none = get_plugin("reward.none")
    assert none.from_feedback(True) == 0.0


def test_retriever_plugins(tmp_path):
    from srtp_memory.attention import MemoryCandidate
    from srtp_memory.resident_pool import ResidentMemoryPool
    pool = ResidentMemoryPool("u", "s", tmp_path / "rp.jsonl")
    pool.upsert(MemoryCandidate(memory_id="m1", text="多智能体记忆共享调度", path="p.md",
                                timestamp=0, user_id="u", session_id="s"))
    r = get_plugin("retriever.bm25")
    got = r.recall("多智能体记忆", pool, top_k=5)
    assert got and got[0].memory_id == "m1"
    # reme 未绑真实 ReMe 时退化为全量
    rr = get_plugin("retriever.reme")
    assert len(rr.recall("q", pool, top_k=5)) == 1
