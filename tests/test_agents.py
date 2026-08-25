"""agents / factory 插件 / CLI 单测（§4.8 / §12.2）。"""
from srtp_memory.agents import build_main_agent, build_sub_agent
from srtp_memory.config import SchedulingConfig
from srtp_memory.plugins import get_plugin, register_all


def test_agent_specs():
    cfg = SchedulingConfig()
    main = build_main_agent(None, cfg, user_id="u_tu", run_id="run_x")
    assert main.workspace == "data/reme/u_tu/main"
    assert main.session_id == "run_x"          # O-7：会话固定可恢复
    assert "scheduling" in main.middlewares
    sub = build_sub_agent("sub1", None, cfg, user_id="u_tu")
    assert sub.workspace == "data/reme/u_tu/sub1"
    assert sub.role == "sub1"


def test_role_factory_plugin():
    register_all()
    f = get_plugin("factory.role", user_id="u_tu", run_id="run_x")
    main = f.build("main", config=SchedulingConfig())
    assert main.role == "main"
    assert main.workspace == "data/reme/u_tu/main"
    sub = f.build("sub2", config=SchedulingConfig())
    assert sub.workspace == "data/reme/u_tu/sub2"


def test_embedding_plugin_dim():
    register_all()
    e = get_plugin("embed.dashscope")
    assert e.dim() == 1024
    import numpy as np
    v = e.encode("中文测试")
    assert v.shape == (1024,)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-3  # 单位向量


def test_model_adapters_need_key(monkeypatch):
    register_all()
    # 与真实环境隔离：即使本机配了 DEEPSEEK_API_KEY，本用例也按"未配置"走
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    m = get_plugin("model.deepseek")
    assert m.name() == "model.deepseek"
    try:
        m.complete("hi")
        raise AssertionError("未配 key 不应成功")
    except RuntimeError as e:
        assert "api_key" in str(e)
