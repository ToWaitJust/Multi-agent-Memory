"""主/副线 Agent 工厂（§4.8，FR-7）。

V2.1：headless 骨架 —— 与真实 AgentScope Agent 装配（middlewares 链）在接环境时完成；
本文件提供构建参数规格（workspace per-user、middlewares 顺序）与占位工厂。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentSpec:
    """Agent 装配规格（供真实 AgentScope 装配使用）。"""

    role: str                 # main / sub1 / sub2 / reviewer
    workspace: str            # data/reme/<user>/main 或 data/reme/<user>/<name>
    model_impl: str = "model.deepseek"
    middlewares: list[str] = field(default_factory=lambda: ["tracing", "scheduling", "reme"])
    toolkit: list[str] = field(default_factory=lambda: ["memory_search"])
    session_id: str | None = None   # 实验可恢复：= run_id（O-7）


def build_main_agent(model, config, user_id: str, run_id: str | None = None) -> AgentSpec:
    """主线：workspace=data/reme/<user>/main，middlewares=[Tracing, Scheduling, main ReMe]。"""
    return AgentSpec(
        role="main",
        workspace=f"data/reme/{user_id}/main",
        model_impl=config.model_impl,
        middlewares=["tracing", "scheduling", "reme"],
        toolkit=["memory_search"],
        session_id=run_id or "main_session",
    )


def build_sub_agent(name: str, model, config, user_id: str) -> AgentSpec:
    """副线：workspace=data/reme/<user>/<name>，共享同一 coordinator（由装配器注入）。"""
    return AgentSpec(
        role=name,
        workspace=f"data/reme/{user_id}/{name}",
        model_impl=config.model_impl,
        middlewares=["tracing", "scheduling", "reme"],
        toolkit=[],
        session_id=f"{name}_session",
    )


def _pick_model() -> tuple[Any, str]:
    """复用 examples/reme_demo.py 逻辑：DeepSeek → DashScope → OpenAI；无 key 返回 (None, "")。"""
    # 真实模型选择在接 AgentScope 环境时实现（主模型 DeepSeek，V2.1 定稿）
    return None, ""
