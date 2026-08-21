"""基础设施层插件：智能体工厂（AgentFactory，§12.2）。

- factory.role ：按 role（main/sub/reviewer）装配 AgentSpec
V2.1：真实 AgentScope 装配（middlewares 链）在接环境时完成；此处返回 AgentSpec 规格。
"""
from __future__ import annotations

from ...agents import AgentSpec, build_main_agent, build_sub_agent
from ..base import AgentFactory
from ..registry import register


@register("factory.role", "factory")
class RoleFactory(AgentFactory):
    def __init__(self, user_id: str = "u_tu", run_id: str | None = None, **kw):
        self.user_id = user_id
        self.run_id = run_id

    def build(self, role: str, **kw) -> AgentSpec:
        if role == "main":
            return build_main_agent(None, kw.get("config"), self.user_id, self.run_id)
        return build_sub_agent(role, None, kw.get("config"), self.user_id)
