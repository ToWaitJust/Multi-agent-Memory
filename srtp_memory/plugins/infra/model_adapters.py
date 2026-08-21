"""基础设施层插件：模型适配器（ModelAdapter）。

- model.deepseek / model.dashscope / model.openai ：占位适配（接真实 API key 时实现 complete）
V2.1：主模型 DeepSeek（LLM 先验 + 回复生成）。
"""
from __future__ import annotations

from ..base import ModelAdapter
from ..registry import register


class _BaseModelAdapter(ModelAdapter):
    """适配器基座：未配置 key 时 complete 抛 RuntimeError（由上层降级捕获）。"""

    def __init__(self, api_key: str | None = None, model_name: str | None = None):
        self.api_key = api_key
        self.model_name = model_name or self._default_model()

    def _default_model(self) -> str:
        raise NotImplementedError

    def complete(self, prompt: str, **kw) -> str:
        if not self.api_key:
            raise RuntimeError(f"{self.name()} 未配置 api_key，无法调用模型")
        raise NotImplementedError("接真实 API 时实现")

    def name(self) -> str:
        return self._plugin_name  # type: ignore[attr-defined]


@register("model.deepseek", "model")
class DeepSeekAdapter(_BaseModelAdapter):
    def _default_model(self) -> str:
        return "deepseek-chat"


@register("model.dashscope", "model")
class DashScopeModelAdapter(_BaseModelAdapter):
    def _default_model(self) -> str:
        return "qwen-plus"


@register("model.openai", "model")
class OpenAIAdapter(_BaseModelAdapter):
    def _default_model(self) -> str:
        return "gpt-4o-mini"
