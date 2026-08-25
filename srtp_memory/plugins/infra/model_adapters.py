"""基础设施层插件：模型适配器（ModelAdapter）。

- model.deepseek ：DeepSeek（主 LLM，先验权重 + 回复生成，已接真实 API）
- model.dashscope ：DashScope（备选，占位）
- model.openai    ：OpenAI（备选，已接真实 API）

密钥一律从环境变量读取（.env 由 srtp_memory 包加载），绝不硬编码。
成本控制：先验场景小 max_tokens=64 + thinking=False 省钱；回答场景大 max_tokens + thinking=True。

可观测性：API 调用失败记录 self.last_error = {kind, code, message, ts}
（复用 embedding_impls.classify_api_error 分类），再原样抛出由上层降级捕获——
"余额不足/鉴权失败"不再静默，中间件 api_status() 可暴露。
"""
from __future__ import annotations

import os
import time

from ..base import ModelAdapter
from ..registry import register
from .embedding_impls import classify_api_error


class _BaseModelAdapter(ModelAdapter):
    """适配器基座：未配置 key 时 complete 抛 RuntimeError（由上层降级捕获）。"""

    # 子类指定从哪个环境变量读取密钥（.env 由 srtp_memory 包加载）
    env_var: str = "DASHSCOPE_API_KEY"
    base_url: str | None = None

    def __init__(self, api_key: str | None = None, model_name: str | None = None):
        # api_key 未显式传入时，从环境变量兜底读取（绝不硬编码）
        self.api_key = api_key or os.environ.get(self.env_var)
        self.model_name = model_name or self._default_model()
        # 最近一次 API 错误（无错误为 None）——供中间件 api_status() 暴露
        self.last_error: dict | None = None

    def _record_api_error(self, code, message) -> None:
        self.last_error = {
            "kind": classify_api_error(code, message),
            "code": str(code), "message": str(message), "ts": time.time(),
        }

    def _default_model(self) -> str:
        raise NotImplementedError

    def complete(self, prompt: str, max_tokens: int = 256, temperature: float = 0.3,
                 thinking: bool | None = None, **kw) -> str:
        """真实 API 调用；无 key 抛 RuntimeError（上层可捕获降级）。

        thinking=None 时按场景自动判断：max_tokens<=64 用 False（先验场景省钱），
        >64 用 True（回答场景需推理质量）。显式传参则用传入值。
        """
        if not self.api_key:
            raise RuntimeError(f"{self.name()} 未配置 api_key，无法调用模型")
        if thinking is None:
            thinking = max_tokens > 64
        from openai import OpenAI
        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        # thinking_enable 按场景控制：先验省钱关掉，回答要质量开
        kwargs["extra_body"] = {"thinking_enable": thinking}
        try:
            resp = OpenAI(api_key=self.api_key, base_url=self.base_url).chat.completions.create(
                **kwargs, **kw,
            )
        except Exception as e:  # noqa: BLE001 - 记录错误后原样抛出（上层降级捕获）
            code = getattr(e, "status_code", None) or getattr(e, "code", "") or type(e).__name__
            msg = getattr(e, "message", None) or str(e)
            self._record_api_error(code, msg)
            raise
        return resp.choices[0].message.content or ""

    def complete_stream(self, prompt: str, max_tokens: int = 512, temperature: float = 0.4,
                        **kw) -> str:
        """流式调用（用于界面逐字输出）：yield 文本片段。

        使用 OpenAI SDK 的 stream=True，逐 chunk 取 delta.content。
        thinking 按场景自动开启（max_tokens>64 → True）。
        """
        if not self.api_key:
            raise RuntimeError(f"{self.name()} 未配置 api_key，无法调用模型")
        from openai import OpenAI
        thinking = max_tokens > 64
        stream = OpenAI(api_key=self.api_key, base_url=self.base_url).chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={"thinking_enable": thinking},
            stream=True,
            **kw,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def name(self) -> str:
        return self._plugin_name  # type: ignore[attr-defined]


@register("model.deepseek", "model")
class DeepSeekAdapter(_BaseModelAdapter):
    env_var = "DEEPSEEK_API_KEY"
    base_url = "https://api.deepseek.com"

    def _default_model(self) -> str:
        return "deepseek-v4-flash"


@register("model.dashscope", "model")
class DashScopeModelAdapter(_BaseModelAdapter):
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def _default_model(self) -> str:
        return "qwen-plus"


@register("model.openai", "model")
class OpenAIAdapter(_BaseModelAdapter):
    env_var = "OPENAI_API_KEY"
    base_url = "https://api.openai.com/v1"

    def _default_model(self) -> str:
        return "gpt-4o-mini"
