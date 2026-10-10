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

    # 关思考的参数写法（各后端不兼容，见 _thinking_extra）
    thinking_param: str = "thinking_enable"

    def __init__(self, api_key: str | None = None, model_name: str | None = None):
        # api_key 未显式传入时，从环境变量兜底读取（绝不硬编码）
        self.api_key = api_key or os.environ.get(self.env_var)
        self.model_name = model_name or self._default_model()
        # 最近一次 API 错误（无错误为 None）——供中间件 api_status() 暴露
        self.last_error: dict | None = None

    def _thinking_extra(self, thinking: bool) -> dict:
        """构造"关思考"的请求体。

        ⚠️ 两个后端的参数名完全不同，混用会**静默失效**（D-060实测）：
        - DashScope / qwen系列：`{"thinking_enable": bool}`
        - DeepSeek：              `{"thinking": {"type": "disabled"|"enabled"}}`
          DeepSeek 收到 `thinking_enable` **不报错、直接忽略**，模型照样思考；
          而思考token 会吃光 max_tokens（实测 mt=64 时 reasoning_tokens=64、
          content 为空串）→ 上层拿到空回答还以为"模型没话说"。

        所以必须按后端分别给，不能共用一个 key。
        """
        if self.thinking_param == "thinking":
            return {"thinking": {"type": "enabled" if thinking else "disabled"}}
        return {"thinking_enable": thinking}

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
        # 关思考参数按后端区分（D-060：DeepSeek 与 qwen 不兼容，混用会静默失效）
        kwargs["extra_body"] = self._thinking_extra(thinking)
        try:
            resp = OpenAI(api_key=self.api_key, base_url=self.base_url).chat.completions.create(
                **kwargs, **kw,
            )
        except Exception as e:  # noqa: BLE001 - 记录错误后原样抛出（上层降级捕获）
            code = getattr(e, "status_code", None) or getattr(e, "code", "") or type(e).__name__
            msg = getattr(e, "message", None) or str(e)
            self._record_api_error(code, msg)
            raise
        content = resp.choices[0].message.content or ""
        if not content.strip():
            # 空content 是"思考 token 吃光 max_tokens"的典型症状（D-060）：
            # HTTP 200 但content=""，上层会误当成"模型没话说"而静默降级。
            # 这里显式记录，让 api_status() 能暴露，避免静默失败。
            u = getattr(resp, "usage", None)
            rt = getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", None)
            self._record_api_error(
                "empty_content",
                f"{self.model_name} 返回空 content（reasoning_tokens={rt}, "
                f"max_tokens={max_tokens}, thinking={thinking}）—— "
                f"思考 token 可能吃光了预算",
            )
        return content

    def complete_stream(self, prompt: str, max_tokens: int = 512, temperature: float = 0.4,
                        thinking: bool | None = None, **kw):
        """流式调用（用于界面逐字输出）：yield 文本片段。

        使用 OpenAI SDK 的 stream=True，逐 chunk 取 delta.content。
        `thinking` 为 None 时沿用"max_tokens>64 自动开启"的旧行为；
        显式传 False 可与同场景的 `complete()` 保持一致（**回答正文应当关闭思考**，
        否则模型会先吐一大段 reasoning，界面上看起来像答非所问）。
        """
        if not self.api_key:
            raise RuntimeError(f"{self.name()} 未配置 api_key，无法调用模型")
        from openai import OpenAI
        use_thinking = (max_tokens > 64) if thinking is None else bool(thinking)
        stream = OpenAI(api_key=self.api_key, base_url=self.base_url).chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=self._thinking_extra(use_thinking),
            stream=True,
            **kw,
        )
        got_any = False
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                got_any = True
                yield chunk.choices[0].delta.content
        if not got_any:
            # 整个流一个content 都没给=D-060 的空 content 同因。
            # 前端会一直转圈等 done，所以这里必须留下可观测痕迹。
            self._record_api_error(
                "empty_content",
                f"{self.model_name} 流式返回空（max_tokens={max_tokens}, "
                f"thinking={use_thinking}）—— 思考 token 可能吃光了预算",
            )

    def name(self) -> str:
        return self._plugin_name  # type: ignore[attr-defined]


@register("model.deepseek", "model")
class DeepSeekAdapter(_BaseModelAdapter):
    env_var = "DEEPSEEK_API_KEY"
    base_url = "https://api.deepseek.com"
    # DeepSeek 关思考的参数名与 qwen 不同（D-060）
    thinking_param = "thinking"

    def _default_model(self) -> str:
        # ⚠️ 模型名以官方报错为准（D-060 实测）：该endpoint 仅接受
        #   deepseek-flash / deepseek-v4-pro
        # 「deepseek-v4-flash」「deepseek-v4.1-flash」都会 HTTP 400
        #   "The supported API model names are deepseek-flash, deepseek-v4-pro"
        # 注意 deepseek-flash 默认**开启深度思考**，思考 token 会吃光 max_tokens
        # 导致 content 为空—— 必须配合 thinking={"type":"disabled"}（见 _thinking_extra）。
        return "deepseek-flash"


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
