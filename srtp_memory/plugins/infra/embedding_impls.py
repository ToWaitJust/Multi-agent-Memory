"""基础设施层插件：向量后端（EmbeddingBackend，§12.2 归基础设施层）。

- embed.dashscope ：DashScope text-embedding-v4（1024 维，D-5 首选）
- embed.openai    ：OpenAI text-embedding-3-small（备选）
- embed.local     ：本地 BGE（仅离线兜底，D-11）
V2.1：embedding 多层兜底（重试→缓存→风控告警→bm25），完整实现在接真实 API 时落地；
本骨架 encode 返回确定性占位向量（维度正确、值域安全），保证 pipeline 可跑。
"""
from __future__ import annotations

import numpy as np

from ..base import EmbeddingBackend
from ..registry import register


class _BaseEmbedding(EmbeddingBackend):
    def __init__(self, dim: int = 1024, api_key: str | None = None, model_name: str | None = None):
        self._dim = dim
        self.api_key = api_key
        self.model_name = model_name or self._default_model()

    def _default_model(self) -> str:
        raise NotImplementedError

    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        # 占位实现：确定性哈希向量（接真实 API 时替换为模型调用）
        if not text:
            return np.zeros(self._dim, dtype=np.float32)
        import hashlib
        h = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(h)
        v = rng.normal(0, 1, self._dim).astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


@register("embed.dashscope", "embed")
class DashScopeEmbedding(_BaseEmbedding):
    def _default_model(self) -> str:
        return "text-embedding-v4"


@register("embed.openai", "embed")
class OpenAIEmbedding(_BaseEmbedding):
    def _default_model(self) -> str:
        return "text-embedding-3-small"


@register("embed.local", "embed")
class LocalBGEEmbedding(_BaseEmbedding):
    def __init__(self, dim: int = 1024, **kw):
        super().__init__(dim=dim, **kw)

    def _default_model(self) -> str:
        return "bge-large-zh"
