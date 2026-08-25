"""基础设施层插件：向量后端（EmbeddingBackend，§12.2 归基础设施层）。

- embed.dashscope ：DashScope text-embedding-v4（1024 维，D-5 首选，已接真实 API）
- embed.openai    ：OpenAI text-embedding-3-small（备选，已接真实 API）
- embed.local     ：本地 BGE（仅离线兜底，D-11，占位实现）

V2.1：embedding 多层兜底（D-11）——真实 API 主链路 → 结果缓存（同文本零调用）→
API 异常时降级为确定性占位向量（保证 pipeline 可跑，语义分退回纯 bm25）。
密钥一律从环境变量读取（.env 由 srtp_memory 包加载），绝不硬编码。

可观测性（新增）：每次 API 失败记录 self.last_error = {kind, code, message, ts}，
供调度中间件 api_status() 暴露给上层（控制台/监控）——降级不再静默。
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

from ..base import EmbeddingBackend
from ..registry import register


def classify_api_error(code, message) -> str:
    """把 API 错误归类为可读类别（供告警提示，如"余额不足"）。"""
    s = f"{code} {message}".lower()
    if ("402" in s or "insufficient balance" in s or "余额" in s
            or "quota" in s or "额度" in s or "overlimit" in s):
        return "balance"
    if ("401" in s or "invalid_api_key" in s or "authentication" in s
            or "鉴权" in s or "apikey" in s):
        return "auth"
    if "403" in s or "forbidden" in s or "permission" in s or "无权" in s:
        return "forbidden"
    if "429" in s or "rate" in s or "limit" in s or "限流" in s or "throttl" in s:
        return "rate_limit"
    if ("timeout" in s or "timed out" in s or "connection" in s
            or "网络" in s or "connect" in s or "resolve" in s):
        return "network"
    return "other"


class _BaseEmbedding(EmbeddingBackend):
    # 子类指定从哪个环境变量读取密钥（.env 由 srtp_memory 包加载）
    env_var: str = "DASHSCOPE_API_KEY"

    def __init__(self, dim: int = 1024, api_key: str | None = None,
                 model_name: str | None = None, cache_dir: str | None = None):
        self._dim = dim
        # api_key 未显式传入时，从环境变量兜底读取（绝不硬编码）
        self.api_key = api_key or os.environ.get(self.env_var)
        self.model_name = model_name or self._default_model()
        self._cache: dict[str, np.ndarray] = {}
        # 落盘缓存（D-11/可复现）：真实向量写盘，重启后复用，避免重复调 API
        self._cache_dir = cache_dir
        self._cache_file = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._cache_file = os.path.join(cache_dir, f"{self.__class__.__name__}.jsonl")
            self._load_disk_cache()
        # 最近一次 API 错误（无错误为 None）——供中间件 api_status() 暴露
        self.last_error: dict | None = None

    def _load_disk_cache(self) -> None:
        """启动时把已有真实向量载入内存（仅真实向量，占位向量不落盘）。"""
        if not self._cache_file or not os.path.exists(self._cache_file):
            return
        with open(self._cache_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    v = np.asarray(rec["vec"], dtype=np.float32)
                    if v.shape[0] == self._dim:
                        self._cache[rec["text"]] = v
                except Exception:  # noqa: BLE001 - 单条损坏不影响整体
                    continue

    def _persist(self, text: str, vec: np.ndarray) -> None:
        """真实向量追加写盘（每次 API 成功调用一次）。"""
        if not self._cache_file:
            return
        with open(self._cache_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"text": text, "vec": vec.tolist()}, ensure_ascii=False) + "\n")

    def _record_api_error(self, code, message) -> None:
        self.last_error = {
            "kind": classify_api_error(code, message),
            "code": str(code), "message": str(message), "ts": time.time(),
        }

    def _default_model(self) -> str:
        raise NotImplementedError

    def dim(self) -> int:
        return self._dim

    def _placeholder_encode(self, text: str) -> np.ndarray:
        """确定性哈希占位向量（D-11 兜底）：同文本恒等、单位范数、值域安全。"""
        import hashlib
        h = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(h)
        v = rng.normal(0, 1, self._dim).astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _api_encode(self, text: str) -> np.ndarray | None:
        """真实 API 调用（子类实现）；失败抛异常 / 无 key 返回 None。"""
        raise NotImplementedError

    def encode(self, text: str) -> np.ndarray:
        """真实 API 优先，缓存命中零调用，失败降级占位向量（D-11）。

        真实向量同时写落盘缓存（cache_dir），重启后复用 → 一次性成本、完全可复现。
        占位向量不落盘，确保日后补 key 时能取到真实向量而非陈旧占位。
        """
        if not text:
            return np.zeros(self._dim, dtype=np.float32)
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        try:
            v = self._api_encode(text)
            if v is not None:
                self._cache[text] = v
                self._persist(text, v)
                return v
        except Exception as e:  # noqa: BLE001 - API 异常按 D-11 降级，不阻塞 pipeline
            self._record_api_error(getattr(e, "code", "") or getattr(e, "status_code", "") or type(e).__name__,
                                   str(e))
        return self._placeholder_encode(text)


@register("embed.dashscope", "embed")
class DashScopeEmbedding(_BaseEmbedding):
    def _default_model(self) -> str:
        return "text-embedding-v4"

    def _api_encode(self, text: str) -> np.ndarray | None:
        if not self.api_key:
            return None
        from dashscope import TextEmbedding
        resp = TextEmbedding.call(model=self.model_name, input=text, api_key=self.api_key)
        if resp.status_code != 200:
            self._record_api_error(resp.status_code, resp.message or resp.code)
            raise RuntimeError(f"dashscope embed {resp.status_code}: {resp.message}")
        v = np.asarray(resp.output["embeddings"][0]["embedding"], dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


@register("embed.openai", "embed")
class OpenAIEmbedding(_BaseEmbedding):
    env_var = "OPENAI_API_KEY"

    def _default_model(self) -> str:
        return "text-embedding-3-small"

    def _api_encode(self, text: str) -> np.ndarray | None:
        if not self.api_key:
            return None
        from openai import OpenAI
        resp = OpenAI(api_key=self.api_key).embeddings.create(
            model=self.model_name, input=text,
        )
        v = np.asarray(resp.data[0].embedding, dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


@register("embed.local", "embed")
class LocalBGEEmbedding(_BaseEmbedding):
    def __init__(self, dim: int = 1024, **kw):
        super().__init__(dim=dim, **kw)

    def _default_model(self) -> str:
        return "bge-large-zh"

    def _api_encode(self, text: str) -> np.ndarray | None:
        # 本地 BGE 仅离线兜底：不接 API，直接走占位向量
        return None
