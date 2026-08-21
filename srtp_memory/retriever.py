"""检索基座（§12.3 / §13.2，D-12）。

在常驻完整池上召回候选，返回 MemoryCandidate 列表。4 个中性基座原语（自实现）：
  - retriever.full   ：全量传输，按写入序取前 top_k
  - retriever.bm25   ：关键词召回（BM25 打分），取原始分
  - retriever.vector ：1024 维余弦召回
  - retriever.rrf    ：BM25 + 向量双路名次融合（自实现）
参考实现（仅 baseline-reme 使用，wrap ReMe 原生 search，内部行为=ReMe 完整通路，V2.1）：
  - retriever.reme   ：委托 ReMe 原生 search（含 RRF+0.7/0.3），本骨架仅占位，接真实 ReMe 时实现
彻底禁用 wikilink 展开与 min_score 过滤（D-12）。
"""
from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .resident_pool import ResidentMemoryPool

STOPWORDS = {
    "的", "了", "是", "在", "和", "有", "我", "你", "他", "她", "它",
    "与", "就", "都", "而", "及", "着", "或", "一个", "这个", "那个",
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
    "and", "or", "for", "on", "with", "at", "by", "from", "as",
}


def _tokenize(text: str) -> list[str]:
    """极简中文/英文分词：中文按字符（可被 jieba 替换增强），英文按单词。"""
    text = text.lower()
    tokens = re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", text)
    return [t for t in tokens if t not in STOPWORDS]


def _idf(tokens_by_doc: list[list[str]]) -> dict[str, float]:
    n = max(len(tokens_by_doc), 1)
    df: dict[str, int] = {}
    for toks in tokens_by_doc:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    return {t: math.log(1 + n / (1 + c)) for t, c in df.items()}


def _bm25_score(query_tokens: list[str], doc_tokens: list[str], idf: dict[str, float], avg_len: float) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    k1, b = 1.5, 0.75
    doc_len = len(doc_tokens)
    tf: dict[str, int] = {}
    for t in doc_tokens:
        tf[t] = tf.get(t, 0) + 1
    score = 0.0
    for q in set(query_tokens):
        if q in tf:
            f = tf[q]
            score += idf.get(q, 0.0) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * doc_len / max(avg_len, 1.0)))
    return float(score)


class BaseRetriever(ABC):
    """检索基座契约。在常驻完整池上召回候选，返回 MemoryCandidate 列表。"""

    @abstractmethod
    def recall(self, query: str, pool: "ResidentMemoryPool", top_k: int = 20) -> list:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """full / bm25 / vector / rrf / reme"""
        ...


class FullRetriever(BaseRetriever):
    """retriever.full：全量传输，按写入序取前 top_k（无融合、无排序）。"""

    @property
    def name(self) -> str:
        return "full"

    def recall(self, query: str, pool, top_k: int = 20) -> list:
        return pool.all()[:top_k]


class BM25Retriever(BaseRetriever):
    """retriever.bm25：关键词召回，doc 的 bm25_score 写入 candidate（取原始分，中性基座）。"""

    @property
    def name(self) -> str:
        return "bm25"

    def recall(self, query: str, pool, top_k: int = 20) -> list:
        cands = pool.all()
        if not cands:
            return []
        doc_tokens = [_tokenize(c.text) for c in cands]
        idf = _idf(doc_tokens)
        avg_len = sum(len(t) for t in doc_tokens) / max(len(doc_tokens), 1)
        q_tokens = _tokenize(query)
        scored = []
        for c, toks in zip(cands, doc_tokens):
            c.bm25_score = _bm25_score(q_tokens, toks, idf, avg_len)
            scored.append(c)
        scored.sort(key=lambda c: c.bm25_score, reverse=True)
        return scored[:top_k]


class VectorRetriever(BaseRetriever):
    """retriever.vector：1024 维余弦召回，vector_score 写入 candidate。

    优先使用 faiss ANN 索引（VectorIndex，O(log n)）；索引不可用或向量缺失时
    线性余弦兜底（linear_cosine_rank）。精度：IndexFlatIP 为精确检索，无近似误差。
    """

    def __init__(self, index: "VectorIndex | None" = None):
        self.index = index

    @property
    def name(self) -> str:
        return "vector"

    def recall(self, query: str, pool, top_k: int = 20, query_emb=None) -> list:
        from .vector_index import linear_cosine_rank
        cands = pool.all()
        if not cands:
            return []
        if query_emb is None:
            # 无外部 embedding 时退化为按 vector_score 排序（已有向量分）
            scored = sorted(cands, key=lambda c: c.vector_score, reverse=True)
            return scored[:top_k]
        # 优先 faiss 索引（若已绑定）
        if self.index is not None and self.index.available and self.index.size() > 0:
            mem_ids, sims = self.index.query(query_emb, top_k)
            by_id = {c.memory_id: c for c in cands}
            ranked = []
            for mid, s in zip(mem_ids, sims):
                c = by_id.get(mid)
                if c is not None:
                    c.vector_score = s
                    ranked.append(c)
            return ranked[:top_k]
        # 兜底：线性余弦（faiss 缺失 / 索引为空 / 未绑定）
        return linear_cosine_rank(query_emb, cands, top_k)


class RRFRetriever(BaseRetriever):
    """retriever.rrf：BM25 + 向量双路名次融合（自实现，k=60）。"""

    K = 60.0

    @property
    def name(self) -> str:
        return "rrf"

    def recall(self, query: str, pool, top_k: int = 20, query_emb=None) -> list:
        cands = pool.all()
        if not cands:
            return []
        bm25 = BM25Retriever().recall(query, pool, top_k=len(cands))
        vec = VectorRetriever().recall(query, pool, top_k=len(cands), query_emb=query_emb)
        rrf: dict[str, float] = {}
        for i, c in enumerate(bm25):
            rrf[c.memory_id] = rrf.get(c.memory_id, 0.0) + 1.0 / (self.K + i + 1)
        for i, c in enumerate(vec):
            rrf[c.memory_id] = rrf.get(c.memory_id, 0.0) + 1.0 / (self.K + i + 1)
        ranked = sorted(cands, key=lambda c: rrf.get(c.memory_id, 0.0), reverse=True)
        return ranked[:top_k]


class RemeNativeRetriever(BaseRetriever):
    """retriever.reme：wrap ReMe 原生 search（含 RRF+0.7/0.3），仅 baseline-reme 使用。

    V2.1 定稿：注册为插件但内部行为 = ReMe 原生 search 完整通路，不裁剪不硬凑。
    本骨架阶段为占位：传入 `reme_search_callable`（如 lambda query, limit: ...）延迟绑定真实 ReMe。
    """

    def __init__(self, reme_search_callable=None):
        self._callable = reme_search_callable

    @property
    def name(self) -> str:
        return "reme"

    def recall(self, query: str, pool, top_k: int = 20) -> list:
        if self._callable is None:
            # 未接真实 ReMe：退回全量传输（占位），接真实环境时替换为 reme_search_callable(query, top_k)
            return pool.all()[:top_k]
        return self._callable(query, top_k)


RETRIEVERS: dict[str, type[BaseRetriever]] = {
    "full": FullRetriever,
    "bm25": BM25Retriever,
    "vector": VectorRetriever,
    "rrf": RRFRetriever,
    "reme": RemeNativeRetriever,
}


def get_retriever(name: str, *args, **kw) -> BaseRetriever:
    """按名取检索器；兼容带 "retriever." 前缀与裸名（如 "retriever.bm25" / "bm25"）。"""
    key = name[len("retriever."):] if name.startswith("retriever.") else name
    if key not in RETRIEVERS:
        raise KeyError(f"未注册检索器: {name}（可用: {list(RETRIEVERS)}）")
    return RETRIEVERS[key](*args, **kw)
