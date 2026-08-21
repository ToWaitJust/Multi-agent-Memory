"""检索基座插件注册（§13.2，D-12）。

把 retriever.py 的 5 个实现注册进统一注册表：
  retriever.full / retriever.bm25 / retriever.vector / retriever.rrf（自研）
  retriever.reme（wrap ReMe 原生，仅 baseline-reme，内部行为=ReMe 完整通路）
"""
from __future__ import annotations

from ....retriever import (
    BM25Retriever,
    FullRetriever,
    RemeNativeRetriever,
    RRFRetriever,
    VectorRetriever,
)
from ...base import BaseRetriever
from ...registry import register


@register("retriever.full", "retriever")
class FullRetrieverPlugin(FullRetriever, BaseRetriever):
    pass


@register("retriever.bm25", "retriever")
class BM25RetrieverPlugin(BM25Retriever, BaseRetriever):
    pass


@register("retriever.vector", "retriever")
class VectorRetrieverPlugin(VectorRetriever, BaseRetriever):
    pass


@register("retriever.rrf", "retriever")
class RRFRetrieverPlugin(RRFRetriever, BaseRetriever):
    pass


@register("retriever.reme", "retriever")
class RemeNativeRetrieverPlugin(RemeNativeRetriever, BaseRetriever):
    pass
