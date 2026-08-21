"""baseline-reme 外部委托管线：wrap ReMe 原生 search（§12.6 / §9.4）。

V2.1：retriever.reme 已注册，内部行为 = ReMe 原生 search 完整通路（含 RRF+0.7/0.3），
不裁剪不硬凑。此文件提供便捷组装（reme_search_callable 延迟绑定真实 ReMe）。
"""
from __future__ import annotations


def build_reme_pipeline(reme_search_callable=None):
    """返回 baseline-reme 的检索器（retriever.reme，可注入真实 ReMe search）。"""
    from ..algorithms.retrieval.retriever_plugins import RemeNativeRetrieverPlugin
    return RemeNativeRetrieverPlugin(reme_search_callable=reme_search_callable)
