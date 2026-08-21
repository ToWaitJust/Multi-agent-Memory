"""baseline-naive 内部桩管线：NoOp + Uniform + TopK 串成一条退化管线（§12.6）。

不注册新插件（复用已注册桩），提供便捷组装函数。
"""
from __future__ import annotations

from ...selector import ActionSelector  # noqa: F401


def build_naive_pipeline():
    """返回退化管线所需的插件实例（attention.noop / weight.uniform / selector.topk）。"""
    from ..algorithms.attention_impls import NoOpAttention
    from ..algorithms.weight_impls import UniformWeight
    from ..algorithms.selector_impls import TopKSelector
    return {
        "attention": NoOpAttention(),
        "weight": UniformWeight(),
        "selector": TopKSelector(top_k=5),
    }
