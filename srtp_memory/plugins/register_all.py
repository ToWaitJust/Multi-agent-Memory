"""插件自动注册入口：import 本模块即注册全部插件。

供 middleware/CLI 装配时调用 register_all() 确保注册表齐全。
"""
from __future__ import annotations


def register_all() -> None:
    """导入各实现模块，触发 @register 装饰器执行。幂等。"""
    from .algorithms import (  # noqa: F401
        attention_impls,
        reward_impls,
        selector_impls,
        weight_impls,
    )
    from .algorithms.retrieval import retriever_plugins  # noqa: F401
    from .infra import (  # noqa: F401
        embedding_impls,
        factory_impls,
        model_adapters,
    )
    from . import baselines  # noqa: F401
