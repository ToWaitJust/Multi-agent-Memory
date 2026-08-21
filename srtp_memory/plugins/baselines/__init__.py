"""基准/桩实现统一机制（§12.6，baselines）。

关键洞察：基准不是另一类插件，而是同一套插槽的桩/参考实现。
- naive_pipeline：内部桩（NoOp + Uniform + TopK 串成退化管线）
- reme_pipeline ：外部委托（wrap ReMe 原生 search，仅 baseline-reme）
"""
from __future__ import annotations

from . import naive_pipeline, reme_pipeline  # noqa: F401  确保注册执行

__all__ = ["naive_pipeline", "reme_pipeline"]
