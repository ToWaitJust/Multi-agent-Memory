"""plugins 包：薄注册表 + 全部插件实现（§12.4）。

导入本包即注册全部插件（register_all）。供 middleware 装配时直接使用。
"""
from .registry import get_plugin, list_plugins, register  # noqa: F401
from .register_all import register_all  # noqa: F401

register_all()
