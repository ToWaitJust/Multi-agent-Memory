"""薄注册表（§12.4）。@register + PLUGINS 字典 + get_plugin，不引入微内核。"""
from __future__ import annotations

from typing import Any, Callable

PLUGINS: dict[str, type] = {}


def register(name: str, kind: str) -> Callable[[type], type]:
    """@register("attention.full", "attention") —— name 全局唯一，kind 用于按层列举。"""
    def deco(cls: type) -> type:
        if name in PLUGINS:
            raise ValueError(f"插件名重复注册: {name}")
        PLUGINS[name] = cls
        setattr(cls, "_plugin_kind", kind)
        setattr(cls, "_plugin_name", name)
        return cls
    return deco


def get_plugin(name: str, *args, **kw) -> Any:
    """按 config 选实现：get_plugin(cfg.attention_impl)(...)。"""
    if name not in PLUGINS:
        raise KeyError(f"未注册插件: {name}（已注册: {sorted(PLUGINS)}）")
    return PLUGINS[name](*args, **kw)


def list_plugins(kind: str | None = None) -> list[str]:
    return [n for n, c in PLUGINS.items() if kind is None or getattr(c, "_plugin_kind", None) == kind]
