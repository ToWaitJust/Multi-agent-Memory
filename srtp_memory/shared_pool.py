"""共享记忆池（§4.5，FR-6/FR-7）。

调度层精选的 ≤10 条"已排序高相关集"，持久化到副线 workspace（data/reme/<user>/sub*/shared_pool.json）。
V2.1：无 query() —— 副线吃全量精选集（top() 全给），不二次检索（保持"排序/选择为唯一变量"消融常量）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SharedMemoryPool:
    def __init__(self, max_size: int = 10, persist_path: str | Path | None = None):
        """persist_path 默认 data/reme/<user>/sub<N>/shared_pool.json（由调用方传入）。"""
        self.max_size = max_size
        self.persist_path = Path(persist_path) if persist_path else None
        self._items: list[dict[str, Any]] = []  # [{memory_id, text, path, start_line, end_line, score, timestamp}]

    def add(self, item: dict[str, Any]) -> bool:
        """按分数插入有序列表（降序）；超过 max_size 丢弃最低分；返回是否进入。"""
        score = float(item.get("score", 0.0))
        if self._items and score <= self._items[-1].get("score", 0.0) and len(self._items) >= self.max_size:
            return False
        self._items.append(item)
        self._items.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        if len(self._items) > self.max_size:
            self._items = self._items[: self.max_size]
        return item in self._items

    def top(self) -> list[dict[str, Any]]:
        """返回当前保留的全部条目（≤max_size），供副线中间件注入 HintBlock。"""
        return list(self._items)

    def clear(self) -> None:
        """清空池（会话重置 / 主副线切换时调用）。"""
        self._items.clear()

    def persist(self) -> None:
        """JSON 序列化到 persist_path，支持跨进程恢复。"""
        if self.persist_path is None:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.persist_path.write_text(
            json.dumps({"items": self._items}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "SharedMemoryPool":
        """从 JSON 恢复共享池。文件不存在或损坏时返回空池（不抛异常，best-effort）。"""
        p = Path(path)
        pool = cls(max_size=10, persist_path=p)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            pool._items = data.get("items", [])
            pool.max_size = data.get("max_size", 10)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        return pool
