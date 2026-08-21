"""L3 调度埋点（§4.9.2，FR-9）。批量缓冲 + 异步落盘，schema 见 §5.2 统一增强版。"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path


class ScheduleLogger:
    def __init__(self, path: str | Path = "data/metrics/schedule.jsonl",
                 buffer_size: int = 20):
        self.path = Path(path)
        self.buffer_size = buffer_size
        self._buffer: list[str] = []

    def log_schedule(self, entry: dict) -> None:
        """写一行 {event:"schedule", ...}；缓冲满批量落盘。"""
        row = {"event": "schedule", **entry}
        self._buffer.append(json.dumps(row, ensure_ascii=False))
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("\n".join(self._buffer) + "\n")
        self._buffer.clear()

    async def aflush(self) -> None:
        """异步落盘（O-8：asyncio.to_thread）。"""
        await asyncio.to_thread(self.flush)
