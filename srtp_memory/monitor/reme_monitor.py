"""L2 ReMe Job 监控（§4.9.1，FR-9）。包装 _run_job：入参/命中数/耗时 → reme_jobs.jsonl。"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path


class ReMeMonitor:
    def __init__(self, path: str | Path = "data/metrics/reme_jobs.jsonl",
                 buffer_size: int = 50):
        self.path = Path(path)
        self.buffer_size = buffer_size
        self._buffer: list[str] = []

    def record_job(self, name: str, kwargs: dict, result, duration_ms: float) -> None:
        row = {
            "event": "reme_job",
            "job": name,
            "timestamp": __import__("time").time(),
            "duration_ms": duration_ms,
            "params": {k: (str(v)[:200] if v else v) for k, v in (kwargs or {}).items()},
            "result_summary": self._summarize(result),
        }
        self._buffer.append(json.dumps(row, ensure_ascii=False))
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    @staticmethod
    def _summarize(result) -> str:
        if result is None:
            return ""
        if isinstance(result, (list, tuple)):
            return f"list[{len(result)}]"
        if isinstance(result, dict):
            return ",".join(f"{k}={len(v) if isinstance(v,(list,dict)) else v}" for k, v in list(result.items())[:5])
        return str(result)[:200]

    def flush(self) -> None:
        if not self._buffer:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("\n".join(self._buffer) + "\n")
        self._buffer.clear()

    async def aflush(self) -> None:
        await asyncio.to_thread(self.flush)
