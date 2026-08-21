"""常驻完整池（§4.5.1，per-user 真池）。

调度层自维护的 memory 索引/摘要，不每次现查 ReMe（D-9）。每条记录含 memory_id 主键
+ 文本 + 1024 维 embedding + 元数据，(user_id, session_id) 隔离。
V2.1：memory_id(uuid) 为唯一主键；user_id/session_id 为隔离过滤维度；path:line 仅书签辅助。
同步时序：auto_memory 写卡片后 upsert，当轮新记忆下一轮才可检索（"延迟一轮"为既定预期）。
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from .attention import MemoryCandidate


class ResidentMemoryPool:
    def __init__(self, user_id: str, session_id: str, path: str | Path,
                 with_vector_index: bool = True):
        """path 默认 data/reme/<user>/main/resident_pool.jsonl（per-user）。
        with_vector_index=True：维护 faiss 向量索引（upsert 同步），供 VectorRetriever ANN 查询。"""
        self.user_id = user_id
        self.session_id = session_id
        self.path = Path(path)
        self._records: dict[str, MemoryCandidate] = {}
        self.vector_index = None
        if with_vector_index:
            from .vector_index import VectorIndex
            self.vector_index = VectorIndex(dim=1024)
        self._load()
        # 加载后重建索引（持久化的 embedding 重新入索引）
        if self.vector_index is not None:
            for r in self.all():
                if r.embedding is not None:
                    self.vector_index.add(r.memory_id, r.embedding)

    # ---- 写 ----

    def upsert(self, candidate: MemoryCandidate) -> None:
        """按 memory_id 写入/更新一条记忆（含 access_count/task_tag 字段）。"""
        if not candidate.memory_id:
            candidate.memory_id = uuid.uuid4().hex
        existed = candidate.memory_id in self._records
        self._records[candidate.memory_id] = candidate
        # 同步向量索引（新增或更新）
        if self.vector_index is not None and candidate.embedding is not None:
            self.vector_index.add(candidate.memory_id, candidate.embedding)
        elif self.vector_index is not None and existed:
            self.vector_index.remove(candidate.memory_id)
        self.persist()

    def mark_access(self, memory_id: str) -> None:
        """access_count += 1（频率头 / L3 同源）。"""
        rec = self._records.get(memory_id)
        if rec is not None:
            rec.access_count += 1

    # ---- 读 ----

    def get(self, memory_id: str) -> MemoryCandidate | None:
        """按 memory_id 主键精确读取。"""
        return self._records.get(memory_id)

    def all(self) -> list[MemoryCandidate]:
        """全量（retriever.full 用），仅返回当前 (user_id, session_id) 的记录。"""
        return [
            r for r in self._records.values()
            if r.user_id == self.user_id and r.session_id == self.session_id
        ]

    def size(self) -> int:
        return len(self._records)

    # ---- 持久化 ----

    def persist(self) -> None:
        """落盘 resident_pool.jsonl（每行一条 JSON，embedding 存为 list）。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            for rec in self._records.values():
                f.write(json.dumps(_candidate_to_dict(rec), ensure_ascii=False) + "\n")

    def _load(self) -> None:
        """启动时加载已有记录（best-effort，文件缺失/损坏返回空池）。"""
        if not self.path.exists():
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = _dict_to_candidate(json.loads(line))
                    self._records[rec.memory_id] = rec
        except (json.JSONDecodeError, OSError, KeyError):
            # 损坏文件不阻断启动
            pass


def _candidate_to_dict(c: MemoryCandidate) -> dict:
    d = {
        "memory_id": c.memory_id,
        "text": c.text,
        "path": c.path,
        "start_line": c.start_line,
        "end_line": c.end_line,
        "timestamp": c.timestamp,
        "access_count": c.access_count,
        "task_tag": c.task_tag,
        "user_id": c.user_id,
        "session_id": c.session_id,
        "bm25_score": c.bm25_score,
        "vector_score": c.vector_score,
    }
    if c.embedding is not None:
        d["embedding"] = c.embedding.tolist()
    return d


def _dict_to_candidate(d: dict) -> MemoryCandidate:
    emb = None
    if d.get("embedding"):
        emb = __import__("numpy").array(d["embedding"], dtype=float)
    return MemoryCandidate(
        memory_id=d.get("memory_id", uuid.uuid4().hex),
        text=d.get("text", ""),
        path=d.get("path", ""),
        start_line=d.get("start_line", 0),
        end_line=d.get("end_line", 0),
        timestamp=d.get("timestamp", 0.0),
        access_count=d.get("access_count", 0),
        task_tag=d.get("task_tag"),
        user_id=d.get("user_id", ""),
        session_id=d.get("session_id", ""),
        bm25_score=d.get("bm25_score", 0.0),
        vector_score=d.get("vector_score", 0.0),
        embedding=emb,
    )
