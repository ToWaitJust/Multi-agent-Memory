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
        with_vector_index=True：维护 faiss 向量索引（upsert 同步），供 VectorRetriever ANN 查询。

        图架构（§2.3）：本池即「大记忆池（EntityPool）」，新增 `node_index`
        （owner_node → memory_id 倒排索引），使"节点专属记忆"= 按标记过滤的零拷贝视图。
        """
        self.user_id = user_id
        self.session_id = session_id
        self.path = Path(path)
        self._records: dict[str, MemoryCandidate] = {}
        self.vector_index = None
        if with_vector_index:
            from .vector_index import VectorIndex
            self.vector_index = VectorIndex(dim=1024)
        # 节点标记倒排索引（可从池重建，不落盘、不新增侧表，保持 D-10）
        from .graph.view import NodeIndex
        self.node_index = NodeIndex()
        self._load()
        # 加载后重建索引（持久化的 embedding 重新入索引）
        if self.vector_index is not None:
            for r in self.all_records():
                if r.embedding is not None:
                    self.vector_index.add(r.memory_id, r.embedding)
        self.node_index.rebuild(self._records.values())

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
        # 同步节点标记索引（图架构：owner_node → memory_id）
        self.node_index.add(candidate.memory_id, getattr(candidate, "owner_node", None))
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

    def all_records(self) -> list[MemoryCandidate]:
        """全量记录，**不做 (user_id, session_id) 过滤**（向量索引/节点索引重建用）。

        与 `all()` 的区别：`all()` 仍保留原有 session 过滤语义（不改既有行为），
        本方法仅供索引维护使用。
        """
        return list(self._records.values())

    # ---- 图架构：节点标记视图（§2.3）----

    def by_node(self, node_id: str) -> list[MemoryCandidate]:
        """某节点**专属记忆**（按 owner_node 标记过滤，零拷贝）。"""
        return [r for r in (self.get(i) for i in self.node_index.ids_of(node_id)) if r is not None]

    def visible_from(self, node_ids, fallback_all: bool = False):
        """返回限定到给定节点集合的**只读池代理**（交给 BaseRetriever 检索）。

        `fallback_all=True`：范围内为空时退回全池（防止冷启动期召不到任何候选）。
        """
        from .graph.view import ScopedPool
        return ScopedPool(self, node_ids, fallback_all=fallback_all)

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
        "owner_node": getattr(c, "owner_node", "main"),      # 图架构：节点标记（§2.1）
        "producer_run": getattr(c, "producer_run", ""),
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
        # 迁移策略（§7.4）：旧行缺字段 → 读入补 "main"，**不回写文件**
        owner_node=d.get("owner_node", "main"),
        producer_run=d.get("producer_run", ""),
    )
