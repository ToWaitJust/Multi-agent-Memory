"""接口缝清单（§12.3）：全部抽象基类（ABC）。

V2.1：删除 MemoryStore —— 双池为内核具体类，不做存储后端插件化。
"""
from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod


class AttentionHead(ABC):
    """四维注意力打分器契约。score 返回 {final, time, semantic, frequency, task}。"""

    @abstractmethod
    def score(self, query_emb: np.ndarray, memory, current_time: float,
              task_tag: str) -> dict: ...

    @abstractmethod
    def names(self) -> list[str]: ...   # 四维名：["time","semantic","frequency","task"]


class WeightCalculator(ABC):
    """混合权重计算器契约。calculate 返回四维权重 dict（Σ=1）。
    V2.1：输入为 query_emb（+condition），不输入召回集合统计特征。"""

    @abstractmethod
    def calculate(self, query_emb, condition) -> dict: ...

    def pretrain_distill(self, prior_dataset: list) -> None: ...   # 可选（完整实现才实现）
    def update_with_feedback(self, query_emb, reward: float, condition) -> None: ...


class EmbeddingBackend(ABC):
    @abstractmethod
    def encode(self, text: str) -> np.ndarray: ...

    @abstractmethod
    def dim(self) -> int: ...           # 768 / 1024


class ActionSelector(ABC):
    @abstractmethod
    def select(self, scored_memories: list[dict]) -> dict: ...   # {kept, discarded}

    @abstractmethod
    def compression_ratio(self, result: dict) -> float: ...


class RewardSignal(ABC):
    """奖励信号契约。V1 仅显式点赞/点踩 → +1 / -1。"""

    @abstractmethod
    def from_feedback(self, like: bool) -> float: ...


class BaseRetriever(ABC):
    """检索基座契约（D-12）。在常驻完整池上召回候选，返回 MemoryCandidate 列表。
    实现：retriever.full / bm25 / vector / rrf（自研）；retriever.reme（wrap ReMe 原生，仅 baseline-reme）。"""

    @abstractmethod
    def recall(self, query: str, pool, top_k: int = 20) -> list: ...

    @property
    @abstractmethod
    def name(self) -> str: ...   # full / bm25 / vector / rrf / reme


class ModelAdapter(ABC):
    @abstractmethod
    def complete(self, prompt: str, **kw) -> str: ...

    @abstractmethod
    def name(self) -> str: ...


class AgentFactory(ABC):
    """智能体角色工厂契约（loop 不插，装配可插）。"""

    @abstractmethod
    def build(self, role: str, **kw): ...   # role: main / sub / reviewer
