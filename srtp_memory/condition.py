"""条件键与条件嵌入（§4.3 / §11.3）。

condition = 可学习 MLP 的可选拼接输入（user/scenario/business 的嵌入），用于个性化。
V2.1 定稿：V1 即实现 —— 各轴哈希查表为定长向量，未知 user → 零向量（冷启动退化为全局）。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConditionKey:
    """个性化条件三轴（全部可空，空 → 该轴不参与调制）。"""

    user_id: str | None = None       # 用户标识（轴 1）
    scenario_id: str | None = None   # 场景标识（轴 2）
    business_id: str | None = None   # 业务标识（轴 3）

    def to_tuple(self) -> tuple[str | None, str | None, str | None]:
        return (self.user_id, self.scenario_id, self.business_id)

    @classmethod
    def parse(cls, condition_key: str) -> "ConditionKey":
        """从 "u_tu|research|srtp" 形式解析；缺失段为 None。"""
        parts = condition_key.split("|") if condition_key else []
        parts = (parts + [None] * 3)[:3]
        return cls(user_id=parts[0], scenario_id=parts[1], business_id=parts[2])


class ConditionStore:
    """将三轴各自哈希为固定向量，拼接为 condition_emb（condition_emb_dim 维）。

    实现：每轴独立哈希 → 单位化伪随机向量（确定性，同输入同输出）；空轴贡献零向量。
    """

    def __init__(self, emb_dim: int = 32):
        # 三轴不等分（不要求 3 整除）：user 轴占一半，scenario/business 各占 1/4（余数并入末轴）
        if emb_dim < 3:
            raise ValueError(f"condition_emb_dim 至少为 3（当前 {emb_dim}）")
        self.emb_dim = emb_dim
        self._dims = [emb_dim // 2, emb_dim // 4, emb_dim - emb_dim // 2 - emb_dim // 4]

    def embed(self, condition: ConditionKey) -> np.ndarray:
        """返回 (emb_dim,) 向量 = concat(user_vec, scenario_vec, business_vec)。"""
        parts = [
            self._axis_vec(condition.user_id, self._dims[0]),
            self._axis_vec(condition.scenario_id, self._dims[1]),
            self._axis_vec(condition.business_id, self._dims[2]),
        ]
        return np.concatenate(parts).astype(np.float32)

    def _axis_vec(self, value: str | None, n: int) -> np.ndarray:
        if value is None or n <= 0:
            return np.zeros(max(n, 0), dtype=np.float32)
        # 确定性伪随机向量：哈希字节 → [0,255] → 映射到 [-1,1]，float64 归一化防溢出
        raw = bytearray()
        while len(raw) < n:
            raw += hashlib.sha256(value.encode("utf-8") + len(raw).to_bytes(4, "big")).digest()
        ints = np.frombuffer(bytes(raw[:n]), dtype=np.uint8).astype(np.float64)
        vec = (ints / 127.5) - 1.0   # [0,255] → [-1,1]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)
