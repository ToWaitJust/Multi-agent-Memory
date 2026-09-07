"""离线假 embedding：用字符 bigram 的确定性哈希向量模拟语义空间。

POC 专用——不联网、不调 DashScope，同主题文本共享 bigram → 余弦自然偏高。
正式实验必须换回真实 embedding，这里只验证调度逻辑。
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

_DIM = 96
_token_re = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


def _bigrams(text: str) -> list[str]:
    """中文按单字、英文按词，再拼相邻 bigram，保证局部相似性可传递。"""
    toks = _token_re.findall(text.lower())
    grams = list(toks)
    grams += [a + b for a, b in zip(toks, toks[1:])]
    return grams


def text_vec(text: str, dim: int = _DIM) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float64)
    for g in _bigrams(text):
        h = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16)
        v += np.random.default_rng(h).normal(size=dim)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0
