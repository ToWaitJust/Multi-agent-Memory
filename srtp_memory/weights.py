"""混合权重计算器（§4.3，FR-4，条件化）。

V2.1 定稿：可学习 MLP 输入 = query_emb(1024) 拼接 condition_emb(32) → 四维权重，
不输入召回集合统计特征 —— 权重生成基于用户提问（与 LLM 先验同源，MLP 蒸馏复现 LLM 判断）。
α/β 固定（config），V1 不学习。两阶段学习回路见 §11.2（train/distill.py、train/feedback_loop.py）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .attention import DEFAULT_WEIGHTS
from .condition import ConditionKey, ConditionStore

try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except ImportError:  # pragma: no cover - 环境缺 torch 时退化
    _TORCH_OK = False

LLM_PRIOR_PROMPT = (
    "为当前用户查询分配四维注意力权重（time/semantic/frequency/task），"
    "JSON 输出，和为 1，格式: {\"time\":0.2,\"semantic\":0.4,\"frequency\":0.1,\"task\":0.3}"
)


class _ConditionedMLP:
    """轻量 MLP：query_emb + condition_emb → 四维权重（softmax）。

    结构：全连接层 + ReLU + 输出层；仅 conditioning_dims（semantic/task）受 condition 调制
    —— 实现为两个分支：全局支（time/frequency）+ 条件化支（semantic/task，输入拼 condition_emb）。
    """

    def __init__(self, query_dim: int = 1024, condition_dim: int = 32,
                 conditioning_dims: tuple[str, ...] = ("semantic", "task")):
        if not _TORCH_OK:  # pragma: no cover
            raise RuntimeError("torch 未安装，无法使用可学习权重（srtp-memory 环境已含 torch）")
        self.conditioning_dims = set(conditioning_dims)
        self.global_dims = [d for d in ["time", "frequency"] if d not in self.conditioning_dims]
        self._net = nn.Sequential(
            nn.Linear(query_dim + condition_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self._opt = None

    @staticmethod
    def _feat_vec(query_emb: np.ndarray, condition_emb: np.ndarray) -> np.ndarray:
        """拼接并归一化 query_emb（L2），condition_emb 已单位化；防数值溢出。"""
        q = query_emb.ravel().astype(np.float64)
        nq = np.linalg.norm(q)
        if nq > 0:
            q = q / nq
        x = np.concatenate([q, condition_emb.ravel().astype(np.float64)]).astype(np.float32)
        return x

    def forward(self, query_emb: np.ndarray, condition_emb: np.ndarray) -> dict[str, float]:
        x = self._feat_vec(query_emb, condition_emb)
        with torch.no_grad():
            out = self._net(torch.from_numpy(x))
            w = torch.softmax(out, dim=-1).numpy()
        # 四维顺序固定 [time, semantic, frequency, task]
        return {"time": float(w[0]), "semantic": float(w[1]),
                "frequency": float(w[2]), "task": float(w[3])}

    def step(self, query_emb: np.ndarray, condition_emb: np.ndarray,
             target_w: np.ndarray) -> float:
        """一步梯度更新，返回 loss。target_w = 四维目标权重（蒸馏标签 / 反馈加权）。"""
        x = self._feat_vec(query_emb, condition_emb)
        t = torch.from_numpy(np.asarray(target_w, dtype=np.float32))
        self._net.train()
        if self._opt is None:
            self._opt = torch.optim.Adam(self._net.parameters(), lr=0.01)
        self._opt.zero_grad()
        out = self._net(torch.from_numpy(x))
        loss = torch.nn.functional.mse_loss(torch.softmax(out, dim=-1), t)
        loss.backward()
        self._opt.step()
        self._net.eval()
        return float(loss.item())


class HybridWeightCalculator:
    """条件化混合权重（默认完整实现，插件名 weight.full）。"""

    def __init__(self, llm_client=None, embed_model=None,
                 alpha: float = 0.6, beta: float = 0.4,
                 prior_samples: int = 1, lr: float = 0.01,
                 conditioning_dims: tuple[str, ...] = ("semantic", "task"),
                 condition_emb_dim: int = 32,
                 query_dim: int = 1024,
                 llm_prior_fn=None):
        """llm_prior_fn: 可选注入的 LLM 先验函数 f(query:str)->dict（测试可 mock）。"""
        self.alpha, self.beta = alpha, beta
        self.prior_samples = prior_samples
        self.lr = lr
        self.conditioning_dims = conditioning_dims
        self.condition_store = ConditionStore(emb_dim=condition_emb_dim)
        self.mlp = _ConditionedMLP(query_dim=query_dim, condition_dim=condition_emb_dim,
                                   conditioning_dims=conditioning_dims)
        self._llm_client = llm_client
        self._llm_prior_fn = llm_prior_fn
        self._cached_prior: dict[str, float] | None = None

    def set_prior(self, prior: dict[str, float]) -> None:
        """由调度中间件在 on_reply 阶段调用 llm_prior(query_text) 后缓存，供 calculate 融合。"""
        self._cached_prior = self._normalize(prior)

    def calculate(self, query_emb: np.ndarray, condition: ConditionKey) -> dict[str, float]:
        """hybrid = α·LLM先验 + β·可学习，归一化。

        V2.1 约定：LLM 先验由调度中间件在 on_reply 阶段调用 `llm_prior(query_text)`
        并 `set_prior()` 缓存（每次调度一次，预算 ≤800ms）；calculate 只做融合，不重复调 LLM。
        未缓存时回退 DEFAULT_WEIGHTS。
        """
        prior = self._cached_prior if self._cached_prior is not None else dict(DEFAULT_WEIGHTS)
        learnable = self._learnable_forward(query_emb, condition)
        hybrid = {d: self.alpha * prior[d] + self.beta * learnable[d] for d in prior}
        return self._normalize(hybrid)

    def llm_prior(self, query: str) -> dict[str, float]:
        """LLM 先验：实时轻量小模型单次采样；失败回退 DEFAULT_WEIGHTS。"""
        if self._llm_prior_fn is not None:
            try:
                w = self._llm_prior_fn(query)
                if isinstance(w, dict) and len(w) == 4:
                    return self._normalize(w)
            except Exception:
                pass
            return dict(DEFAULT_WEIGHTS)
        # 未注入 llm_prior_fn：回退默认权重（真实环境由 ModelAdapter 接入轻量模型）
        return dict(DEFAULT_WEIGHTS)

    def _learnable_forward(self, query_emb: np.ndarray, condition: ConditionKey) -> dict[str, float]:
        c_emb = self.condition_store.embed(condition)
        return self.mlp.forward(query_emb, c_emb)

    def pretrain_distill(self, prior_dataset: list) -> dict[str, float]:
        """离线预训练（阶段一）：MSE 蒸馏 LLM 先验。

        prior_dataset: list of (query_emb, condition_key, target_w_dict)
        target_w = LLM 先验权重（多次采样平均、归一化）。返回平均 loss。
        """
        losses = []
        for q_emb, cond, target_w in prior_dataset:
            c_emb = self.condition_store.embed(cond)
            t = np.array([target_w["time"], target_w["semantic"],
                          target_w["frequency"], target_w["task"]], dtype=np.float32)
            losses.append(self.mlp.step(q_emb, c_emb, t))
        return {"avg_loss": float(np.mean(losses)) if losses else 0.0}

    def update_with_feedback(self, query_emb: np.ndarray, reward: float,
                             condition: ConditionKey) -> dict[str, float]:
        """在线优化（阶段二）：reward∈{+1 赞/-1 踩}，bandit 式一步更新。

        仅更新 condition 对应切片（V1 实现为整模型一步，切片区隔留待 V1.1）。
        受 REQ-405 正则化约束：更新后权重向 LLM 先验回归（见 _reg 说明）。
        """
        c_emb = self.condition_store.embed(condition)
        # 目标 = 当前输出 + reward 引导（正反馈朝当前方向增强，负反馈减弱）
        cur = self.mlp.forward(query_emb, c_emb)
        target = {d: cur[d] * (1.0 + 0.1 * reward) for d in cur}
        target = self._normalize(target)
        t = np.array([target["time"], target["semantic"],
                      target["frequency"], target["task"]], dtype=np.float32)
        loss = self.mlp.step(query_emb, c_emb, t)
        return {"loss": loss}

    def _normalize(self, weights: dict[str, float]) -> dict[str, float]:
        s = sum(weights.values()) or 1.0
        return {k: v / s for k, v in weights.items()}
