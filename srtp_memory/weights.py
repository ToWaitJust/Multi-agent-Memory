"""混合权重计算器（§4.3，FR-4，条件化）。

V2.1 定稿：可学习 MLP 输入 = query_emb(1024) 拼接 condition_emb(32) → 四维权重，
不输入召回集合统计特征 —— 权重生成基于用户提问（与 LLM 先验同源，MLP 蒸馏复现 LLM 判断）。
α/β 固定（config），V1 不学习。两阶段学习回路见 §11.2（train/distill.py、train/feedback_loop.py）。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

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

    def fit(self, dataset: list, batch_size: int = 16, epochs: int = 20,
            lr: float = 1e-3, weight_decay: float = 1e-4,
            val_split: float = 0.0, seed: int = 42) -> dict:
        """批量 MSE 蒸馏训练（替代逐样本 step，更稳定、可复现）。

        dataset: list[(query_emb, condition_emb, target_w_np(4))]
        损失：MSE(softmax(net(x)), target)，与 step 一致（soft 分布匹配，非 one-hot）。
        优化：Adam + weight_decay；每 epoch 打乱训练集；可选验证切分监控过拟合。
        返回 {history:[{epoch,train_loss,val_loss}], final_train_loss, final_val_loss}。
        """
        if not dataset:
            return {"history": [], "final_train_loss": 0.0, "final_val_loss": 0.0}
        rng = np.random.default_rng(seed)
        X = np.stack([self._feat_vec(q, c) for (q, c, _) in dataset])
        Y = np.stack([np.asarray(t, dtype=np.float32) for (_, _, t) in dataset])
        n = len(dataset)
        n_val = int(n * val_split) if 0.0 < val_split < 1.0 else 0
        perm_all = rng.permutation(n)
        val_idx = perm_all[:n_val]
        train_idx = perm_all[n_val:]

        self._net.train()
        opt = torch.optim.Adam(self._net.parameters(), lr=lr, weight_decay=weight_decay)
        history = []
        for ep in range(epochs):
            perm = rng.permutation(len(train_idx))
            train_losses = []
            for i in range(0, len(train_idx), max(1, batch_size)):
                b = train_idx[perm[i:i + batch_size]]
                xb = torch.from_numpy(X[b])
                yb = torch.from_numpy(Y[b])
                opt.zero_grad()
                out = self._net(xb)
                loss = torch.nn.functional.mse_loss(torch.softmax(out, dim=-1), yb)
                loss.backward()
                opt.step()
                train_losses.append(loss.item())
            self._net.eval()
            val_loss = None
            if n_val > 0:
                with torch.no_grad():
                    vout = self._net(torch.from_numpy(X[val_idx]))
                    val_loss = torch.nn.functional.mse_loss(
                        torch.softmax(vout, dim=-1), torch.from_numpy(Y[val_idx])).item()
            history.append({
                "epoch": ep + 1,
                "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
                "val_loss": val_loss,
            })
            self._net.train()
        self._net.eval()
        return {
            "history": history,
            "final_train_loss": history[-1]["train_loss"],
            "final_val_loss": history[-1]["val_loss"],
        }

    def save(self, path: str) -> None:
        """持久化 MLP 权重（state_dict）。训练产物落盘，供推理加载。"""
        torch.save(self._net.state_dict(), path)

    def load(self, path: str) -> None:
        """从 state_dict 载入 MLP 权重。"""
        self._net.load_state_dict(torch.load(path, map_location="cpu"))


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

    def pretrain_distill(self, prior_dataset: list, batch_size: int = 16,
                          epochs: int = 20, lr: float = 1e-3,
                          weight_decay: float = 1e-4, val_split: float = 0.0) -> dict:
        """离线预训练（阶段一，批量版）：MSE 蒸馏目标权重。

        prior_dataset: list of (query_emb, condition_key, target_w_dict)
        target_w = 目标权重（外包标签/LLM 先验，多次采样平均、归一化）。
        内部转为 (query_emb, condition_emb, target_np) 调 _ConditionedMLP.fit 批量训练。
        返回 {avg_loss, history, final_train_loss, final_val_loss}。
        """
        if not prior_dataset:
            return {"avg_loss": 0.0, "history": [], "final_train_loss": 0.0, "final_val_loss": 0.0}
        fit_ds = []
        for q_emb, cond, target_w in prior_dataset:
            c_emb = self.condition_store.embed(cond)
            t = np.array([target_w["time"], target_w["semantic"],
                          target_w["frequency"], target_w["task"]], dtype=np.float32)
            fit_ds.append((q_emb, c_emb, t))
        res = self.mlp.fit(fit_ds, batch_size=batch_size, epochs=epochs,
                           lr=lr, weight_decay=weight_decay, val_split=val_split)
        return {"avg_loss": res["final_train_loss"], **res}

    def pretrain_from_jsonl(self, jsonl_path: str, embed_fn: Callable[[str], np.ndarray],
                            batch_size: int = 16, epochs: int = 20, lr: float = 1e-3,
                            weight_decay: float = 1e-4, val_split: float = 0.1,
                            use_labels_as_prior: bool = True) -> dict:
        """从作者格式 JSONL 读取训练集并批量蒸馏（第二种喂法默认开启）。

        jsonl 每行：{"query":str, "condition":{user_id,scenario_id,business_id},
                     "target_weights":{time,semantic,frequency,task}}
        embed_fn: 文本→1024 维向量（必须与推理同源：DashScope text-embedding-v4）。
        use_labels_as_prior=True（第二种喂法）：把标签建为精确查询→权重查表，
        注入 llm_prior_fn，使推理命中训练集时先验直接取该标签（外包标签主导），
        未命中则由蒸馏后的 MLP 泛化给出并被 α 地板约束防崩。
        返回与 pretrain_distill 一致的训练结果 dict。
        """
        prior_dataset = []
        label_index: dict[str, dict[str, float]] = {}
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                q = rec["query"]
                cond_d = rec.get("condition") or {}
                cond = ConditionKey(user_id=cond_d.get("user_id"),
                                    scenario_id=cond_d.get("scenario_id"),
                                    business_id=cond_d.get("business_id"))
                tw = rec["target_weights"]
                target_w = {"time": float(tw["time"]), "semantic": float(tw["semantic"]),
                            "frequency": float(tw["frequency"]), "task": float(tw["task"])}
                q_emb = embed_fn(q)
                prior_dataset.append((q_emb, cond, target_w))
                if use_labels_as_prior:
                    label_index[q] = self._normalize(target_w)
        if use_labels_as_prior and label_index:
            self._llm_prior_fn = lambda q: label_index.get(q, dict(DEFAULT_WEIGHTS))
        return self.pretrain_distill(prior_dataset, batch_size=batch_size, epochs=epochs,
                                     lr=lr, weight_decay=weight_decay, val_split=val_split)

    def save_weights(self, path: str) -> None:
        """持久化可学习 MLP 权重到文件。"""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.mlp.save(path)

    def load_weights(self, path: str) -> None:
        """从文件载入可学习 MLP 权重。"""
        self.mlp.load(path)

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
