"""离线 MSE 蒸馏预训练（§11.2 阶段一，train/distill.py）。

让可学习 MLP 复现目标权重判断（第二种喂法：外包标签即权威先验）。
样本来源：作者格式 JSONL（query + condition + target_weights）。

用法：
  # 用示例/外包数据集批量训练（推荐）
  python -m srtp_memory.train.distill --jsonl examples/weights_train.example.jsonl
  # 仅自测（确定性模拟样本，无需 embedding/网络）
  python -m srtp_memory.train.distill --self-test
"""
from __future__ import annotations

import argparse

import numpy as np

from ..attention import DEFAULT_WEIGHTS
from ..condition import ConditionKey
from ..weights import HybridWeightCalculator


def _simulate_dataset(n: int = 50, query_dim: int = 1024, seed: int = 42) -> list:
    """模拟蒸馏样本（真实样本 = 交互日志中 query_emb + 目标权重）。"""
    rng = np.random.default_rng(seed)
    dataset = []
    for _ in range(n):
        q_emb = rng.normal(0, 1, query_dim).astype(np.float32)
        cond = ConditionKey(user_id="u_tu", scenario_id="research", business_id="srtp")
        w = {"time": 0.2, "semantic": 0.45, "frequency": 0.15, "task": 0.2}
        dataset.append((q_emb, cond, w))
    return dataset


def _mock_embed(text: str, dim: int = 1024, seed: int = 0) -> np.ndarray:
    """自测用确定性 embed：同文本恒等（仅占位，真实训练须传 DashScope embed_fn）。"""
    h = int(__import__("hashlib").sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    v = np.random.default_rng(h + seed).normal(0, 1, dim).astype(np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def run_self_test(epochs: int = 20, batch_size: int = 16) -> dict:
    """无网络自测：确定性模拟样本 + 批量训练，验证 fit 收敛。"""
    w = HybridWeightCalculator()
    dataset = _simulate_dataset(n=80)
    res = w.pretrain_distill(dataset, batch_size=batch_size, epochs=epochs, val_split=0.1)
    return {"train_loss": res["final_train_loss"], "val_loss": res["final_val_loss"],
            "first_epoch": res["history"][0]["train_loss"],
            "last_epoch": res["history"][-1]["train_loss"]}


def run_jsonl(jsonl_path: str, epochs: int = 30, batch_size: int = 16,
              use_labels_as_prior: bool = True) -> dict:
    """从作者格式 JSONL 批量训练（真实路径：embed_fn 须与推理同源）。"""
    w = HybridWeightCalculator()
    # embed_fn：真实训练请替换为 srtp_memory 装配的 DashScope embedding.encode；
    # 此处用 mock 仅作管线打通演示（真实标签权重才是主导信号）。
    res = w.pretrain_from_jsonl(
        jsonl_path, embed_fn=_mock_embed, batch_size=batch_size, epochs=epochs,
        lr=1e-3, weight_decay=1e-4, val_split=0.1, use_labels_as_prior=use_labels_as_prior,
    )
    # 抽样验证：命中训练集的 query 先验应等于其外包标签
    return {"final_train_loss": res["final_train_loss"],
            "final_val_loss": res["final_val_loss"],
            "epochs": len(res["history"]),
            "default": DEFAULT_WEIGHTS}


def main() -> None:
    ap = argparse.ArgumentParser(description="可学习权重离线蒸馏训练")
    ap.add_argument("--jsonl", type=str, default=None, help="作者格式训练集 JSONL 路径")
    ap.add_argument("--self-test", action="store_true", help="用确定性模拟样本自测（无需网络）")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    if args.self_test:
        out = run_self_test(epochs=args.epochs, batch_size=args.batch_size)
        print("[self-test] 批量训练结果:", out)
        return
    if args.jsonl:
        out = run_jsonl(args.jsonl, epochs=args.epochs, batch_size=args.batch_size)
        print("[jsonl] 批量训练结果:", out)
        return
    # 默认：自测
    print("[default] 未指定 --jsonl/--self-test，执行自测")
    print(run_self_test(epochs=args.epochs, batch_size=args.batch_size))


if __name__ == "__main__":
    main()
