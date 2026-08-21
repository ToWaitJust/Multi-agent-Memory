"""装配入口（§4.10）：加载 srtp.yaml → 构建主副线 → 跑端到端演示 → 登记 run_manifest。

用法: python main.py [--demo] [--ablation]
V2.1：真实 AgentScope 装配在接环境时完成；本入口先跑 headless 调度演示 + 消融批量。
"""
from __future__ import annotations

import argparse
import time

from srtp_memory.cli import run_ablation, run_one
from srtp_memory.config import SchedulingConfig
from srtp_memory.monitor.run_registry import RunRegistry


def main() -> None:
    ap = argparse.ArgumentParser(description="srtp_memory 装配入口")
    ap.add_argument("--demo", action="store_true", help="跑端到端调度演示")
    ap.add_argument("--ablation", action="store_true", help="批量跑 5 组消融")
    ap.add_argument("--config", default="config/srtp.yaml")
    args = ap.parse_args()

    cfg = SchedulingConfig.load(args.config)

    # run_manifest 登记（FR-10）
    reg = RunRegistry()
    reg.register({
        "run_id": f"run_{time.strftime('%Y%m%d_%H%M%S')}",
        "timestamp": time.time(),
        "config_hash": RunRegistry.config_hash(cfg.model_dump()),
        "seed": 42,
        "model": cfg.model_impl,
        "embedding_provider": cfg.embedding_provider,
        "embedding_dim": cfg.embedding_dimensions,
        "conditioning_dims": ",".join(cfg.conditioning_dims),
        "alpha": cfg.alpha,
        "beta": cfg.beta,
        "retrieval_mode": cfg.retrieval_mode,
        "ablation_group": "full_ours" if args.config.endswith("full_ours.yaml") else "srtp",
        "note": "headless demo run",
    })

    if args.ablation:
        run_ablation("config/ablation")
    else:
        # 默认：单组跑（full-ours 视图）
        r = run_one(args.config)
        print(f"[run] group={r['group']} retriever={r['retriever']}")
        print(f"  kept={r['kept_count']}/{r['candidate_count']} ratio={r['compression_ratio']:.2f}")
        print(f"  weights={r['weights']}")


if __name__ == "__main__":
    main()
