"""端到端演示（§3.3 / §4.10 验收用例）：主线写入 → 协调层调度 → 共享池 → 副线读取。

验证 D5 验收子项：全链路跑通 + 四维打分/混合/选择 + 指标落盘。
用法: python examples/sched_demo.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from srtp_memory.attention import MemoryCandidate
from srtp_memory.config import SchedulingConfig
from srtp_memory.condition import ConditionKey
from srtp_memory.middleware import MemorySchedulingMiddleware
from srtp_memory.plugins import register_all


def main() -> None:
    print("=" * 60)
    print("srtp_memory 端到端调度演示（headless）")
    print("=" * 60)

    register_all()
    cfg = SchedulingConfig.load("config/srtp.yaml")

    # 1. 装配调度中间件（full-ours 视图）
    mid = MemorySchedulingMiddleware(
        config=cfg,
        user_id="u_tu", session_id="demo_main",
        log_path="data/metrics/schedule.jsonl",
        llm_prior_fn=lambda q: {"time": 0.20, "semantic": 0.40, "frequency": 0.15, "task": 0.25},
    )
    print("\n[1] 中间件装配完成:", cfg.retriever_impl, "/", cfg.attention_impl,
          "/", cfg.weight_impl, "/", cfg.selector_impl)

    # 2. 主线"对话"沉淀记忆（真实场景由 ReMe auto_memory 写卡片后 upsert 进常驻池）
    now = time.time()
    memories = [
        MemoryCandidate(memory_id="m_tech", text="技术栈：AgentScope 2.0 + ReMe + DashScope embedding",
                        path="daily/2026-08-19/tech.md", timestamp=now - 7200,
                        task_tag="tech_stack", user_id="u_tu", session_id="demo_main"),
        MemoryCandidate(memory_id="m_user", text="用户涂伟健：信息管理专业，正在做多智能体记忆调度课题",
                        path="daily/2026-08-19/user.md", timestamp=now - 3600,
                        task_tag="profile", user_id="u_tu", session_id="demo_main"),
        MemoryCandidate(memory_id="m_old", text="项目早期用 BERT-base 做语义向量（已弃用）",
                        path="daily/2026-08-10/old.md", timestamp=now - 3600 * 72,
                        task_tag="tech_stack", user_id="u_tu", session_id="demo_main"),
    ]
    for m in memories:
        mid.resident_pool.upsert(m)
    print(f"[2] 常驻池写入 {len(memories)} 条记忆（per-user: u_tu）")

    # 3. 副线提问 → 调度（pre_schedule：检索→打分→权重→选择）
    query = "我们的项目现在用什么技术栈？"
    q_emb = np.zeros(cfg.embedding_dimensions, dtype=np.float32)
    q_emb[0] = 1.0
    cond = ConditionKey(user_id="u_tu", scenario_id="research", business_id="srtp")
    res = mid.schedule_once(query, query_emb=q_emb, task_tag="tech_stack",
                            now=time.time(), condition=cond)
    print(f"\n[3] 调度完成: query='{query}'")
    print(f"    候选 {len(res.candidates)} 条 → 保留 {len(res.kept)} 条"
          f"（压缩比 {res.compression_ratio:.2f}）")
    print(f"    动态权重: { {k: round(v,3) for k,v in res.weights.items()} }")

    # 4. 副线读取共享池（dispatch = 注入 HintBlock 的内容）
    print("\n[4] 副线注入内容（共享池 top() 全量）:")
    for item in mid.shared_pool.top():
        print(f"    - {item['text']}  ({item['path']}:{item['start_line']}-{item['end_line']})")

    # 5. 指标落盘验证
    mid.flush()
    metrics_path = Path("data/metrics/schedule.jsonl")
    n_rows = sum(1 for _ in open(metrics_path, encoding="utf-8")) if metrics_path.exists() else 0
    pool_path = Path("data/reme/u_tu/sub1/shared_pool.json")
    print(f"\n[5] 落盘验证: schedule.jsonl 累计 {n_rows} 行 | shared_pool.json {'存在' if pool_path.exists() else '待持久化'}")
    mid.shared_pool.persist()
    print("    共享池已持久化:", pool_path)
    print("\n✅ 端到端演示完成（全链路：写入 → 调度 → 注入 → 落盘）")


if __name__ == "__main__":
    main()
