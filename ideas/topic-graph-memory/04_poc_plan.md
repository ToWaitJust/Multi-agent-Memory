# 04 · 最小验证方案（POC）

> 目标：**不碰主链路**，用最小代价判断这个 idea 值不值得做。
> 时间预算：1 天搭 + 半天跑。产出是一个 go / no-go 结论，不是可交付代码。

---

## POC 边界（明确不做）

| 不做 | 原因 |
|---|---|
| 不接 AgentScope 真实运行时 | 用固定图 + 固定 query，先验证调度逻辑 |
| 不做 LLM 实时调用 | `w_prior` 直接用 `DEFAULT_WEIGHTS` 兜底值 |
| 不做可视化 | 图视图是 P2 |
| 不写 `graph.py` 进 `srtp_memory/` | 先写在 `ideas/topic-graph-memory/poc/` 下，验证通过再迁移 |
| 不做关系识别（LLM 判边） | 图结构**手工构造 / 脚本合成**（ground-truth 明确） |

---

## Step 1 · 造图（半天）

在 `poc/` 下写一个脚本，构造一个带 ground-truth 的小图：

```
节点 8–12 个（topic），边 12–20 条
要求：至少 1 个分叉（一个节点有两个下游）
      至少 1 个汇聚（一个节点有两个上游）
      平均出度 1.5–4
```

**优先用真实的**：拿自己项目里的一段真实多轮会话（比如当前这个 SRTP 项目的若干次讨论：权重方案 / 消融设计 / AgentScope 集成 / 图 idea），手工切成 topic 节点，边按真实依赖打。
**没有就用合成的**：脚本生成，标注清楚是合成的（结论说服力打折，但足以验证调度逻辑）。

每条记忆挂在一个节点下（写回实体池），并**人工标注**：
- 对某个 query，哪些记忆是"应该被召回的"（recall ground-truth）
- 哪些源节点是"应该被参考的"（source ground-truth）

---

## Step 2 · 跑调度（半天）

在 `poc/` 里实现最小版 Stage A：

```python
def select_sources(cur, graph, topk, quota_mode, tau, K=10) -> dict[str, int]:
    # 1. 邻域展开（depth=1，强边优先）
    # 2. 环检测（本 POC 里手工图保证 DAG，只做 assert）
    # 3. 按 quota_mode 分配配额（uniform / strength）
    # 4. 返回 {node_id: quota}
```

Stage B 直接**复用现成的** `srtp_memory` 四维打分 + 混合权重（把权重固定为 `DEFAULT_WEIGHTS`，跳过 LLM 与 MLP 训练）。

跨源合并 + 去重 + 截断 top-K = 10。

---

## Step 3 · 只跑三组（关键，不要贪多）

**G2（单源） / G3（多源 + 均匀配额） / G4（多源 + 边强度配额）**

G0/G1 是下界，POC 阶段可以只跑 G1（图 + 全量）作为对照，省一半时间。

指标只算 4 个（离线可算，不需要 LLM 判分）：

| 指标 | 怎么算 |
|---|---|
| `recall_acc` | 注入集 ∩ recall ground-truth / |ground-truth| |
| `source_hit_rate` | 选中源 ∩ source ground-truth / |source ground-truth| |
| `multi_source_ratio` | 注入集中来自 ≥2 个源的条目占比 |
| `compression_ratio` | 1 − |注入集| / |实体池| |

---

## Step 4 · 判据（写死，跑完就照着判）

| 结果 | 结论 |
|---|---|
| `G4 > G2` 且 `multi_source_ratio(G4) > 0.3` | ✅ 多对多优于一対一，**继续** |
| `G4 > G3` | ✅ 边强度配额有效，**继续**（否则简化掉 Stage A 配额） |
| `G4 ≈ G2` 或 `multi_source_ratio ≈ 0` | ❌ 多源没生效，idea 不成立，**归档** |
| `G1 ≈ G4` | ⚠️ 增益主要来自图结构而非算法，创新主张要改口径 |

**任一条 ❌ 就停**，不要继续投入。

---

## Step 5 · 决策后动作

- **go**：把 `poc/` 里的 `graph.py` / `source_selector.py` 迁移进 `srtp_memory/`，按 `03_ablation_design.md` 落 5 份 YAML，回推主文档 V3.0。
- **no-go**：本目录整体移到 `docs/_archive_2026xxxx/`，主文档零改动；论文里可作为 future work 一句话带过。
