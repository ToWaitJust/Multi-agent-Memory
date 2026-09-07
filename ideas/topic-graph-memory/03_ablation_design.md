# 03 · 以图为基底的消融重设计（G0–G4）

> 设计原则：**图结构固定为常量，只有调度组件是变量**。
> 这样才能回答"提升来自算法还是来自图"。

---

## 1. 为什么必须重设计

原 5 组消融（baseline-naive / baseline-reme / +attention / +weight / full-ours）是在**主副线**语境下设计的，变量是"注意力 / 权重 / 选择器"。

换成图之后，多出一个变量层：**Stage A（选源）**。如果不重设计，会出现两个问题：

1. 无法回答"多对多优于一対一"（缺了主副线等价物这一环）；
2. 图的贡献和算法的贡献混在一起，审稿人一问就穿。

---

## 2. 五组设计

| 组 | 图结构 | Stage A 选源 | Stage B 四维打分 | 混合权重 | 边强度配额 | 等价/证明什么 |
|---|---|---|---|---|---|---|
| **G0 · 邻域全量** | ✓ | 邻域内全部源 | ✗（全量塞入） | ✗ | ✗ | **下界**：有图但完全不调度 |
| **G1 · 图 + 均匀** | ✓ | 邻域全部源，配额**均匀** | ✗（按写入序取） | ✗ | ✗ | **下界**：图结构 + 无算法（Q2 对照组） |
| **G2 · 图 + 单源** | ✓ | **只取边强度最高的 1 个源** | ✓ | ✓ | 单源（无配额概念） | **主副线等价物** ★ 证明"多对多优于一対一" |
| **G3 · 多源 + 四维** | ✓ | 多源，配额**均匀** | ✓ | ✓ | ✗ | 剥离"边强度配额"这一层的贡献 |
| **G4 · full-ours** | ✓ | 多源，配额按**边强度** | ✓ | ✓ | ✓ | 完整方案 |

### 关键对比（论文里要主打的三条）

```
G0 → G1 ：证明「需要调度」（有图也不够）
G2 → G3 ：证明「多对多优于一対一」★ 核心创新点
G3 → G4 ：证明「边强度参与配额有效」  （Q2 的判据）
G1 → G4 ：证明「增益来自算法而非图结构」（Q1 的判据，必须报告）
```

---

## 3. 变量控制表

| 变量 | G0 | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|
| `graph.enabled` | true | true | true | true | true |
| `source_mode` | auto | auto | auto | auto | auto |
| `source.topk` | ∞（全邻域） | ∞（全邻域） | **1** | 3 | 3 |
| `quota.mode` | none | **uniform** | n/a（单源） | **uniform** | **strength** |
| `attention.impl` | attention.noop | attention.noop | attention.full | attention.full | attention.full |
| `weight.impl` | weight.uniform | weight.uniform | weight.full | weight.full | weight.full |
| `retriever.impl` | **retriever.vector（五组统一）** | 同左 | 同左 | 同左 | 同左 |
| `selector.impl` | selector.topk | selector.topk | selector.full | selector.full | selector.full |
| `quota.lambda` | n/a | n/a | n/a | — | **0.7（五组共用的固定超参）** |

> **`λ`（静态边强度 vs query 相关度的融合系数）不进消融变量链**，五组共用 0.7（见 `01_idea_draft.md` §4.5）。
> 它回答的是"纯静态够不够"，属于**架构内部超参**而非"调度组件的有无"，因此：
> - 主链 G0–G4 **不为其增设组别**（避免 5 组膨胀到 7 组）；
> - 改在论文附录做**敏感性分析**：`λ ∈ {0.5, 0.7, 0.9, 1.0}`，报告 `source_hit_rate` / `recall_acc` 曲线。
> - 若 λ=1.0（纯静态）与 λ=0.7 差异 < 2%，则结论为"纯静态足够"，Stage A 实现可进一步简化（对工程是好事，要在论文里主动说明）。

> **检索基座必须五组统一为 `retriever.vector`**：这是主文档 V2.0 踩过的坑（`full-ours` 单独换 retriever 导致归因不纯），本设计直接修正。
> 若要保留 naive 下界，可再加一组 **G-none = `retriever.full` + 无边**，但注意它与 G0 的差异同时包含"检索器"和"图"，归因不纯，只能作为参考下界，不进主链。

---

## 4. 指标（在现有基础上新增 4 个）

| 指标 | 现有 | 说明 |
|---|---|---|
| `recall_acc` | ✓ | 命中人工标注应召回记忆的比例 |
| `task_completion` | ✓ | 任务完成度（LLM 判分） |
| `token_reduction` / `compression_ratio` | ✓ | 压缩效果 |
| `avg_latency_ms` | ✓ | NFR-1 < 1s |
| **`source_hit_rate`** | **新增** | 人工标注"应参考的源节点"中被实际选中的比例 → 衡量 Stage A |
| **`multi_source_ratio`** | **新增** | 注入集中来自 ≥2 个不同源节点的条目占比 → 证明多对多真的发生 |
| **`cross_source_relevance`** | **新增** | 每条记忆与其源节点 topic 的平均相关分 → 防凑覆盖率 |
| **`dedup_rate`** | **新增** | 被路径去重掉的重复条目占比 → 衡量多路径冗余 |

### 期望趋势（用来判断实验是否跑对）

```
recall_acc:        G0 < G1 < G2 < G3 < G4
multi_source_ratio: G0/G1 高但无意义(全塞)，G2 ≈ 0，G3/G4 显著 > 0
source_hit_rate:   G3 < G4  （边强度配额应提升选源质量）
token_reduction:   G0 最差（全量），G4 最优
```

如果 `multi_source_ratio` 在 G3/G4 接近 0，说明**多源根本没生效**（配额或去重有 bug），必须先修再谈结论。

---

## 5. 配置落地（沿用插件式 + YAML 唯一真源）

新增 5 份 `config/ablation/graph_g0.yaml ... graph_g4.yaml`，字段沿用现有 `config/srtp.yaml` 的 `*_impl` 命名，新增：

```yaml
# config/ablation/graph_g4.yaml
retriever_impl: retriever.vector
attention_impl: attention.full
weight_impl:    weight.full
selector_impl:  selector.full

graph:
  enabled: true
  source_mode: auto        # auto | manual
  depth: 1                 # 邻域扩散深度
  source_topk: 3           # 最多取几个源
  quota_mode: strength     # none | uniform | strength
  quota_tau: 0.5           # softmax 温度
  quota_lambda: 0.7        # 静态 strength vs query 相关度 的融合系数（固定超参，不进消融变量链）
  min_quota: 1             # 保底配额，防止源被饿死
  edge_types: [derives_from, depends_on, references, similar_to]
```

`ablation_manifest.json` 仍由 `analysis/gen_ablation_manifest.py` 从 YAML 自动生成，**不手写**。

---

## 6. 未决

- **数据集**：见 `02_open_questions.md` Q4（阻塞项）
- **原 5 组消融怎么办**：建议本设计**替换**原 5 组，而不是叠加（叠加 = 10 组，跑不完也没必要）。原 5 组中的 `baseline-reme`（wrap ReMe 原生）可保留为 **G-reme** 一个额外参考点，但不进主链。
- **主副线的位置**：降级为 G2 的脚注说明（"G2 等价于主副线架构"），不再作为架构叙事的一部分。
