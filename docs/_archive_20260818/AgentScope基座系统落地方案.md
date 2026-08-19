# 基于 AgentScope 的多智能体记忆共享调度系统
## 基座系统落地方案（v2.0）

> 面向课题：基于注意力引导和 LLMs 可学习权重的多智能体记忆共享调度方法研究
> v2.0 更新：适配 AgentScope 2.0.6（Middleware 架构），整合 ReMe 深度解析与三层监控方案
> 更新日期：2026-08-12

---

## 目录

- [一、选型结论与环境现状](#一选型结论与环境现状)
- [二、系统架构设计（v2.0）](#二系统架构设计v20)
- [三、核心模块实现方案](#三核心模块实现方案)
- [四、开发路线图](#四开发路线图)
- [五、关键技术点与注意事项](#五关键技术点与注意事项)
- [六、环境与依赖](#六环境与依赖)
- [七、风险与应对](#七风险与应对)
- [八、配套文档索引](#八配套文档索引)

---

## 一、选型结论与环境现状

### 1.1 选型结论

**AgentScope 是本课题的最优基座框架选择。**

本课题研究的是"多智能体记忆共享调度机制"，属于多智能体系统的底层核心机制研究。AgentScope 作为专门面向多智能体研发的企业级框架，其"透明可控、模块化可扩展、内置记忆组件、全链路可观测"的设计理念，与我们的研究需求高度契合。

| 优势 | 对本课题的价值 |
|------|---------------|
| 全透明无黑盒 | 可完全干预每一步智能体行为，便于实验与消融分析 |
| 内置 ReMe 记忆组件 | 在 ReMe 基础上扩展双池架构，节省基础开发量 |
| Middleware 中间件机制 | 记忆读写完全中间件化，调度算法可通过自定义中间件插入，符合"继承而非修改"原则 |
| 原生消息调度引擎 | 主副线通信、状态同步由框架提供，专注调度逻辑 |
| 内置 OpenTelemetry 追踪 | 自动采集 Token/耗时/调用链，为科研提供量化数据 |
| 模型无关 | 一次编程适配所有主流模型，方便对比实验 |

### 1.2 环境现状（已就绪）

**开发环境已全部搭建完成：**

| 项目 | 状态 | 详情 |
|------|------|------|
| Anaconda | ✅ | `E:\1shujukuyuanli\anaconda`，Python 3.13.9，已加入系统 PATH |
| 项目环境 | ✅ | conda 环境 `srtp-memory`（`E:\1shujukuyuanli\anaconda\envs\srtp-memory`） |
| AgentScope | ✅ | **v2.0.6**（注意：非 1.x，架构为 Middleware 机制） |
| 核心依赖 | ✅ | numpy 2.5.1 / pandas 3.0.5 / scikit-learn 1.9.0 / chromadb 1.5.9 / torch 2.13.0 / sentence-transformers 5.7.0 |
| AgentScope Studio | ✅ | npm 全局安装 `@agentscope/studio`，`as_studio` 启动 |
| OpenCode 桌面版 | ⚠️ | 安装程序在 `D:\dev-tools\opencode-desktop-setup.exe`，待手动安装 |
| reme-ai | ❌ | 待安装：`pip install "agentscope[memory-reme]"`（ReMe 记忆底层） |

### 1.3 关键架构变化：1.x → 2.0（务必注意）

**v1.0 落地方案中的记忆 API 已过时**，需按 2.0 架构重新实现：

| 维度 | 1.x（旧方案） | 2.0（本方案） |
|------|--------------|---------------|
| 记忆核心类 | `agentscope.memory.MemoryBase` / `ReMeModule` | `MiddlewareBase` 子类（中间件） |
| 集成方式 | Agent 的 `memory` 参数 | Agent 的 `middlewares=[...]` 列表 |
| 记忆管理 | 手动 `add()` / `get_memory()` | Hook 驱动：`on_reply` / `on_reasoning` / `on_system_prompt` |
| 记忆注入 | 记忆列表拼接上下文 | 以 `HintBlock` 注入 |
| 追踪 | Studio 独立服务 | 内置 `TracingMiddleware`（OpenTelemetry 标准） |

**对我们的意义**：调度算法不再修改 Agent 本身，而是**编写自定义 Middleware**（`MemorySchedulingMiddleware`）插入记忆读写链路——这天然支持三层监控与实验数据采集。

---

## 二、系统架构设计（v2.0）

### 2.1 整体架构：五层 + 中间件链

```
┌─────────────────────────────────────────────────────────────────┐
│                    ⑤ 监控层（三层可观测）                         │
│  TracingMiddleware（OTel）｜ ReMe Job 监控 ｜ 调度层埋点          │
└─────────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────────┐
│                     ④ 协调层（核心创新层）                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ 四维注意力    │  │ LLM先验+可学习│  │ 动作选择与调度决策   │   │
│  │ 打分器       │  │ 混合权重计算  │  │ （替代固定top_k）    │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │       MemorySchedulingMiddleware（自定义中间件）           │   │
│  │       检索 → 打分 → 融合 → 选择 → 注入 全链路可观测        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────────┐
│                     ③ 记忆层（核心创新层）                        │
│  ┌──────────────────────┐      ┌──────────────────────┐         │
│  │  完整上下文记忆池     │─────►│    共享记忆池        │         │
│  │  （ReMe workspace）  │ 调度 │  （调度器精选输出）   │         │
│  │  auto_memory写入     │      │  供副线 Agent 检索   │         │
│  │  BM25+向量混合检索   │      │  数量少、低延迟      │         │
│  └──────────────────────┘      └──────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────────┐
│                     ② 智能体层（AgentScope 原生）                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ 主线智能体    │  │ 副线智能体1   │  │ 副线智能体2   │  ...     │
│  │ (MainAgent)  │  │ (SubAgent)   │  │ (SubAgent)   │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│  每个 Agent 挂载：TracingMiddleware + ReMeMiddleware             │
│                 + MemorySchedulingMiddleware（调度）             │
└─────────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────────┐
│                     ① 接口层                                     │
│  Electron 多窗口前端 ｜ LLM API 封装（AgentScope 原生）          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 中间件链机制（v2.0 核心）

每个 Agent 的回复流程是一条 **Middleware 链**，按序执行：

```
用户输入
  → [TracingMiddleware.on_reply]        记录回复/耗时（OTel Span）
  → [MemorySchedulingMiddleware.on_reply]
       ① 从完整池召回候选（包装 ReMe search）
       ② 四维注意力打分 + 混合权重 + 动作选择
       ③ 记录调度指标（schedule.jsonl）
       ④ 选中记忆写入共享池
  → [ReMeMiddleware.on_reply]           自动写入本轮对话（auto_memory）
  → LLM 调用（on_model_call 自动采集 token/耗时）
  → [MemorySchedulingMiddleware.on_reasoning] 注入调度结果（HintBlock）
  → 回复输出
```

**关键点**：
- 写入（auto_memory）在每次回复后自动执行，无需手动 add
- 检索与回复**并行**（异步任务），检索完成经 `on_reasoning` 注入
- 调度算法与监控埋点在同一中间件内，一码两用

### 2.3 架构设计思路

- **AgentScope 原生提供**（直接用）：智能体生命周期、消息通信、LLM 封装、ReMe 记忆底层、OTel 追踪
- **我们扩展实现**（核心创新）：
  - 双池记忆架构（完整池 = ReMe workspace + 共享池 = 调度输出）
  - 四维注意力机制（时间/语义/频率/任务）
  - LLM 先验 + 可学习权重的混合权重模型
  - 动作选择与记忆调度决策
  - 主副线隔离 + 书签定位
  - 三层监控与实验数据采集

---

## 三、核心模块实现方案

### 3.1 记忆层：双池架构（基于 ReMe）

#### 3.1.1 ReMe 是什么（简要）

ReMe（`reme-ai`）是 AgentScope 生态的 **local-first 文件化记忆层**，核心是 "Memory as File"：把对话沉淀为带 frontmatter 和 wikilink 的 Markdown 记忆节点，自动建立索引与关联。其论文《Remember Me, Refine Me》已被 **Findings of ACL 2026** 接收。

核心 Job：
- `auto_memory`：从对话轨迹 LLM 蒸馏有用事实（自动写入）
- `search`：渐进式混合检索（BM25 关键词 + 向量 + RRF 融合 + wikilink 链接展开）

> 详细机制见《记忆模块算法解析与调度重构方案.md》第 2-4 章。

#### 3.1.2 双池结构设计

```
┌────────────────────────── 记忆层（基于 ReMe 改造）────────────────┐
│                                                                    │
│  【完整上下文记忆池】= ReMe workspace（MainAgent 专属）             │
│    ├─ auto_memory 自动写入主线对话（原始轨迹 + 蒸馏记忆卡片）      │
│    ├─ BM25 + embedding + wikilink 全量索引                        │
│    └─ 语义初筛入口：search(query, top_k=50)                       │
│                                                                    │
│  【共享记忆池】= 调度器输出（SubAgent 专属）                       │
│    ├─ 由 MemoryCoordinator 从完整池调度选出                       │
│    ├─ 数量少、精选（≤10 条）                                      │
│    └─ 由副线 Agent 的中间件注入上下文                              │
│                                                                    │
│  数据流：MainAgent对话 → 完整池 → [MemoryCoordinator] → 共享池 → SubAgent│
└────────────────────────────────────────────────────────────────────┘
```

#### 3.1.3 主副线 workspace 隔离

```python
# 主线与副线各自独立 workspace（session 级隔离）
main_agent = Agent(
    name="main_agent",
    model=model,
    middlewares=[
        TracingMiddleware(),
        MemorySchedulingMiddleware(
            reme_middleware=ReMeMiddleware(
                workspace_dir="./data/reme/main",       # 主线工作区
                parameters=ReMeMiddleware.Parameters(mode="both"),
            ),
            embed_model=embed_model,
            llm_client=llm_client,
            task_tag="main_task",
        ),
    ],
)

sub_agent = Agent(
    name="sub_agent_1",
    model=model,
    middlewares=[
        TracingMiddleware(),
        MemorySchedulingMiddleware(
            reme_middleware=ReMeMiddleware(
                workspace_dir="./data/reme/sub1",       # 副线工作区
                parameters=ReMeMiddleware.Parameters(mode="both"),
            ),
            embed_model=embed_model,
            llm_client=llm_client,
            task_tag="sub_task",
        ),
    ],
)
```

**书签定位**：复用 ReMe 检索结果中的 `path:start_line-end_line` 行号定位能力，书签 = 位置快照 + 上下文提取。

### 3.2 协调层：核心调度算法（全部自研）

#### 3.2.1 MemorySchedulingMiddleware（总入口）

```python
# coordinator/middleware.py
from agentscope.middleware import MiddlewareBase


class MemorySchedulingMiddleware(MiddlewareBase):
    """记忆调度中间件：检索 → 打分 → 融合 → 选择 → 注入"""

    def __init__(
        self,
        reme_middleware,            # 包装的 ReMeMiddleware（完整池）
        embed_model,                # 向量化模型
        llm_client,                 # LLM（先验权重用）
        threshold: float = 0.6,     # 动作选择阈值
        max_shared: int = 10,       # 共享池上限
        task_tag: str = None,       # 当前任务标签
    ):
        self._reme = reme_middleware
        self.embed_model = embed_model
        self.scorer = MultiHeadAttentionMemoryScorer()
        self.weights = HybridWeightCalculator(llm_client, embed_model)
        self.selector = ActionSelector(threshold, max_shared)
        self._task_tag = task_tag
        self._shared_pool = None

    async def on_reply(self, agent, input_kwargs, next_handler):
        # 1. 召回候选（包装 ReMe search）
        # 2. 四维注意力打分 → 混合权重 → 动作选择
        # 3. 选中记忆 → 共享池（供副线检索）
        # 4. 记录调度指标（见监控方案）
        async for item in next_handler(**input_kwargs):
            yield item

    async def on_reasoning(self, agent, input_kwargs, next_handler):
        # 将调度结果以 HintBlock 注入当前 Agent 上下文
        async for event in next_handler(**input_kwargs):
            yield event
```

#### 3.2.2 四维注意力打分器

```python
# coordinator/attention.py
import math
import numpy as np


class MultiHeadAttentionMemoryScorer:
    """四维注意力记忆打分器"""

    def __init__(self, weights: dict | None = None):
        self.weights = weights or {
            "time": 0.25, "semantic": 0.35,
            "frequency": 0.15, "task": 0.25,
        }

    def score(self, query_emb, memory, current_time, task_context):
        """对一条候选记忆计算四维得分"""
        # 时间注意力：指数衰减（可学习速率）
        t = (current_time - memory.timestamp) / 3600.0
        time_score = math.exp(-0.01 * t)

        # 语义注意力：余弦相似度（可与 ReMe RRF 分数融合）
        semantic_score = float(np.dot(query_emb, memory.embedding))

        # 频率注意力：频率编码
        frequency_score = 1 - math.exp(-0.1 * memory.access_count)

        # 任务注意力：任务标签匹配
        task_score = 1.0 if memory.task_tag == task_context.task_type else 0.3

        # 加权融合
        final = (
            self.weights["time"] * time_score
            + self.weights["semantic"] * semantic_score
            + self.weights["frequency"] * frequency_score
            + self.weights["task"] * task_score
        )
        return {
            "final": final, "time": time_score, "semantic": semantic_score,
            "frequency": frequency_score, "task": task_score,
        }
```

#### 3.2.3 混合权重计算器

```python
# coordinator/weights.py
class HybridWeightCalculator:
    """混合权重：LLM 先验 α + 可学习权重 β"""

    def __init__(self, llm_client, embed_model, alpha=0.6, beta=0.4):
        self.llm = llm_client
        self.alpha, self.beta = alpha, beta
        self.learnable = {"time": .25, "semantic": .25,
                          "frequency": .25, "task": .25}

    def calculate(self, query, context) -> dict:
        # 1. LLM 多次采样先验权重（Prompt：请分配四维权重）
        llm_weights = self._llm_prior(query, context)
        # 2. 可学习权重（轻量模型前向）
        # 3. 混合 + 归一化
        hybrid = {
            k: self.alpha * llm_weights[k] + self.beta * self.learnable[k]
            for k in self.learnable
        }
        total = sum(hybrid.values())
        return {k: v / total for k, v in hybrid.items()}

    def update_with_feedback(self, query, context, feedback):
        """在线学习：根据用户反馈更新可学习权重（PyTorch 轻量模型）"""
        pass
```

#### 3.2.4 动作选择器

```python
# coordinator/selector.py
class ActionSelector:
    """动作选择：根据最终得分动态决定保留/丢弃（替代固定 top_k）"""

    def __init__(self, threshold=0.6, max_shared=10):
        self.threshold, self.max_shared = threshold, max_shared

    def select(self, scored_memories):
        kept = [m for m in scored_memories
                if m["score"]["final"] >= self.threshold]
        kept.sort(key=lambda m: m["score"]["final"], reverse=True)
        return {
            "kept": kept[:self.max_shared],
            "discarded": [m for m in scored_memories
                          if m["score"]["final"] < self.threshold],
        }
```

#### 3.2.5 与 ReMe 检索链路的衔接

```
ReMe search 原链路：
query → [BM25召回] + [向量召回] → RRF融合 → min_score → limit截断 → 链接展开

融入四维注意力后：
query → [BM25召回] + [向量召回] → RRF融合（候选集放宽到 50~200）
      → 【四维注意力打分】          ← 我们插入
      → 【混合权重融合】            ← LLM先验 α + 可学习 β
      → 【动作选择】               ← 动态保留/丢弃（替代固定 top_k）
      → 保留条目进入共享记忆池 → 注入副线上下文
```

> 注：ReMe 候选 chunk 自带 RRF 融合分，可将其作为语义注意力的先验（`semantic_score = α·cosine + β·rrf_score`），平滑衔接。

### 3.3 智能体层：主副线协同

- **主线智能体**：推进核心任务，维护完整上下文记忆池，可创建书签
- **副线智能体**：处理分支查询，从共享记忆池获取调度后的精选记忆
- **记忆总线**：副线向主线请求记忆 → 调度中间件执行调度 → 生成共享池

> 主副线交互细节与原方案一致（见 v1.0 的 MemoryBus 设计），但实现载体从 `AgentBase` 改为 **Agent + middlewares**，通信由框架消息机制承担。

### 3.4 监控层：三层可观测（科研量化支撑）

```
第 1 层：TracingMiddleware（框架内置，OTel）
  └─ 自动采集：回复/LLM 调用/工具调用 全链路 Span
     ├─ 响应时间、input/output token、状态
     └─ 数据源：平均响应时间 / Token 消耗 / 任务完成率

第 2 层：ReMe Job 监控（包装 _run_job）
  └─ 记录 search / auto_memory 的入参、命中数、耗时
     └─ 数据源：检索有效性 / 记忆增长速率

第 3 层：调度层埋点（MemorySchedulingMiddleware 内）
  └─ 记录四维得分、权重快照、保留/丢弃、压缩比、调度延迟
     └─ 数据源：调度决策延迟 / 压缩比 / 权重动态性 / 消融实验
```

> 完整指标字典、JSONL 格式、统计脚本骨架、消融实验组设计见《监控与实验数据采集方案.md》。

---

## 四、开发路线图

### 阶段一：AgentScope 上手与基础搭建（2026.08，已完成环境）

| 周次 | 任务 | 交付物 |
|------|------|--------|
| W1-W2 | ✅ conda 环境 + AgentScope 2.0.6 + Studio | 已就绪 |
| W1-W2 | 安装 reme-ai，跑通官方 ReMe 示例 | 记忆 demo |
| W2 | 理解 2.0 Middleware 机制（三个 hook） | 概念笔记 |
| W2 | 实现最简单的主副线智能体原型 | 原型 v0.1 |

### 阶段二：记忆层扩展实现（2026.08-09）

| 周次 | 任务 | 交付物 |
|------|------|--------|
| W3 | 双 workspace 隔离 + ReMe 自动写入验证 | 完整池基础版 |
| W3 | 向量化模块（BGE 模型）+ ReMe 向量检索开启 | 向量检索模块 |
| W4 | 共享记忆池实现（调度器输出容器） | SharedMemoryPool |
| W5 | 记忆层单元测试 + 性能测试 | 测试报告 |

### 阶段三：协调层核心算法实现（2026.09-10）

| 周次 | 任务 | 交付物 |
|------|------|--------|
| W6 | 四维注意力机制（时间/语义/频率/任务） | MultiHeadAttentionMemoryScorer |
| W7 | LLM 先验权重模块（Prompt + 多次采样） | LLM 权重生成 |
| W8 | 可学习权重模块 + 在线更新 | 可学习权重 |
| W8 | 混合权重计算与融合 | HybridWeightCalculator |
| W9 | 动作选择 + MemorySchedulingMiddleware 总装 | 调度中间件 v1.0 |

### 阶段四：智能体层与系统集成（2026.11-2027.01）

| 周次 | 任务 | 交付物 |
|------|------|--------|
| W10 | 主副线智能体 + 记忆总线 | MainAgent + SubAgent |
| W11 | 书签定位功能 | 书签管理器 |
| W11 | 前后端接口联调 | API 接口文档 |
| W12 | 端到端集成测试 | 系统 v0.9 |

### 阶段五：监控接入与实验框架（2026.12-2027.01，与阶段四并行）

| 周次 | 任务 | 交付物 |
|------|------|--------|
| W13 | TracingMiddleware 接入 + ReMe Job 监控 | 第 1、2 层数据 |
| W13 | 调度层埋点 + 指标字典落地 | schedule.jsonl |
| W14 | run_manifest.csv + 统计脚本 + 图表 | 自动化实验框架 |

### 阶段六：实验与优化（2027.02-03）

- 6 组消融实验（baseline → 完整方案，见监控方案第 9 章）
- 6 项验收指标对比（召回准确率/任务完成率/响应时间/调度延迟/Token/压缩比）
- 性能优化与参数调优

---

## 五、关键技术点与注意事项

### 5.1 AgentScope 2.0 扩展最佳实践

1. **继承而非修改**：所有扩展继承 `MiddlewareBase`，不修改框架源码，方便升级
2. **使用官方扩展点**：`on_reply` / `on_reasoning` / `on_system_prompt` 三个 hook 是调度算法的挂载点
3. **保持 Msg 兼容**：记忆注入统一用 `HintBlock`，与框架其他组件兼容
4. **配置集中管理**：阈值、权重、workspace 路径等参数放配置文件

### 5.2 性能优化要点

| 优化点 | 方法 | 预期效果 |
|--------|------|----------|
| 向量检索 | 开启 ReMe embedding_store + FAISS HNSW | 检索延迟 < 50ms |
| 注意力计算 | 批量向量化计算，避免 Python 循环 | 打分 < 10ms/条 |
| LLM 权重调用 | 缓存相似查询的权重结果 | LLM 调用减少 50%+ |
| 记忆池规模 | 调度层动作选择 + 定期清理 | 共享池 ≤ 10 条 |
| 埋点 I/O | 批量落盘（≥20 条一次）+ 只记统计摘要 | 不影响调度链路 |

### 5.3 实验数据采集（简要）

利用三层监控自动采集：

1. **时间指标**：总响应时间、检索耗时、调度决策延迟（目标 < 200ms）、LLM 调用耗时
2. **成本指标**：Token 消耗总量、LLM 调用次数、记忆池大小、共享记忆数量
3. **质量指标**：记忆召回准确率（人工标注）、任务完成率、用户满意度
4. **调度分析**：保留/丢弃比例、压缩比（目标 ≤ 0.3）、四维权重分布、权重变化

> 完整方案见《监控与实验数据采集方案.md》。

---

## 六、环境与依赖

### 6.1 核心依赖（srtp-memory 环境，已安装 ✅）

```
agentscope==2.0.6               # 核心框架（已装）
sentence-transformers==5.7.0    # 向量化（已装）
chromadb==1.5.9                 # 向量数据库（已装）
torch==2.13.0+cpu               # 可学习权重（已装）
scikit-learn==1.9.0             # 相似度计算（已装）
pandas==3.0.5 / numpy==2.5.1    # 数据处理（已装）
agentscope[memory-reme]         # ReMe 记忆底层（待安装 ⚠️）
@agentscope/studio              # 可视化（已装，as_studio 启动）
```

### 6.2 使用方式

```bash
# 激活项目环境
conda activate srtp-memory

# 安装 ReMe 记忆底层
pip install "agentscope[memory-reme]"

# 启动 Studio 可视化（可选）
as_studio

# 运行项目脚本
python main.py
```

---

## 七、风险与应对

| 风险 | 概率 | 应对策略 |
|------|------|----------|
| reme-ai 版本更新快（0.3.x 重构频繁） | 中 | 锁版本：`pip install reme-ai==<版本>` |
| ReMe search 是服务级 Job，粒度较粗 | 中 | 优先用 ReMeMiddleware 封装的 `_search`，必要时直接调 `run_job("search")` |
| 四维注意力效果不达预期 | 中 | 先单头验证再融合；RRF 分数作语义先验兜底；充分的消融实验 |
| LLM 先验权重不稳定 | 中 | 多次采样取平均 + 校验机制 + 可学习权重兜底 |
| 埋点影响性能 | 低 | 批量落盘 + 异步写 + 可开关 |
| 数据量不足导致统计不可靠 | 中 | 数据集 ≥ 100 用例；统一 run_manifest 登记 |

---

## 八、配套文档索引

| 文档 | 内容 | 关系 |
|------|------|------|
| 《记忆模块算法解析与调度重构方案.md》 | ReMe 深度解析、能力边界、重构方案 A/B | 本文档的记忆层与协调层依据 |
| 《监控与实验数据采集方案.md》 | 三层监控、指标字典、消融实验、统计脚本 | 本文档监控层与阶段六依据 |
| 《研究推进方案.md》 | 总体计划、里程碑、验收标准 | 本文档路线图依据 |
| 《AgentScope使用说明.md》 | 上手教程 | 其中 ReMe 章节需按 2.0 更新 |

---

**文档版本**：v2.0（自 v1.0 全面升级）
**制定日期**：2026年8月7日（v1.0）
**更新日期**：2026年8月12日（v2.0）
**主要变更**：适配 AgentScope 2.0.6 Middleware 架构；记忆层/协调层/监控层实现方案全面重写；环境与依赖更新为实际安装状态；整合 ReMe 深度解析与三层监控方案
**下一步**：安装 reme-ai → 跑通官方 ReMe 示例 → 实现主副线原型 v0.1
