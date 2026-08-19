# AgentScope 使用说明：多智能体记忆共享调度系统

> 面向课题：基于注意力引导和LLMs可学习权重的多智能体记忆共享调度方法研究  
> 适用人员：项目组全体成员  
> 本文档从 AgentScope 上手到深度定制，覆盖环境搭建、核心概念、智能体自定义、记忆模块替换与双池结构搭建。

---

## 目录

1. [AgentScope 简介](#1-agentscope-简介)
2. [环境搭建](#2-环境搭建)
3. [核心概念速览](#3-核心概念速览)
4. [最小可运行示例](#4-最小可运行示例)
5. [定制化：面向本研究](#5-定制化面向本研究)
6. [自定义智能体（MainAgent / SubAgent）](#6-自定义智能体mainagent--subagent)
7. [替换记忆模块：脱离 ReMe](#7-替换记忆模块脱离-reme)
8. [搭建双池结构](#8-搭建双池结构)
9. [记忆调度协调器](#9-记忆调度协调器)
10. [接入 AgentScope Studio 监控](#10-接入-agentscope-studio-监控)
11. [多模型配置](#11-多模型配置)
12. [项目工程结构建议](#12-项目工程结构建议)
13. [常见问题与排查](#13-常见问题与排查)

---

## 1. AgentScope 简介

### 1.1 什么是 AgentScope

AgentScope 是阿里云魔搭（ModelScope）社区开源的**企业级多智能体开发框架**，专为 LLM 驱动的多智能体系统研发而设计。其核心理念是"**透明可控、模块化可扩展、内置记忆组件、全链路可观测**"。

### 1.2 为什么选 AgentScope

| 优势 | 对课题的价值 |
|------|-------------|
| 全透明无黑盒 | 可完全干预每一步智能体行为，方便实验与消融分析 |
| 内置 ReMe 记忆组件 | 在此基础上扩展双池架构，节省基础开发量 |
| 原生消息调度引擎 | 主副线通信、状态同步由框架提供，专注调度逻辑 |
| 模型无关 | 一次编程适配 OpenAI / 通义千问 / Claude 等所有主流模型 |
| Studio 监控 | 自动采集行为轨迹、Token 消耗、响应时间等实验数据 |
| 继承式扩展 | 所有创新通过继承基类实现，不修改框架源码 |

### 1.3 官方资源

- **GitHub 仓库**：https://github.com/modelscope/agentscope
- **官方文档**：https://agentscope.io/
- **中文文档**：https://agentscope.readthedocs.io/zh-cn/latest/

---

## 2. 环境搭建

### 2.1 系统要求

- **Python**：3.10 或更高版本
- **内存**：建议 16GB 以上
- **GPU**：可选，有 GPU 时向量化和训练更快
- **LLM API Key**：至少准备一个（通义千问 / OpenAI）

### 2.2 安装 AgentScope

```bash
# 基础安装
pip install agentscope

# 含 Studio 完整安装（推荐）
pip install "agentscope[studio]"
```

### 2.3 安装项目核心依赖

```bash
# 向量化模型
pip install sentence-transformers

# 向量数据库
pip install chromadb

# 可学习权重模型训练
pip install torch

# 后端框架
pip install fastapi uvicorn

# 消息队列
pip install redis

# 实验工具
pip install pytest wandb
```

### 2.4 验证安装

```bash
python -c "import agentscope; print(agentscope.__version__)"
```

正常输出版本号即为安装成功。

### 2.5 配置 LLM API Key

AgentScope 支持通过环境变量或直接传参两种方式配置 API Key。

**方式一：环境变量**
```bash
# OpenAI
set OPENAI_API_KEY=your-api-key

# 通义千问（DashScope）
set DASHSCOPE_API_KEY=your-dashscope-key
```

**方式二：代码中直接传入**
```python
from agentscope.models import OpenAIWrapper

model = OpenAIWrapper(
    model_name="gpt-4o",
    api_key="sk-xxxxx",
)
```

> **建议**：开发阶段使用环境变量方式，避免将 API Key 写入代码。

---

## 3. 核心概念速览

### 3.1 概念映射表

理解 AgentScope 的核心概念与本课题的对应关系，是上手第一步。

| AgentScope 概念 | 本课题对应 | 关键 API 类 |
|---|---|---|
| `AgentBase` | 智能体基类 | 所有自定义智能体继承此类 |
| `DialogAgent` | 对话智能体 | 内置快速建成的对话 Agent |
| `Msg` | 消息 | 智能体间通信的唯一载体 |
| `ModelWrapper` | LLM 封装 | 统一所有大模型 API |
| `MemoryBase` / `ReMe` | 记忆组件 | 我们要替换并扩展的部分 |
| `Pipeline` | 管道编排 | 串联多智能体调用流程 |
| `Studio` | 监控平台 | 实验数据采集与可视化 |

### 3.2 Msg（消息）的结构

Msg 是 AgentScope 中**所有通信的基础单元**，主副线之间、Agent 与 LLM 之间，一切信息都通过 Msg 传递。

```python
from agentscope.message import Msg

msg = Msg(
    name="assistant",       # 消息发送者的名称
    role="assistant",        # 角色：user / assistant / system
    content="你好，有什么可以帮你？",  # 消息内容
    url=None,               # 可选：附件 URL
    meta={
        "timestamp": 1700000000,  # 可选：自定义元数据
        "memory_id": "mem_001",
    },
)
```

我们的记忆调度模块会大量使用 `msg.meta` 来传递记忆 ID、书签位置等自定义信息。

---

## 4. 最小可运行示例

### 4.1 单智能体对话

先把 AgentScope 跑通，验证环境没问题。

```python
from agentscope.agents import DialogAgent
from agentscope.message import Msg
from agentscope.models import DashScopeWrapper  # 通义千问

# 配置模型
model = DashScopeWrapper(
    model_name="qwen-max",
    api_key="your-dashscope-key",
)

# 创建智能体
agent = DialogAgent(
    name="测试助理",
    model_cfg=model,
    sys_prompt="你是一个友好的对话助手。",
)

# 发起对话
response = agent(Msg(
    name="user",
    role="user",
    content="你好，请介绍一下你自己",
))
print(response.content)
```

### 4.2 多智能体协作

跑通两个 Agent 之间的基本通信。

```python
from agentscope.agents import DialogAgent
from agentscope.message import Msg
from agentscope.pipelines import SequentialPipeline
from agentscope.models import DashScopeWrapper

model = DashScopeWrapper(
    model_name="qwen-max",
    api_key="your-dashscope-key",
)

# 两个智能体
planner = DialogAgent(
    name="规划员",
    model_cfg=model,
    sys_prompt="你负责制定任务规划，输出简洁的行动步骤。",
)

executor = DialogAgent(
    name="执行员",
    model_cfg=model,
    sys_prompt="你负责根据规划执行具体操作。",
)

# 用户输入
user_msg = Msg(name="user", role="user", content="帮我写一份项目周报")

# 使用管道串联
pipeline = SequentialPipeline([planner, executor])
result = pipeline(user_msg)
print(result[-1].content)  # 管道输出是消息列表，取最后一个
```

> **关键点**：`SequentialPipeline` 会自动将前一个 Agent 的输出作为下一个 Agent 的输入，这就是多智能体协作的最小形态。我们后续的主副线架构就是在这个基础上扩展的。

### 4.3 ReMe 内置记忆组件

看看 AgentScope 自带的记忆组件长什么样，了解它之后再替换。

```python
from agentscope.agents import DialogAgent
from agentscope.memory import ReMeModule
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.models import DashScopeWrapper

model = DashScopeWrapper(
    model_name="qwen-max",
    api_key="your-dashscope-key",
)

# InMemoryMemory：最简单的内置记忆，纯内存存储
memory = InMemoryMemory()

agent = DialogAgent(
    name="带记忆的助理",
    model_cfg=model,
    sys_prompt="你是一个有助手的助理，请记住用户说过的话。",
    memory=memory,  # 挂载记忆组件
)

# 第一轮对话
r1 = agent(Msg(name="user", role="user", content="我叫张三，我是学信息管理的"))
print(r1.content)

# 第二轮对话，Agent 能"记住"上一轮
r2 = agent(Msg(name="user", role="user", content="我叫什么名字？学什么的？"))
print(r2.content)
```

**核心认识**：`InMemoryMemory` 只是把对话历史存到内存里，没有向量化、没有语义检索——这远远不够我们的研究需求。下一部分开始做真正的替换。

---

## 5. 定制化：面向本研究

### 5.1 我们的定制策略

AgentScope 的扩展原则是：**继承而非修改**。所有创新点通过继承 AgentScope 的基类实现，绝不修改框架源码。

```
AgentScope 原生提供（直接用）：
├── AgentBase          → 智能体生命周期管理
├── ModelWrapper       → LLM API 统一封装
├── Msg                → 消息协议
├── Pipeline           → 消息流编排
└── Studio             → 监控与调试

我们扩展实现（核心创新）：
├── MainAgent / SubAgent          → 继承 AgentBase
├── FullContextMemoryPool         → 继承 MemoryBase，向量化存储
├── SharedMemoryPool              → 继承 MemoryBase，轻量共享
├── MemoryCoordinator             → 四维注意力 + 混合权重 + 动作选择
├── MemoryBus                     → 主副线记忆调度总线
└── BookmarkManager               → 书签定位与上下文锚点
```

### 5.2 定制开发规范

1. **所有自定义类放在独立模块中**，不与 AgentScope 的 import 混淆
2. **保持 Msg 格式兼容**，方便与 AgentScope 其他组件集成
3. **记忆存储格式与框架无关**，AgentScope 更新不影响我们的数据
4. **配置集中管理**，参数放在配置文件而非硬编码

---

## 6. 自定义智能体（MainAgent / SubAgent）

### 6.1 主线智能体（MainAgent）

主线智能体继承 `AgentBase`，负责推进核心任务，维护完整上下文记忆池。

```python
# agents/main_agent.py

import time
from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.memory import MemoryBase

from memory.full_context_memory_pool import FullContextMemoryPool
from memory.bookmark_manager import BookmarkManager


class MainAgent(AgentBase):
    """主线智能体：推进核心任务，维护完整上下文记忆池"""

    def __init__(
        self,
        name: str,
        model_config,
        memory_pool: FullContextMemoryPool,
        sys_prompt: str = "你是主线智能体，负责推进核心任务。",
    ):
        super().__init__(
            name=name,
            model_cfg=model_config,
            sys_prompt=sys_prompt,
        )
        self.memory_pool = memory_pool
        self.bookmark_manager = BookmarkManager()

    def reply(self, x: Msg = None) -> Msg:
        """主线回复流程：记录 → 推理 → 记录"""

        # 1. 将用户消息写入完整上下文记忆池
        self.memory_pool.add(x)

        # 2. 从记忆池获取完整上下文历史，构建 Prompt
        context_messages = self.memory_pool.get_full_context(
            role_filter=["user", "assistant"]
        )

        # 3. 调用 LLM 生成回复（使用 AgentScope 自带的 model 接口）
        response = self.model(context_messages)

        # 4. 将回复写入完整上下文记忆池
        reply_msg = Msg(
            name=self.name,
            role="assistant",
            content=response.content,
            meta={
                "timestamp": time.time(),
                "agent": "main",
            },
        )
        self.memory_pool.add(reply_msg)

        return reply_msg

    def create_bookmark(self, position_id: str, description: str) -> dict:
        """在指定位置创建书签，用于副线定位上下文"""
        return self.bookmark_manager.create(position_id, description)

    def get_context_at_bookmark(self, bookmark_id: str) -> list:
        """获取书签位置处的上下文快照"""
        return self.bookmark_manager.get_context(bookmark_id, self.memory_pool)
```

### 6.2 副线智能体（SubAgent）

副线智能体处理分支查询，从共享记忆池获取上下文。

```python
# agents/sub_agent.py

from agentscope.agents import AgentBase
from agentscope.message import Msg

from memory.shared_memory_pool import SharedMemoryPool


class SubAgent(AgentBase):
    """副线智能体：处理分支查询，从共享记忆池获取相关记忆"""

    def __init__(
        self,
        name: str,
        model_config,
        sys_prompt: str = "你是副线智能体，负责处理分支问题。",
    ):
        super().__init__(
            name=name,
            model_cfg=model_config,
            sys_prompt=sys_prompt,
        )
        self.shared_memory: SharedMemoryPool = None
        self.local_memory: list = []

    def set_shared_memory(self, shared_memory: SharedMemoryPool) -> None:
        """注入共享记忆池（由 MemoryCoordinator 调度后传入）"""
        self.shared_memory = shared_memory

    def reply(self, x: Msg = None) -> Msg:
        """副线回复流程：检索共享记忆 → 构建 Prompt → 生成回复"""

        # 1. 从共享记忆池中检索相关记忆
        relevant_memories = self.shared_memory.retrieve_relevant(x.content)

        # 2. 构建 Prompt：共享记忆 + 当前查询
        context_messages = self._build_context(relevant_memories, x)

        # 3. 调用 LLM
        response = self.model(context_messages)

        # 4. 记录副线本地交互
        self.local_memory.append(x)
        self.local_memory.append(
            Msg(name=self.name, role="assistant", content=response.content)
        )

        return response

    def _build_context(
        self, relevant_memories: list, user_query: Msg
    ) -> list:
        """将共享记忆和当前查询组装成 LLM 可理解的上下文"""
        context = [
            Msg(
                name="system",
                role="system",
                content=(
                    "以下是从主线对话中为你检索到的相关上下文，"
                    "请基于这些信息进行回答。"
                ),
            ),
        ]

        for mem in relevant_memories:
            context.append(
                Msg(
                    name=mem["role"],
                    role=mem["role"],
                    content=mem["content"],
                )
            )

        context.append(user_query)
        return context
```

### 6.3 智能体间通信：MemoryBus

记忆总线负责主副线之间的记忆调度，是 `MemoryCoordinator` 的使用者。

```python
# agents/memory_bus.py

from agentscope.message import Msg
from agentscope.agents import AgentBase

from memory.full_context_memory_pool import FullContextMemoryPool
from memory.shared_memory_pool import SharedMemoryPool
from coordinator.memory_coordinator import MemoryCoordinator


class MemoryBus:
    """记忆总线：主副线之间的记忆调度桥梁"""

    def __init__(
        self,
        main_agent: AgentBase,
        coordinator: MemoryCoordinator,
    ):
        self.main_agent = main_agent
        self.coordinator = coordinator

    def request_memory(
        self,
        query: str,
        bookmark_id: str = None,
        top_k: int = 50,
    ) -> SharedMemoryPool:
        """副线向主线请求记忆，返回调度后的共享记忆池"""

        # 1. 获取书签位置上下文（如果提供了书签）
        bookmark_context = {}
        if bookmark_id:
            bookmark_context = self.main_agent.get_context_at_bookmark(
                bookmark_id
            )

        # 2. 获取完整上下文记忆池
        candidate_pool = self.main_agent.memory_pool

        # 3. 调用记忆协调器进行调度
        shared_pool, schedule_result = self.coordinator.schedule(
            query=query,
            candidate_pool=candidate_pool,
            bookmark_context=bookmark_context,
            top_k=top_k,
        )

        return shared_pool, schedule_result

    def log_schedule(self, sub_agent_id: str, query: str, result: dict) -> None:
        """记录调度日志，用于后续实验分析"""
        # 实际实现中可对接数据库或日志系统
        pass
```

### 6.4 主副线协作最小示例

把上面的组件组装起来，跑通主副线交互。

```python
# main.py

from agentscope.models import DashScopeWrapper
from agentscope.message import Msg

from agents.main_agent import MainAgent
from agents.sub_agent import SubAgent
from agents.memory_bus import MemoryBus
from memory.full_context_memory_pool import FullContextMemoryPool
from coordinator.memory_coordinator import MemoryCoordinator

# 初始化模型
model = DashScopeWrapper(
    model_name="qwen-max",
    api_key="your-api-key",
)

# 初始化记忆池
memory_pool = FullContextMemoryPool()
coordinator = MemoryCoordinator()

# 初始化智能体
main_agent = MainAgent(name="主线", model_config=model, memory_pool=memory_pool)
sub_agent = SubAgent(name="副线", model_config=model)

# 初始化记忆总线
bus = MemoryBus(main_agent=main_agent, coordinator=coordinator)

# === 模拟场景 ===

# 用户在主线对话
main_agent.reply(Msg(name="user", role="user", content="我们要做一个多智能体项目"))
main_agent.reply(Msg(name="user", role="user", content="技术栈用 Python 和 AgentScope"))

# 用户创建一个书签
bookmark_id = main_agent.create_bookmark("bm_001", "项目技术选型讨论")

# 用户在副线提问
sub_query = Msg(name="user", role="user", content="我们的项目用什么框架？")

# 从主记忆池调度记忆
shared_pool, _ = bus.request_memory(
    query=sub_query.content,
    bookmark_id=bookmark_id,
    top_k=10,
)

# 副线使用调度后的记忆回答
sub_agent.set_shared_memory(shared_pool)
response = sub_agent.reply(sub_query)
print(response.content)
```

---

## 7. 替换记忆模块：脱离 ReMe

### 7.1 为什么替换 ReMe

AgentScope 内置的 ReMe 记忆组件虽然好用，但它的存储策略（纯文本对话历史）不满足我们的研究需求：

| ReMe 的问题 | 我们的需求 |
|---|---|
| 仅按对话顺序存储 | 需要向量化存储，支持语义检索 |
| 无多维度索引 | 需要时间戳、任务标签、访问频率等多维索引 |
| 无评分机制 | 需要重要性评分和注意力打分 |
| 单一记忆池 | 需要双池分离架构 |

### 7.2 MemoryBase 接口概览

AgentScope 的记忆组件继承自 `MemoryBase`，我们要实现它的核心接口。

```python
from agentscope.memory import MemoryBase

class MyCustomMemory(MemoryBase):
    """自定义记忆组件"""

    def __init__(self):
        super().__init__()

    def add(self, msg: Msg) -> None:
        """添加一条记忆"""
        ...

    def get_memory(self) -> list:
        """获取记忆列表，用于构建上下文"""
        ...

    def get_memory_by_role(self, role: str) -> list:
        """按角色过滤记忆"""
        ...

    def delete_memory(self, memory_id: str) -> None:
        """删除指定记忆"""
        ...
```

我们的 `FullContextMemoryPool` 和 `SharedMemoryPool` 都将继承 `MemoryBase` 并实现这些方法。

### 7.3 将自定义记忆挂载到 Agent

AgentScope 通过 `AgentBase.__init__` 的 `memory` 参数挂载记忆组件：

```python
from agentscope.agents import AgentBase
from agentscope.memory import MemoryBase

class MyAgent(AgentBase):
    def __init__(self, name, model_cfg, memory: MemoryBase):
        super().__init__(
            name=name,
            model_cfg=model_cfg,
            memory=memory,   # 传入自定义记忆
        )
        self.memory = memory  # 也可直接保存引用
```

> **注意**：如果不想使用 `AgentBase` 内置的 `memory` 参数，也可以完全绕过它，在自己的 `reply()` 方法里直接操作自定义记忆池。我们落地方案里的 `MainAgent` 就是这么做的——记忆池作为独立属性挂载，由 Agent 主动调用。

---

## 8. 搭建双池结构

### 8.1 双池架构设计

```
┌─────────────────────────────────────────────────┐
│              记忆层（双池结构）                     │
│                                                   │
│  ┌──────────────────────┐     ┌──────────────┐  │
│  │  完整上下文记忆池      │────►│  共享记忆池   │  │
│  │  FullContextPool     │     │  SharedPool  │  │
│  │                      │     │              │  │
│  │  • 全量存储           │     │  • 精选记忆   │  │
│  │  • 向量索引（Chroma） │     │  • 轻量索引   │  │
│  │  • 多维标签           │     │  • 低延迟     │  │
│  │  • 书签锚点           │     │  • 面向副线   │  │
│  └──────────────────────┘     └──────────────┘  │
│                                                   │
│  调度方向： 完整池 ──[MemoryCoordinator]──► 共享池  │
└─────────────────────────────────────────────────┘
```

### 8.2 完整上下文记忆池（FullContextMemoryPool）

完整池负责全量存储所有交互记忆，使用 Chroma 向量数据库 + HNSW 索引实现高效的语义检索。

```python
# memory/full_context_memory_pool.py

import time
import uuid
import chromadb
from agentscope.memory import MemoryBase
from agentscope.message import Msg
from sentence_transformers import SentenceTransformer


class FullContextMemoryPool(MemoryBase):
    """完整上下文记忆池：全量存储，向量化索引，多维检索"""

    def __init__(
        self,
        embed_model_name: str = "BAAI/bge-large-zh-v1.5",
        chroma_persist_dir: str = "./data/memory_db",
        embedding_dim: int = 1024,
    ):
        super().__init__()

        # 向量化模型
        self.embed_model = SentenceTransformer(embed_model_name)

        # Chroma 持久化客户端
        self.client = chromadb.PersistentClient(path=chroma_persist_dir)

        # 获取或创建集合
        try:
            self.collection = self.client.get_collection(
                name="full_context_memory"
            )
        except Exception:
            self.collection = self.client.create_collection(
                name="full_context_memory",
                metadata={"hnsw:space": "cosine"},
            )

        # 内存中的元数据索引（用于快速访问）
        self._meta_store: dict = {}

    def add(self, msg: Msg) -> str:
        """添加一条记忆，返回 memory_id"""

        memory_id = str(uuid.uuid4())[:8]
        content = msg.content

        # 向量化
        embedding = self.embed_model.encode(content).tolist()

        # 存储元数据
        self._meta_store[memory_id] = {
            "id": memory_id,
            "content": content,
            "role": msg.role,
            "name": msg.name,
            "timestamp": time.time(),
            "access_count": 0,
            "task_tag": msg.meta.get("task_tag", None) if msg.meta else None,
            "importance_score": 0.5,
            "embedding": embedding,
            "meta": msg.meta or {},
        }

        # 存入向量数据库
        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            metadatas=[{
                "role": msg.role,
                "timestamp": self._meta_store[memory_id]["timestamp"],
            }],
        )

        return memory_id

    def get_candidates(self, query: str, top_k: int = 50) -> list:
        """语义初筛：返回与查询最相关的 top_k 条记忆"""

        query_embedding = self.embed_model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        candidates = []
        for mem_id in results["ids"][0]:
            memory = self._meta_store[mem_id]
            memory["access_count"] += 1  # 更新访问计数
            candidates.append(memory)

        return candidates

    def get_full_context(
        self,
        role_filter: list = None,
        limit: int = None,
    ) -> list:
        """获取完整的上下文消息列表（按时间排序）"""

        memories = list(self._meta_store.values())

        # 按时间排序
        memories.sort(key=lambda m: m["timestamp"])

        # 角色过滤
        if role_filter:
            memories = [m for m in memories if m["role"] in role_filter]

        # 数量限制
        if limit:
            memories = memories[-limit:]

        # 转换为 Msg 对象
        return [
            Msg(
                name=m["name"],
                role=m["role"],
                content=m["content"],
            )
            for m in memories
        ]

    def get_memory(self) -> list:
        """MemoryBase 接口：获取全部记忆"""
        return self.get_full_context()

    def get_memory_by_role(self, role: str) -> list:
        """MemoryBase 接口：按角色获取记忆"""
        return self.get_full_context(role_filter=[role])

    def delete_memory(self, memory_id: str) -> None:
        """删除指定记忆"""
        if memory_id in self.collection.get(ids=[memory_id])["ids"]:
            self.collection.delete(ids=[memory_id])
        self._meta_store.pop(memory_id, None)

    def get_all_memories(self) -> list:
        """获取所有记忆的元数据"""
        return list(self._meta_store.values())

    def get_total_count(self) -> int:
        """获取记忆总数"""
        return len(self._meta_store)
```

### 8.3 共享记忆池（SharedMemoryPool）

共享池存储调度后选中的记忆，数量少、速度快，面向副线智能体。

```python
# memory/shared_memory_pool.py

from agentscope.memory import MemoryBase
from agentscope.message import Msg
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SharedMemoryPool(MemoryBase):
    """共享记忆池：轻量级，存储调度后的精选记忆"""

    def __init__(self, memories: list = None, target_agent: str = None):
        super().__init__()
        self.memories = memories or []
        self.target_agent = target_agent

        # 轻量级向量化（共享池数据量小，可以延迟计算）
        self._embeddings_cache = {}
        self.embed_model = None

    def set_memories(self, memories: list) -> None:
        """设置共享记忆内容"""
        self.memories = memories
        self._embeddings_cache = {}

    def _ensure_embed_model(self):
        """懒加载向量化模型"""
        if self.embed_model is None:
            self.embed_model = SentenceTransformer("BAAI/bge-large-zh-v1.5")

    def _get_embedding(self, content: str) -> list:
        """获取文本向量（带缓存）"""
        if content not in self._embeddings_cache:
            self._ensure_embed_model()
            self._embeddings_cache[content] = self.embed_model.encode(
                content
            ).tolist()
        return self._embeddings_cache[content]

    def retrieve_relevant(
        self,
        query: str,
        top_k: int = 10,
    ) -> list:
        """检索与查询最相关的 top_k 条记忆"""

        if not self.memories:
            return []

        query_emb = self._get_embedding(query)

        # 计算每条记忆与查询的余弦相似度
        scored = []
        for mem in self.memories:
            mem_emb = self._get_embedding(mem["content"])
            sim = cosine_similarity(
                [query_emb], [mem_emb]
            )[0][0]
            scored.append((sim, mem))

        # 按相似度降序排列
        scored.sort(key=lambda x: x[0], reverse=True)

        return [mem for _, mem in scored[:top_k]]

    def get_memory(self) -> list:
        """MemoryBase 接口"""
        return [
            Msg(
                name=m.get("name", "shared"),
                role=m.get("role", "user"),
                content=m["content"],
            )
            for m in self.memories
        ]

    def get_memory_by_role(self, role: str) -> list:
        """MemoryBase 接口"""
        return [
            m for m in self.get_memory() if m.role == role
        ]

    def delete_memory(self, memory_id: str) -> None:
        """共享池的删除操作"""
        self.memories = [m for m in self.memories if m["id"] != memory_id]

    def size(self) -> int:
        """共享记忆池的大小"""
        return len(self.memories)

    def get_all_memories(self) -> list:
        return list(self.memories)
```

### 8.4 书签管理器（BookmarkManager）

书签是主副线切换的定位锚点，是研究方案中的关键创新之一。

```python
# memory/bookmark_manager.py

from dataclasses import dataclass, field
import time


@dataclass
class Bookmark:
    """书签数据结构"""
    bookmark_id: str
    position_id: str
    description: str
    created_at: float
    context_snapshot: list = field(default_factory=list)


class BookmarkManager:
    """书签管理器：管理主线对话中的书签标记"""

    def __init__(self):
        self._bookmarks: dict = {}

    def create(
        self,
        position_id: str,
        description: str,
        bookmark_id: str = None,
    ) -> Bookmark:
        """创建一个书签"""

        if bookmark_id is None:
            bookmark_id = f"bm_{len(self._bookmarks) + 1:04d}"

        bookmark = Bookmark(
            bookmark_id=bookmark_id,
            position_id=position_id,
            description=description,
            created_at=time.time(),
        )

        self._bookmarks[bookmark_id] = bookmark
        return bookmark

    def get_context(self, bookmark_id: str, memory_pool) -> dict:
        """根据书签获取上下文快照"""

        if bookmark_id not in self._bookmarks:
            return {}

        bookmark = self._bookmarks[bookmark_id]

        # 从记忆池中获取书签位置附近的上下文
        context = memory_pool.get_full_context(
            role_filter=["user", "assistant"],
            limit=20,
        )

        return {
            "bookmark_id": bookmark_id,
            "position_id": bookmark.position_id,
            "description": bookmark.description,
            "context": context,
        }

    def list_all(self) -> list:
        """列出所有书签"""
        return list(self._bookmarks.values())

    def delete(self, bookmark_id: str) -> bool:
        """删除书签"""
        if bookmark_id in self._bookmarks:
            del self._bookmarks[bookmark_id]
            return True
        return False
```

### 8.5 双池数据流转示例

完整演示数据如何从完整池流向共享池。

```python
from memory.full_context_memory_pool import FullContextMemoryPool
from memory.shared_memory_pool import SharedMemoryPool
from memory.bookmark_manager import BookmarkManager
from agentscope.message import Msg

# 初始化完整上下文记忆池
pool = FullContextMemoryPool()

# 模拟主线交互（写入完整池）
m1 = pool.add(Msg(name="user", role="user", content="我们需要搭建多智能体系统"))
m2 = pool.add(Msg(name="assistant", role="assistant", content="好的，我建议用 AgentScope 框架"))
m3 = pool.add(Msg(name="user", role="user", content="记忆模块要支持双池架构"))
m4 = pool.add(Msg(name="assistant", role="assistant", content="双池架构分为完整上下文池和共享池"))

# 创建书签
bm = BookmarkManager()
bm.create("position_m3", "讨论记忆架构时")

# 语义检索候选记忆
candidates = pool.get_candidates("我们用什么框架搭建系统？", top_k=4)

# 将候选记忆放入共享池（模拟调度后的结果）
shared_pool = SharedMemoryPool(memories=candidates, target_agent="sub_agent_1")

# 副线检索
relevant = shared_pool.retrieve_relevant("系统架构用什么框架？", top_k=2)
for r in relevant:
    print(f"相关性记忆: {r['content']}")
```

---

## 9. 记忆调度协调器

### 9.1 协调器职责

协调层是核心创新所在，包含三个模块串联工作：

```
MemoryCoordinator
│
├── HybridWeightCalculator   → 计算四维注意力混合权重
├── MultiHeadAttentionScorer → 对每条记忆做多维打分
└── ActionSelector           → 决定保留或丢弃
```

### 9.2 四维注意力打分器

```python
# coordinator/attention_scorer.py

import math
from sklearn.metrics.pairwise import cosine_similarity


class MultiHeadAttentionMemoryScorer:
    """四维注意力记忆打分模块"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.time_decay_rate = self.config.get("time_decay_rate", 0.01)
        self.frequency_factor = self.config.get("frequency_factor", 0.1)
        self.head_weights = self.config.get("head_weights", {
            "time": 0.25,
            "semantic": 0.35,
            "frequency": 0.15,
            "task": 0.25,
        })

    def score(
        self,
        query_embedding: list,
        memory: dict,
        current_time: float,
        task_context: dict = None,
    ) -> dict:
        """对单条记忆计算四维注意力得分"""

        # 1. 时间注意力
        time_score = self._time_attention(
            memory["timestamp"], current_time
        )

        # 2. 语义注意力
        semantic_score = self._semantic_attention(
            query_embedding, memory["embedding"]
        )

        # 3. 频率注意力
        frequency_score = self._frequency_attention(
            memory["access_count"]
        )

        # 4. 任务注意力
        task_score = self._task_attention(task_context, memory)

        # 5. 多头融合
        final_score = (
            self.head_weights["time"] * time_score
            + self.head_weights["semantic"] * semantic_score
            + self.head_weights["frequency"] * frequency_score
            + self.head_weights["task"] * task_score
        )

        return {
            "final_score": final_score,
            "time_score": time_score,
            "semantic_score": semantic_score,
            "frequency_score": frequency_score,
            "task_score": task_score,
        }

    def _time_attention(self, mem_timestamp: float, current_time: float) -> float:
        """时间衰减注意力"""
        time_diff = (current_time - mem_timestamp) / 3600.0  # 换算为小时
        return math.exp(-self.time_decay_rate * time_diff)

    def _semantic_attention(self, query_emb: list, mem_emb: list) -> float:
        """语义注意力：余弦相似度"""
        return float(cosine_similarity([query_emb], [mem_emb])[0][0])

    def _frequency_attention(self, access_count: int) -> float:
        """频率注意力：访问频率越高，分数越高"""
        return 1 - math.exp(-self.frequency_factor * access_count)

    def _task_attention(self, task_context: dict, memory: dict) -> float:
        """任务注意力：任务标签匹配"""
        if task_context is None:
            return 0.5
        if memory.get("task_tag") == task_context.get("task_type"):
            return 1.0
        return 0.3

    def set_head_weights(self, weights: dict) -> None:
        """动态更新注意力头权重（由 HybridWeightCalculator 提供）"""
        self.head_weights = weights
```

### 9.3 混合权重计算器

```python
# coordinator/hybrid_weight_calculator.py

import json
import numpy as np


class HybridWeightCalculator:
    """混合权重计算器：LLM 先验 + 可学习权重"""

    def __init__(
        self,
        llm_client,
        embed_model,
        alpha: float = 0.6,
        beta: float = 0.4,
        llm_sample_times: int = 3,
    ):
        self.llm = llm_client
        self.embed_model = embed_model
        self.alpha = alpha
        self.beta = beta
        self.llm_sample_times = llm_sample_times

        # 可学习权重（初始化为均匀权重）
        self.learnable_weights = {
            "time": 0.25,
            "semantic": 0.25,
            "frequency": 0.25,
            "task": 0.25,
        }

    def calculate_weights(
        self,
        query: str,
        context: dict,
    ) -> dict:
        """计算四维注意力的混合权重"""

        # 1. LLM 先验权重
        llm_weights = self._get_llm_prior_weights(query, context)

        # 2. 混合计算
        hybrid = {}
        for key in ["time", "semantic", "frequency", "task"]:
            hybrid[key] = (
                self.alpha * llm_weights[key]
                + self.beta * self.learnable_weights[key]
            )

        # 归一化
        total = sum(hybrid.values())
        return {k: v / total for k, v in hybrid.items()}

    def _get_llm_prior_weights(self, query: str, context: dict) -> dict:
        """通过 LLM 多次采样获取先验权重"""

        weights_list = []
        for _ in range(self.llm_sample_times):
            prompt = self._build_weight_prompt(query, context)
            response = self.llm(prompt)
            weights = self._parse_weight_response(response)
            weights_list.append(weights)

        # 取平均
        avg = {}
        for key in ["time", "semantic", "frequency", "task"]:
            avg[key] = np.mean([w[key] for w in weights_list])

        # 归一化
        total = sum(avg.values())
        return {k: v / total for k, v in avg.items()}

    def _build_weight_prompt(self, query: str, context: dict) -> str:
        """构建 LLM 权重判断 Prompt"""
        return f"""你是记忆调度系统的权重评估专家。

当前查询：{query}
上下文信息：{json.dumps(context, ensure_ascii=False)}

请为以下四个注意力维度分配权重，要求总和为 1.0：
- time（时间相关性）：当前查询与近期记忆的相关程度
- semantic（语义相关性）：查询与记忆的语义相似度重要性
- frequency（频率相关性）：记忆被访问频率的重要性
- task（任务相关性）：记忆与当前任务标签的匹配度

请只输出 JSON 格式：{{"time": 0.x, "semantic": 0.x, "frequency": 0.x, "task": 0.x}}"""

    def _parse_weight_response(self, response) -> dict:
        """解析 LLM 返回的权重 JSON"""
        try:
            content = response.content if hasattr(response, "content") else str(response)
            data = json.loads(content.strip().strip("```").replace("json", ""))
            return {k: float(v) for k, v in data.items()}
        except Exception:
            return {"time": 0.25, "semantic": 0.25, "frequency": 0.25, "task": 0.25}

    def update_with_feedback(
        self,
        query: str,
        feedback_score: float,
        learning_rate: float = 0.01,
    ) -> None:
        """根据用户反馈更新可学习权重"""
        # 简单示例：反馈好则加强语义权重，反馈差则加强时间权重
        if feedback_score > 0.5:
            self.learnable_weights["semantic"] += learning_rate
        else:
            self.learnable_weights["time"] += learning_rate

        # 归一化
        total = sum(self.learnable_weights.values())
        for k in self.learnable_weights:
            self.learnable_weights[k] = max(
                0.0, min(1.0, self.learnable_weights[k] / total)
            )
```

### 9.4 动作选择器

```python
# coordinator/action_selector.py


class ActionSelector:
    """动作选择器：基于得分决定保留或丢弃记忆"""

    def __init__(self, threshold: float = 0.6, max_shared: int = 10):
        self.threshold = threshold
        self.max_shared = max_shared

    def select(self, scored_memories: list) -> dict:
        """选择保留和丢弃的记忆"""

        kept = []
        discarded = []

        for mem in scored_memories:
            final_score = mem["attention_score"]["final_score"]
            if final_score >= self.threshold:
                kept.append(mem)
            else:
                discarded.append(mem)

        # 按得分降序排列
        kept.sort(key=lambda x: x["attention_score"]["final_score"], reverse=True)

        # 限制最大数量
        kept = kept[:self.max_shared]

        return {
            "kept": kept,
            "discarded": discarded,
        }
```

### 9.5 记忆协调器总入口

```python
# coordinator/memory_coordinator.py

import time
from sentence_transformers import SentenceTransformer

from coordinator.attention_scorer import MultiHeadAttentionMemoryScorer
from coordinator.hybrid_weight_calculator import HybridWeightCalculator
from coordinator.action_selector import ActionSelector
from memory.shared_memory_pool import SharedMemoryPool


class MemoryCoordinator:
    """记忆协调器：记忆调度总入口"""

    def __init__(
        self,
        embed_model_name: str = "BAAI/bge-large-zh-v1.5",
        threshold: float = 0.6,
        max_shared: int = 10,
        llm_client=None,
    ):
        self.embed_model = SentenceTransformer(embed_model_name)

        self.attention_scorer = MultiHeadAttentionMemoryScorer()
        self.weight_calculator = HybridWeightCalculator(
            llm_client=llm_client,
            embed_model=self.embed_model,
        )
        self.action_selector = ActionSelector(
            threshold=threshold,
            max_shared=max_shared,
        )

    def schedule(
        self,
        query: str,
        candidate_pool,
        bookmark_context: dict = None,
        top_k: int = 50,
    ) -> tuple:
        """执行完整记忆调度流程

        Returns:
            (SharedMemoryPool, dict) — 调度后的共享记忆池与调度结果
        """

        current_time = time.time()

        # 1. 从完整记忆池检索候选（语义初筛）
        candidates = candidate_pool.get_candidates(query, top_k=top_k)

        if not candidates:
            return SharedMemoryPool([], target_agent="unknown"), {
                "status": "empty",
                "message": "未找到候选记忆",
            }

        # 2. 查询向量化
        query_embedding = self.embed_model.encode(query).tolist()

        # 3. 计算混合权重
        weights = self.weight_calculator.calculate_weights(
            query=query,
            context=bookmark_context or {},
        )
        self.attention_scorer.set_head_weights(weights)

        # 4. 对每条记忆打分
        task_context = (
            bookmark_context.get("task", {})
            if bookmark_context
            else None
        )

        scored_memories = []
        for mem in candidates:
            scores = self.attention_scorer.score(
                query_embedding=query_embedding,
                memory=mem,
                current_time=current_time,
                task_context=task_context,
            )
            mem["attention_score"] = scores
            scored_memories.append(mem)

        # 5. 动作选择
        result = self.action_selector.select(scored_memories)

        # 6. 构建共享记忆池
        shared_pool = SharedMemoryPool(
            memories=result["kept"],
            target_agent="sub_agent",
        )

        # 7. 记录调度结果
        schedule_result = {
            "status": "ok",
            "kept_count": len(result["kept"]),
            "discarded_count": len(result["discarded"]),
            "weights": weights,
            "threshold": self.action_selector.threshold,
        }

        return shared_pool, schedule_result
```

---

## 10. 接入 AgentScope Studio 监控

AgentScope Studio 提供全链路可视化监控，自动采集实验数据。

### 10.1 启动 Studio

在项目中启用 Studio，启动时会自动在本地启动监控服务。

```python
import agentscope

# 初始化 AgentScope，启动 Studio 服务
agentscope.init(
    project="multi-agent-memory-scheduling",
    studio_kwargs={
        "host": "0.0.0.0",
        "port": 5679,
    },
)
```

### 10.2 Studio 自动采集的数据

启动后，Studio 会自动记录以下信息：

| 数据类别 | 内容 | 实验用途 |
|---|---|---|
| 行为轨迹 | 每个 Agent 的每次调用过程 | 分析调度决策过程 |
| 消息流 | Agent 之间的消息传递 | 分析通信效率 |
| LLM 调用 | 每次模型调用的 prompt / response | Token 消耗统计 |
| 响应时间 | 每次调用的耗时 | 效率评估 |
| 错误日志 | 异常捕获与堆栈 | 故障排查 |

### 10.3 自定义指标埋点

对于研究中需要额外追踪的指标，可以通过 `agentscope.util.tracer` 添加自定义埋点。

```python
from agentscope.tracing import trace  # 视版本可能 API 不同

# 在关键路径添加埋点
@trace("memory_retrieval")
def retrieve_memory(self, query: str) -> list:
    start = time.time()
    # ... 检索逻辑 ...
    elapsed = time.time() - start
    # 记录自定义指标
    return results


# 手动记录调度统计
def record_schedule_stats(
    self,
    kept_count: int,
    discarded_count: int,
    retrieval_time_ms: float,
) -> None:
    """记录调度统计数据"""
    stats = {
        "kept": kept_count,
        "discarded": discarded_count,
        "retrieval_time_ms": retrieval_time_ms,
        "timestamp": time.time(),
    }
    # 存储到文件或数据库
    pass
```

### 10.4 访问 Studio 界面

启动后访问：`http://localhost:5679`

在界面中可以查看：
- 每次对话的完整执行轨迹
- Agent 之间的消息流图
- LLM 调用的 prompt 与 response
- 响应时间与 Token 消耗统计

---

## 11. 多模型配置

AgentScope 是模型无关的，一套代码可以适配多个大模型。以下是本项目最常用的两种配置。

### 11.1 通义千问（DashScope）

```python
from agentscope.models import DashScopeWrapper

model = DashScopeWrapper(
    model_name="qwen-max",
    api_key="your-dashscope-key",
)
```

### 11.2 OpenAI

```python
from agentscope.models import OpenAIWrapper

model = OpenAIWrapper(
    model_name="gpt-4o",
    api_key="your-openai-key",
)
```

### 11.3 切换模型（对比实验用）

把模型配置抽离为配置文件，方便对比实验。

```python
# config/models.yaml
models:
  default:
    provider: dashscope
    model_name: qwen-max
    api_key_env: DASHSCOPE_API_KEY

  openai:
    provider: openai
    model_name: gpt-4o
    api_key_env: OPENAI_API_KEY
```

```python
# config/model_loader.py

import os
import yaml
from agentscope.models import DashScopeWrapper, OpenAIWrapper


def load_model(config_name: str = "default") -> object:
    with open("config/models.yaml", "r") as f:
        all_configs = yaml.safe_load(f)

    cfg = all_configs["models"][config_name]
    api_key = os.environ.get(cfg["api_key_env"])

    if cfg["provider"] == "dashscope":
        return DashScopeWrapper(model_name=cfg["model_name"], api_key=api_key)
    elif cfg["provider"] == "openai":
        return OpenAIWrapper(model_name=cfg["model_name"], api_key=api_key)
```

---

## 12. 项目工程结构建议

参考落地方案文档，推荐以下工程结构：

```
SRTP_Multi_Agent_Memory/
├── agents/                        # 智能体层
│   ├── __init__.py
│   ├── main_agent.py              # 主线智能体
│   ├── sub_agent.py               # 副线智能体
│   └── memory_bus.py              # 记忆总线
│
├── memory/                        # 记忆层
│   ├── __init__.py
│   ├── full_context_memory_pool.py # 完整上下文记忆池
│   ├── shared_memory_pool.py      # 共享记忆池
│   └── bookmark_manager.py        # 书签管理器
│
├── coordinator/                   # 协调层（核心创新）
│   ├── __init__.py
│   ├── memory_coordinator.py      # 记忆协调器总入口
│   ├── attention_scorer.py        # 四维注意力打分
│   ├── hybrid_weight_calculator.py # 混合权重计算
│   └── action_selector.py         # 动作选择器
│
├── api/                           # API 层
│   ├── __init__.py
│   └── routes.py                  # FastAPI 路由
│
├── config/                        # 配置文件
│   ├── __init__.py
│   ├── models.yaml                # 模型配置
│   └── model_loader.py            # 模型加载器
│
├── data/                          # 数据目录
│   └── memory_db/                 # Chroma 持久化存储
│
├── tests/                         # 测试
│   ├── test_memory_pool.py
│   ├── test_attention_scorer.py
│   ├── test_coordinator.py
│   └── test_agents.py
│
├── docs/                          # 文档
│   ├── AgentScope使用说明.md       ← 本文档
│   ├── AgentScope基座系统落地方案.md
│   └── 研究推进方案.md
│
├── main.py                        # 入口文件
├── requirements.txt               # Python 依赖
└── README.md                      # 项目说明
```

---

## 13. 常见问题与排查

### Q1: 安装 `sentence-transformers` 时报错？

```bash
# 确保 torch 已安装且版本兼容
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 再安装 sentence-transformers
pip install sentence-transformers
```

### Q2: Chroma 持久化存储找不到数据库文件？

确保 `chroma_persist_dir` 指向的路径存在且可写：

```python
import os
os.makedirs("./data/memory_db", exist_ok=True)
```

### Q3: LLM 返回的权重 JSON 解析失败？

```python
# 检查 LLM 实际返回的内容
print(repr(response.content))

# 常见原因：返回了代码块包裹的 JSON
# 修复：在解析前去除 ```json ... ``` 包裹
```

### Q4: Agent 的 `reply()` 方法报错 "model is not callable"？

确保 `AgentBase` 初始化时正确传入了 `model_cfg`：

```python
class MyAgent(AgentBase):
    def __init__(self, name, model_cfg):
        super().__init__(name=name, model_cfg=model_cfg)
        # model_cfg 会自动绑定为 self.model
```

### Q5: 向量化模型下载很慢？

```python
# 设置 HuggingFace 镜像
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 或手动下载到本地后加载
model = SentenceTransformer("./models/bge-large-zh-v1.5")
```

### Q6: Studio 启动后访问不到页面？

```python
# 确认端口没有被占用
# 修改端口号
agentscope.init(
    studio_kwargs={
        "port": 5680,  # 换一个端口
    },
)
```

### Q7: 主副线消息中记忆丢失？

确保 `Msg.meta` 中传递的记忆 ID 在下游 Agent 中正确解析：

```python
# 发送方
msg = Msg(
    name="main",
    role="assistant",
    content="共享记忆已准备",
    meta={"memory_ids": ["m1", "m2"], "bookmark_id": "bm_001"},
)

# 接收方
def reply(self, x: Msg = None) -> Msg:
    memory_ids = x.meta.get("memory_ids", [])
    bookmark_id = x.meta.get("bookmark_id")
    # 根据 memory_ids 从共享池获取具体记忆内容
```

---

## 附录：快速参考

### A. 核心类速查

| 类名 | 文件 | 作用 |
|------|------|------|
| `MainAgent` | `agents/main_agent.py` | 主线智能体 |
| `SubAgent` | `agents/sub_agent.py` | 副线智能体 |
| `MemoryBus` | `agents/memory_bus.py` | 主副线记忆总线 |
| `FullContextMemoryPool` | `memory/full_context_memory_pool.py` | 完整上下文记忆池 |
| `SharedMemoryPool` | `memory/shared_memory_pool.py` | 共享记忆池 |
| `BookmarkManager` | `memory/bookmark_manager.py` | 书签管理器 |
| `MemoryCoordinator` | `coordinator/memory_coordinator.py` | 记忆协调器 |
| `MultiHeadAttentionMemoryScorer` | `coordinator/attention_scorer.py` | 四维注意力打分 |
| `HybridWeightCalculator` | `coordinator/hybrid_weight_calculator.py` | 混合权重计算 |
| `ActionSelector` | `coordinator/action_selector.py` | 动作选择器 |

### B. 核心依赖清单

```txt
agentscope>=1.0.0
sentence-transformers
chromadb
torch
scikit-learn
fastapi
uvicorn
redis
pytest
pyyaml
```

### C. 模型下载命令（可选，离线使用）

```bash
# 下载 BGE 中文向量模型
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-zh-v1.5')"
```

---

> 本文档版本：v1.0  
> 更新日期：2026-08-08  
> 适用框架版本：AgentScope >= 1.0.0  
> 维护团队：SRTP 多智能体记忆共享项目组
